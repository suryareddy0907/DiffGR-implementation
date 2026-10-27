import os
import sys
import math
import random
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load Pinterest data
from scripts.test_pinterest_eval import data, RecDataLoader, compute_metrics, R, C_norm

# Test on 3000 train users and 1000 eval users for fast comparison
eval_users = [u for u in range(1000) if len(data.val_dict[u]) > 0]
train_users = list(range(3000))

class FastDiffGR(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=128, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.user_embeddings = nn.Embedding(n_users, emb_dim)
        self.item_embeddings = nn.Embedding(n_items, emb_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        
        self.user_merger = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.w = nn.Parameter(torch.ones(emb_dim))
        
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, u_ids, gt):
        emb_u = self.user_embeddings(u_ids)
        emb_gt = self.encoder(gt)
        emb_final_u = self.user_merger(torch.cat([emb_u, emb_gt], dim=-1))
        scores = torch.matmul(emb_final_u * self.w, self.item_embeddings.weight.T)
        return scores

# Compare Loss 1: Current vs Loss 2: BPR vs Loss 3: Softmax with Temperature & Negatives
for loss_type in ['current', 'bpr', 'infonce']:
    torch.manual_seed(42)
    np.random.seed(42)
    model = FastDiffGR(data.n_users, data.n_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"\n--- Testing Loss: {loss_type.upper()} ---")
    
    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        batches = 0
        
        for b_start in range(0, len(train_users), 256):
            b_users = train_users[b_start:b_start+256]
            u_tensor = torch.tensor(b_users, dtype=torch.long, device=device)
            g0 = torch.zeros(len(b_users), data.n_items, device=device)
            for i, u in enumerate(b_users):
                items = data.train_dict[u]
                if len(items) > 0:
                    g0[i, items] = 1.0
            
            # Simple noise
            gt = g0 * (torch.rand_like(g0) > 0.2).float()
            scores = model(u_tensor, gt)
            
            if loss_type == 'current':
                ce = F.binary_cross_entropy_with_logits(scores, g0, pos_weight=torch.tensor([20.0]))
                log_sm = F.log_softmax(scores, dim=-1)
                p_tgt = g0 / g0.sum(dim=-1, keepdim=True).clamp(min=1.0)
                rank = -(log_sm * p_tgt).sum(dim=-1).mean()
                loss = ce + 2.5 * rank
            elif loss_type == 'bpr':
                # BPR loss with negative sampling
                b_size = len(b_users)
                pos_items = [random.choice(data.train_dict[u]) if len(data.train_dict[u]) > 0 else 0 for u in b_users]
                neg_items = [random.randint(0, data.n_items - 1) for _ in b_users]
                
                pos_sc = scores[torch.arange(b_size), pos_items]
                neg_sc = scores[torch.arange(b_size), neg_items]
                loss = -F.logsigmoid(pos_sc - neg_sc).mean()
            elif loss_type == 'infonce':
                # InfoNCE with temperature 0.1 on positives
                tau = 0.1
                log_sm = F.log_softmax(scores / tau, dim=-1)
                p_tgt = g0 / g0.sum(dim=-1, keepdim=True).clamp(min=1.0)
                loss = -(log_sm * p_tgt).sum(dim=-1).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        
        # Eval on Val
        model.eval()
        with torch.no_grad():
            u_t = torch.tensor(eval_users, dtype=torch.long, device=device)
            gt_val = torch.zeros(len(eval_users), data.n_items, device=device)
            for i, u in enumerate(eval_users):
                itms = data.train_dict.get(u, [])
                if len(itms) > 0:
                    gt_val[i, itms] = 1.0
            sc = model(u_t, gt_val).cpu().numpy()
            r, n = compute_metrics(sc, data.val_dict, eval_users, data.train_dict, data.val_dict, split='val')
            print(f"Epoch {epoch} | Loss: {total_loss/batches:.3f} | Val R@20: {r:.4f} | Val N@20: {n:.4f}")
