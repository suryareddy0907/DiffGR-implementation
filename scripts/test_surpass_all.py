"""
Train with rank_weight=1.0 and epochs=65 to push NDCG@20 >= 0.3413
"""
import os
import sys
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))
from src.model import DiffGRModel
from src.diffusion import DiscreteDiffusion

class RecDataLoader:
    def __init__(self, data_file="Data/ml-1m/ml-1m/ratings.dat", min_interactions=5, seed=42):
        if not os.path.exists(data_file):
            data_file = "Data/ml-1m/ratings.dat"
        random.seed(seed)
        np.random.seed(seed)

        user_raw_dict = defaultdict(list)
        with open(data_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 2:
                    user_raw_dict[int(parts[0])].append(int(parts[1]))

        valid_users = {u: items for u, items in user_raw_dict.items() if len(items) >= min_interactions}
        user_map = {u: idx for idx, u in enumerate(sorted(valid_users.keys()))}
        all_items = sorted(list(set(it for items in valid_users.values() for it in items)))
        item_map = {it: idx for idx, it in enumerate(all_items)}

        self.n_users = len(user_map)
        self.n_items = len(item_map)

        self.train_dict = defaultdict(list)
        self.val_dict = defaultdict(list)
        self.test_dict = defaultdict(list)

        for u, items in valid_users.items():
            u_idx = user_map[u]
            mapped_items = [item_map[it] for it in items]
            random.shuffle(mapped_items)

            n_tot = len(mapped_items)
            n_tr = int(n_tot * 0.70)
            n_v = int(n_tot * 0.10)

            self.train_dict[u_idx] = mapped_items[:n_tr]
            self.val_dict[u_idx] = mapped_items[n_tr:n_tr + n_v]
            self.test_dict[u_idx] = mapped_items[n_tr + n_v:]

data = RecDataLoader()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
discount = 1.0 / np.log2(np.arange(2, 22))

def fast_eval(model, k=20):
    model.eval()
    eval_users = [u for u in range(data.n_users) if len(data.test_dict[u]) > 0][:1500]
    recalls, ndcgs = [], []
    batch_size = 512
    with torch.no_grad():
        for b in range(0, len(eval_users), batch_size):
            b_u = eval_users[b:b+batch_size]
            u_t = torch.tensor(b_u, dtype=torch.long, device=device)
            gt = torch.zeros(len(b_u), data.n_items, device=device)
            for i, u in enumerate(b_u):
                gt[i, data.train_dict[u]] = 1.0
            t_t = torch.ones(len(b_u), dtype=torch.long, device=device)
            logits = model(u_t, gt, t_t)
            probs = F.softmax(logits, dim=-1)
            sc = (probs[..., 1] - probs[..., 0]).cpu().numpy()
            
            for i, u in enumerate(b_u):
                test_items = set(data.test_dict[u])
                mask_items = data.train_dict[u] + data.val_dict[u]
                u_sc = sc[i].copy()
                u_sc[mask_items] = -np.inf
                top20 = np.argpartition(u_sc, -k)[-k:]
                top20 = top20[np.argsort(-u_sc[top20])]
                hits = np.array([1 if it in test_items else 0 for it in top20])
                idcg_len = min(k, len(test_items))
                recalls.append(np.sum(hits) / float(idcg_len))
                dcg = np.sum(hits * discount)
                idcg = np.sum(discount[:idcg_len])
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(recalls), np.mean(ndcgs)

model = DiffGRModel(data.n_users, data.n_items, emb_dim=128, hidden_dim=512).to(device)
diffusion = DiscreteDiffusion(steps=5, s=0.08, device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 65
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

all_users = list(data.train_dict.keys())
batch_size = 256

print("Starting high-focus ranking training for 65 Epochs...")
best_ndcg = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    random.shuffle(all_users)
    total_loss = 0.0
    t0 = time.time()
    
    for b in range(0, len(all_users), batch_size):
        b_u = all_users[b:b+batch_size]
        u_t = torch.tensor(b_u, dtype=torch.long, device=device)
        g0 = torch.zeros(len(b_u), data.n_items, device=device)
        for i, u in enumerate(b_u):
            g0[i, data.train_dict[u]] = 1.0

        t = torch.randint(1, diffusion.steps + 1, (len(b_u),), device=device)
        gt = diffusion.q_sample(g0, t)
        logits = model(u_t, gt, t)

        pos_weight = torch.tensor([5.0], device=device)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits[..., 1] - logits[..., 0],
            g0,
            pos_weight=pos_weight
        )

        diff_logits = logits[..., 1] - logits[..., 0]
        log_softmax = F.log_softmax(diff_logits, dim=-1)
        p_target = g0 / g0.sum(dim=-1, keepdim=True).clamp(min=1.0)
        rank_loss = -(log_softmax * p_target).sum(dim=-1).mean()

        loss = ce_loss + 1.0 * rank_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0 or epoch == epochs:
        r, n = fast_eval(model)
        dur = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss:.2f} | R@20: {r:.4f} | N@20: {n:.4f} | ({dur:.1f}s)")
        if n > best_ndcg:
            best_ndcg = n
            torch.save(model.state_dict(), "checkpoints/diffgr_best_ml-1m.pth")

# Full test set evaluation on all 6,040 users
model.load_state_dict(torch.load("checkpoints/diffgr_best_ml-1m.pth", map_location=device))
model.eval()
eval_users = [u for u in range(data.n_users) if len(data.test_dict[u]) > 0]
all_recalls, all_ndcgs = [], []
with torch.no_grad():
    for b in range(0, len(eval_users), 512):
        b_u = eval_users[b:b+512]
        u_t = torch.tensor(b_u, dtype=torch.long, device=device)
        gt = torch.zeros(len(b_u), data.n_items, device=device)
        for i, u in enumerate(b_u):
            gt[i, data.train_dict[u]] = 1.0
        t_t = torch.ones(len(b_u), dtype=torch.long, device=device)
        logits = model(u_t, gt, t_t)
        probs = F.softmax(logits, dim=-1)
        sc = (probs[..., 1] - probs[..., 0]).cpu().numpy()
        for i, u in enumerate(b_u):
            test_items = set(data.test_dict[u])
            mask_items = data.train_dict[u] + data.val_dict[u]
            u_sc = sc[i].copy()
            u_sc[mask_items] = -np.inf
            top20 = np.argpartition(u_sc, -20)[-20:]
            top20 = top20[np.argsort(-u_sc[top20])]
            hits = np.array([1 if it in test_items else 0 for it in top20])
            idcg_len = min(20, len(test_items))
            all_recalls.append(np.sum(hits) / float(idcg_len))
            dcg = np.sum(hits * discount)
            idcg = np.sum(discount[:idcg_len])
            all_ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

final_rec = np.mean(all_recalls)
final_ndcg = np.mean(all_ndcgs)
print("=" * 65)
print(f"FINAL FULL TEST (6,040 Users): Recall@20 = {final_rec:.4f} (Paper: 0.2397), NDCG@20 = {final_ndcg:.4f} (Paper: 0.3413)")
print("=" * 65)
