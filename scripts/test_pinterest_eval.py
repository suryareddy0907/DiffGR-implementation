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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Load Data
class RecDataLoader:
    def __init__(self, data_dir="Data/Pinterest", min_interactions=5, seed=42):
        self.min_interactions = min_interactions
        self.seed = seed
        self._load_and_split(data_dir)

    def _load_and_split(self, data_dir):
        random.seed(self.seed)
        np.random.seed(self.seed)

        interactions = []
        for fname in ['pinterest-20.train.rating', 'pinterest-20.test.rating']:
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        interactions.append((int(parts[0]), int(parts[1])))

        user_items_raw = defaultdict(set)
        for u, i in interactions:
            user_items_raw[u].add(i)

        filtered_users = {u: list(items) for u, items in user_items_raw.items() if len(items) >= self.min_interactions}
        user_map = {u: idx for idx, u in enumerate(sorted(filtered_users.keys()))}
        all_items = sorted(list(set(it for items in filtered_users.values() for it in items)))
        item_map = {it: idx for idx, it in enumerate(all_items)}

        self.n_users = len(user_map)
        self.n_items = len(item_map)

        self.train_dict = defaultdict(list)
        self.val_dict = defaultdict(list)
        self.test_dict = defaultdict(list)

        for raw_u, items in filtered_users.items():
            u_id = user_map[raw_u]
            mapped_items = [item_map[it] for it in items]
            random.shuffle(mapped_items)

            n_inter = len(mapped_items)
            n_train = max(1, int(round(n_inter * 0.70)))
            n_val = max(1, int(round(n_inter * 0.10))) if n_inter >= 10 else 1
            if n_train + n_val >= n_inter:
                n_val = max(0, n_inter - n_train - 1)

            self.train_dict[u_id] = mapped_items[:n_train]
            self.val_dict[u_id] = mapped_items[n_train:n_train + n_val]
            self.test_dict[u_id] = mapped_items[n_train + n_val:]
            if len(self.test_dict[u_id]) == 0 and len(self.val_dict[u_id]) > 1:
                self.test_dict[u_id] = [self.val_dict[u_id].pop()]

data = RecDataLoader()
print(f"Data loaded: {data.n_users} users, {data.n_items} items")

# 2. Model
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1) if half_dim > 1 else 1.0
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class DiffGRModel(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=128, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

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

        self.time_embed = SinusoidalTimeEmbedding(emb_dim)

        self.user_merger = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        self.item_merger = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        self.w_pos = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.w_neg = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.bias_pos = nn.Parameter(torch.zeros(1))
        self.bias_neg = nn.Parameter(torch.zeros(1))

    def forward(self, u_ids, gt, t):
        emb_u = self.user_embeddings(u_ids)
        emb_gt = self.encoder(gt)
        emb_t = self.time_embed(t)

        user_concat = torch.cat([emb_u, emb_gt, emb_t], dim=-1)
        emb_final_u = self.user_merger(user_concat)

        emb_i = self.item_embeddings.weight
        emb_final_i = self.item_merger(emb_i)

        logits_neg = torch.matmul(emb_final_u * self.w_neg, emb_final_i.T) + self.bias_neg
        logits_pos = torch.matmul(emb_final_u * self.w_pos, emb_final_i.T) + self.bias_pos

        logits = torch.stack([logits_neg, logits_pos], dim=-1)
        return logits

    @torch.no_grad()
    def predict_scores(self, u_ids, train_dict=None):
        self.eval()
        batch_size = u_ids.size(0)
        device = u_ids.device

        gt = torch.zeros(batch_size, self.n_items, device=device)
        if train_dict is not None:
            for i, uid in enumerate(u_ids.cpu().tolist()):
                items = train_dict.get(uid, [])
                if len(items) > 0:
                    gt[i, items] = 1.0

        t = torch.full((batch_size,), 5, dtype=torch.long, device=device)
        logits = self.forward(u_ids, gt, t)
        diff_logits = logits[..., 1] - logits[..., 0]
        return diff_logits

model = DiffGRModel(data.n_users, data.n_items, emb_dim=128, hidden_dim=512, dropout=0.1).to(device)
ckpt_path = "checkpoints/diffgr_best_pinterest.pth"
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {ckpt_path}")
else:
    print("Checkpoint not found!")

# 3. Build Graph
print("Building graph propagation...")
rows, cols = [], []
for u, items in data.train_dict.items():
    for i in items:
        rows.append(u)
        cols.append(i)
vals = np.ones(len(rows), dtype=np.float32)
R = sp.csr_matrix((vals, (rows, cols)), shape=(data.n_users, data.n_items))
C = R.T @ R
deg = np.array(C.sum(axis=1)).flatten()
deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
deg_inv_sqrt[deg <= 0] = 0.0
D_inv = sp.diags(deg_inv_sqrt)
C_norm = (D_inv @ C @ D_inv).tolil()
C_norm.setdiag(0)
C_norm = C_norm.tocsr()

def compute_metrics(scores, target_dict, eval_users, train_dict, val_dict, split='test', k=20):
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    recalls, ndcgs = [], []
    for idx, u in enumerate(eval_users):
        t_items = set(target_dict[u])
        if len(t_items) == 0:
            continue
        u_sc = scores[idx].copy()
        mask = train_dict.get(u, []) + (val_dict.get(u, []) if split == 'test' else [])
        if len(mask) > 0:
            u_sc[mask] = -np.inf
        top_k = np.argpartition(u_sc, -k)[-k:]
        top_k = top_k[np.argsort(-u_sc[top_k])]
        hits = np.array([1 if it in t_items else 0 for it in top_k])
        idcg_len = min(k, len(t_items))
        recall = np.sum(hits) / float(idcg_len)
        recalls.append(recall)
        dcg = np.sum(hits * discount)
        idcg = np.sum(discount[:idcg_len])
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(recalls)), float(np.mean(ndcgs))

# Test on 2,000 users for quick evaluation of validation vs test with different betas
eval_users = [u for u in range(2000) if len(data.test_dict[u]) > 0]
print(f"Evaluating {len(eval_users)} test users...")

with torch.no_grad():
    u_tensor = torch.tensor(eval_users, dtype=torch.long, device=device)
    base_sc = model.predict_scores(u_tensor, train_dict=data.train_dict).cpu().numpy()

R_sub = R[eval_users]
S_graph = (R_sub @ C_norm).toarray()
S_graph = S_graph / (np.max(S_graph, axis=-1, keepdims=True) + 1e-6)

for beta in [0.0, 0.5, 1.0, 1.2, 1.5, 2.0]:
    sc = base_sc + beta * S_graph
    r, n = compute_metrics(sc, data.test_dict, eval_users, data.train_dict, data.val_dict, split='test')
    print(f"Beta={beta:.1f} | Test Recall@20: {r:.4f} | Test NDCG@20: {n:.4f}")

# Also evaluate on Validation set for the same users
eval_val_users = [u for u in range(2000) if len(data.val_dict[u]) > 0]
with torch.no_grad():
    u_tensor_val = torch.tensor(eval_val_users, dtype=torch.long, device=device)
    base_sc_val = model.predict_scores(u_tensor_val, train_dict=data.train_dict).cpu().numpy()
R_sub_val = R[eval_val_users]
S_graph_val = (R_sub_val @ C_norm).toarray()
S_graph_val = S_graph_val / (np.max(S_graph_val, axis=-1, keepdims=True) + 1e-6)

print("\n--- Validation Split ---")
for beta in [0.0, 1.2]:
    sc_val = base_sc_val + beta * S_graph_val
    r_val, n_val = compute_metrics(sc_val, data.val_dict, eval_val_users, data.train_dict, data.val_dict, split='val')
    print(f"Val Beta={beta:.1f} | Val Recall@20: {r_val:.4f} | Val NDCG@20: {n_val:.4f}")
