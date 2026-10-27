import os
import sys
import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))
from src.model import DiffGRModel

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

ckpt = torch.load("checkpoints/diffgr_best_ml-1m.pth", map_location=device, weights_only=True)
emb_dim = ckpt['user_embeddings.weight'].shape[1]
hidden_dim = ckpt['encoder.0.weight'].shape[0]

model = DiffGRModel(data.n_users, data.n_items, emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
model.load_state_dict(ckpt)
model.eval()

eval_users = [u for u in range(data.n_users) if len(data.test_dict[u]) > 0]
batch_size = 512

discount = 1.0 / np.log2(np.arange(2, 22))

# Test with feed = train, mask = train ONLY (Current behavior)
scores_list = []
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
        scores_list.append((probs[..., 1] - probs[..., 0]).cpu().numpy())
scores = np.concatenate(scores_list, axis=0)

rec_only_tr, ndcg_only_tr = [], []
for idx, u in enumerate(eval_users):
    test_items = set(data.test_dict[u])
    tr_items = data.train_dict[u]
    u_sc = scores[idx].copy()
    u_sc[tr_items] = -np.inf
    top20 = np.argpartition(u_sc, -20)[-20:]
    top20 = top20[np.argsort(-u_sc[top20])]
    hits = np.array([1 if it in test_items else 0 for it in top20])
    idcg_len = min(20, len(test_items))
    rec_only_tr.append(np.sum(hits) / float(idcg_len))
    dcg = np.sum(hits * discount)
    idcg = np.sum(discount[:idcg_len])
    ndcg_only_tr.append(dcg / idcg if idcg > 0 else 0.0)

print(f"Mask Train Only:      Recall@20 = {np.mean(rec_only_tr):.4f}, NDCG@20 = {np.mean(ndcg_only_tr):.4f}")

# Test with feed = train, mask = train + val (DiffRec standard protocol)
rec_mask_tv, ndcg_mask_tv = [], []
for idx, u in enumerate(eval_users):
    test_items = set(data.test_dict[u])
    mask_items = data.train_dict[u] + data.val_dict[u]
    u_sc = scores[idx].copy()
    u_sc[mask_items] = -np.inf
    top20 = np.argpartition(u_sc, -20)[-20:]
    top20 = top20[np.argsort(-u_sc[top20])]
    hits = np.array([1 if it in test_items else 0 for it in top20])
    idcg_len = min(20, len(test_items))
    rec_mask_tv.append(np.sum(hits) / float(idcg_len))
    dcg = np.sum(hits * discount)
    idcg = np.sum(discount[:idcg_len])
    ndcg_mask_tv.append(dcg / idcg if idcg > 0 else 0.0)

print(f"Mask Train + Val:     Recall@20 = {np.mean(rec_mask_tv):.4f}, NDCG@20 = {np.mean(ndcg_mask_tv):.4f}")

# Test with feed = train + val (tst_w_val in DiffRec), mask = train + val
scores_tv_list = []
with torch.no_grad():
    for b in range(0, len(eval_users), batch_size):
        b_u = eval_users[b:b+batch_size]
        u_t = torch.tensor(b_u, dtype=torch.long, device=device)
        gt = torch.zeros(len(b_u), data.n_items, device=device)
        for i, u in enumerate(b_u):
            gt[i, data.train_dict[u] + data.val_dict[u]] = 1.0
        t_t = torch.ones(len(b_u), dtype=torch.long, device=device)
        logits = model(u_t, gt, t_t)
        probs = F.softmax(logits, dim=-1)
        scores_tv_list.append((probs[..., 1] - probs[..., 0]).cpu().numpy())
scores_tv = np.concatenate(scores_tv_list, axis=0)

rec_feed_tv, ndcg_feed_tv = [], []
for idx, u in enumerate(eval_users):
    test_items = set(data.test_dict[u])
    mask_items = data.train_dict[u] + data.val_dict[u]
    u_sc = scores_tv[idx].copy()
    u_sc[mask_items] = -np.inf
    top20 = np.argpartition(u_sc, -20)[-20:]
    top20 = top20[np.argsort(-u_sc[top20])]
    hits = np.array([1 if it in test_items else 0 for it in top20])
    idcg_len = min(20, len(test_items))
    rec_feed_tv.append(np.sum(hits) / float(idcg_len))
    dcg = np.sum(hits * discount)
    idcg = np.sum(discount[:idcg_len])
    ndcg_feed_tv.append(dcg / idcg if idcg > 0 else 0.0)

print(f"Feed & Mask Train+Val: Recall@20 = {np.mean(rec_feed_tv):.4f}, NDCG@20 = {np.mean(ndcg_feed_tv):.4f}")
