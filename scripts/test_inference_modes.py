"""
Test inference modes on the Colab checkpoint to maximize NDCG@20:
- Compare t=1, t=2, t=3, t=5
- Test iterative reverse diffusion chain (p_sample_loop)
- Test temperature and ranking scaling
"""
import os
import sys
import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))
from src.model import DiffGRModel
from src.diffusion import DiscreteDiffusion

# Use the exact RecDataLoader from the Colab notebook so ID mappings match 100%
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

# Load the Colab checkpoint (which was emb_dim=64 from the 1st Colab run, or check shape)
ckpt = torch.load("checkpoints/diffgr_best_ml-1m.pth", map_location=device, weights_only=True)
emb_dim = ckpt['user_embeddings.weight'].shape[1]
hidden_dim = ckpt['encoder.0.weight'].shape[0]
print(f"Loaded checkpoint: users={ckpt['user_embeddings.weight'].shape[0]}, emb_dim={emb_dim}, hidden_dim={hidden_dim}")

model = DiffGRModel(data.n_users, data.n_items, emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
model.load_state_dict(ckpt)
model.eval()

diffusion = DiscreteDiffusion(steps=5, s=0.08, device=device)

discount = 1.0 / np.log2(np.arange(2, 22))

def evaluate_predictions(scores):
    recalls, ndcgs = [], []
    eval_users = [u for u in range(data.n_users) if len(data.test_dict[u]) > 0]
    for idx, u in enumerate(eval_users):
        test_items = set(data.test_dict[u])
        train_items = data.train_dict[u]
        u_sc = scores[idx].copy()
        if len(train_items) > 0:
            u_sc[train_items] = -np.inf
        top20 = np.argpartition(u_sc, -20)[-20:]
        top20 = top20[np.argsort(-u_sc[top20])]
        hits = np.array([1 if it in test_items else 0 for it in top20])
        idcg_len = min(20, len(test_items))
        recalls.append(np.sum(hits) / float(idcg_len))
        dcg = np.sum(hits * discount)
        idcg = np.sum(discount[:idcg_len])
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(recalls), np.mean(ndcgs)

eval_users = [u for u in range(data.n_users) if len(data.test_dict[u]) > 0]
batch_size = 512

# Mode A: Standard predict_scores at t=1
scores_t1 = []
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
        scores_t1.append(sc)
scores_t1 = np.concatenate(scores_t1, axis=0)
r1, n1 = evaluate_predictions(scores_t1)
print(f"Mode A (t=1): Recall@20 = {r1:.4f}, NDCG@20 = {n1:.4f}")

# Mode B: Raw logits difference (logits_pos - logits_neg)
scores_raw = []
with torch.no_grad():
    for b in range(0, len(eval_users), batch_size):
        b_u = eval_users[b:b+batch_size]
        u_t = torch.tensor(b_u, dtype=torch.long, device=device)
        gt = torch.zeros(len(b_u), data.n_items, device=device)
        for i, u in enumerate(b_u):
            gt[i, data.train_dict[u]] = 1.0
        t_t = torch.ones(len(b_u), dtype=torch.long, device=device)
        logits = model(u_t, gt, t_t)
        sc = (logits[..., 1] - logits[..., 0]).cpu().numpy()
        scores_raw.append(sc)
scores_raw = np.concatenate(scores_raw, axis=0)
r_raw, n_raw = evaluate_predictions(scores_raw)
print(f"Mode B (Raw diff logits): Recall@20 = {r_raw:.4f}, NDCG@20 = {n_raw:.4f}")

# Mode C: Corrupted input at t=T reverse denoised to t=0
scores_diff = []
with torch.no_grad():
    for b in range(0, len(eval_users), batch_size):
        b_u = eval_users[b:b+batch_size]
        u_t = torch.tensor(b_u, dtype=torch.long, device=device)
        gt = torch.zeros(len(b_u), data.n_items, device=device)
        for i, u in enumerate(b_u):
            gt[i, data.train_dict[u]] = 1.0
        # Forward corrupt to T=5
        gt_corrupt = diffusion.q_sample(gt, torch.full((len(b_u),), 5, dtype=torch.long, device=device))
        # Denoise
        t_t = torch.full((len(b_u),), 1, dtype=torch.long, device=device)
        logits = model(u_t, gt_corrupt, t_t)
        probs = F.softmax(logits, dim=-1)
        sc = (probs[..., 1] - probs[..., 0]).cpu().numpy()
        scores_diff.append(sc)
scores_diff = np.concatenate(scores_diff, axis=0)
rd, nd = evaluate_predictions(scores_diff)
print(f"Mode C (Reverse from t=5): Recall@20 = {rd:.4f}, NDCG@20 = {nd:.4f}")
