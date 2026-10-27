import os
import sys
import torch
import numpy as np
import scipy.stats as stats
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))
from src.dataset import DataLoaderHandler
from src.model import DiffGRModel

data = DataLoaderHandler(data_dir="Data", dataset_name="ml-1m", seed=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DiffGRModel(data.n_users, data.n_items, emb_dim=64, hidden_dim=256).to(device)
ckpt = torch.load("checkpoints/diffgr_best_ml-1m.pth", map_location=device, weights_only=True)
model.load_state_dict(ckpt)
model.eval()

orig_u_deg = [len(data.train_dict[u]) for u in range(data.n_users)]
it_cnt = defaultdict(int)
for u, items in data.train_dict.items():
    for it in items:
        it_cnt[it] += 1
orig_i_deg = [it_cnt[i] for i in range(data.n_items)]

# Compute predictions for all users
all_scores = []
with torch.no_grad():
    for b in range(0, data.n_users, 512):
        users = list(range(b, min(b + 512, data.n_users)))
        u_t = torch.tensor(users, dtype=torch.long, device=device)
        sc = model.predict_scores(u_t, train_dict=data.train_dict).cpu().numpy()
        all_scores.append(sc)
all_scores = np.concatenate(all_scores, axis=0)

# Strategy 1: Current notebook logic (probabilistic count of top items)
for p in [0.95, 0.985, 0.995, 1.0]:
    gen_u_deg = []
    gen_i_deg = np.zeros(data.n_items)
    for u in range(data.n_users):
        orig_d = orig_u_deg[u]
        n_edges = int(np.random.binomial(n=orig_d, p=p)) if p < 1.0 else orig_d
        top_it = np.argpartition(all_scores[u], -n_edges)[-n_edges:]
        gen_u_deg.append(len(top_it))
        for it in top_it:
            gen_i_deg[it] += 1
    rho_u, _ = stats.spearmanr(orig_u_deg, gen_u_deg)
    rho_i, _ = stats.spearmanr(orig_i_deg, gen_i_deg)
    print(f"Strategy 1 (p={p}): User rho = {rho_u:.4f} (target 0.9868), Item rho = {rho_i:.4f} (target 0.9949)")

# Strategy 2: Direct Bernoulli edge retention based on Eq. 7 and Section 2.2
# Section 2.2: "The user-item bipartite sub-graph G0 is reconstructed by eliminating the edges with a high probability of being 0 from G^0."
# For existing edges in G^0, edge exists unless denoised model predicts 0 with high confidence.
for keep_train_p in [0.98, 0.99, 0.995]:
    gen_i_deg = np.zeros(data.n_items)
    gen_u_deg = []
    for u in range(data.n_users):
        tr = data.train_dict[u]
        # Keep training edges with high probability (or threshold)
        kept_tr = [it for it in tr if np.random.rand() < keep_train_p]
        # Add top predicted new edges
        n_add = max(0, int(len(tr) * (1 - keep_train_p)))
        u_sc = all_scores[u].copy()
        u_sc[tr] = -np.inf
        top_add = np.argpartition(u_sc, -n_add)[-n_add:] if n_add > 0 else []
        tot_edges = kept_tr + list(top_add)
        gen_u_deg.append(len(tot_edges))
        for it in tot_edges:
            gen_i_deg[it] += 1
    rho_u, _ = stats.spearmanr(orig_u_deg, gen_u_deg)
    rho_i, _ = stats.spearmanr(orig_i_deg, gen_i_deg)
    print(f"Strategy 2 (keep_p={keep_train_p}): User rho = {rho_u:.4f}, Item rho = {rho_i:.4f}")
