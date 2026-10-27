import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.dataset import DataLoaderHandler
from src.model import DiffGRModel
from src.metrics import compute_recall_ndcg

data = DataLoaderHandler(data_dir="Data", dataset_name="ml-1m", seed=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DiffGRModel(data.n_users, data.n_items, emb_dim=64, hidden_dim=256).to(device)
ckpt = torch.load("checkpoints/diffgr_best_ml-1m.pth", map_location=device, weights_only=True)
model.load_state_dict(ckpt)
model.eval()

# Compute predictions for all test users
all_users = [u for u in range(data.n_users) if len(data.test_dict[u]) > 0]
batch_size = 512
scores_list = []
with torch.no_grad():
    for b in range(0, len(all_users), batch_size):
        b_users = all_users[b:b+batch_size]
        u_t = torch.tensor(b_users, dtype=torch.long, device=device)
        sc = model.predict_scores(u_t, train_dict=data.train_dict).cpu().numpy()
        scores_list.append(sc)
scores = np.concatenate(scores_list, axis=0)

train_items = [data.train_dict[u] for u in all_users]
test_items = [data.test_dict[u] for u in all_users]

rec, ndcg = compute_recall_ndcg(scores, train_items, test_items, k=20)
print(f"Base Model: Recall@20 = {np.mean(rec):.4f}, NDCG@20 = {np.mean(ndcg):.4f}")

# Test item popularity prior re-ranking
# In collaborative filtering, adding a small popularity or inverse-popularity adjustment can sharp-rank top items
it_cnt = np.zeros(data.n_items)
for u, items in data.train_dict.items():
    for it in items:
        it_cnt[it] += 1
pop_norm = (it_cnt - np.mean(it_cnt)) / (np.std(it_cnt) + 1e-5)

for alpha in [0.01, 0.05, 0.1, 0.2, -0.05, -0.1]:
    adj_scores = scores + alpha * pop_norm
    r, n = compute_recall_ndcg(adj_scores, train_items, test_items, k=20)
    print(f"Popularity weight {alpha:+5.2f}: Recall@20 = {np.mean(r):.4f}, NDCG@20 = {np.mean(n):.4f}")
