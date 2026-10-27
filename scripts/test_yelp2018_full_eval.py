import sys
import numpy as np
import scipy.sparse as sp
import torch

sys.path.append('.')
from src.dataset import DataLoaderHandler
from src.metrics import compute_recall_ndcg

data = DataLoaderHandler(data_dir='Data', dataset_name='yelp2018', min_interactions=5, seed=42)

print("Building bipartite graph propagation matrix...")
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

all_eval_users = [u for u in range(data.n_users) if len(data.test_dict[u]) > 0]
print(f"Evaluating {len(all_eval_users):,} test users in batches of 1,000...")

all_recalls, all_ndcgs = [], []
batch_size = 1000

for b_start in range(0, len(all_eval_users), batch_size):
    b_users = all_eval_users[b_start:b_start + batch_size]
    R_b = R[b_users]
    S_graph = (R_b @ C_norm).toarray()
    
    train_items = [data.train_dict[u] + data.val_dict[u] for u in b_users]
    test_items = [data.test_dict[u] for u in b_users]
    
    recalls, ndcgs = compute_recall_ndcg(S_graph, train_items, test_items, k=20)
    all_recalls.extend(recalls)
    all_ndcgs.extend(ndcgs)

final_recall = np.mean(all_recalls)
final_ndcg = np.mean(all_ndcgs)

print("=" * 75)
print("FINAL BENCHMARK TABLE 1: YELP2018 (ALL 31,668 USERS EVALUATED)")
print("=" * 75)
print(f"Recall@20: {final_recall:.4f} (Paper Target: 0.0844) -> {'SURPASSED (+{:.4f})'.format(final_recall - 0.0844) if final_recall >= 0.0844 else 'FAIL'}")
print(f"NDCG@20:   {final_ndcg:.4f} (Paper Target: 0.0730) -> {'SURPASSED (+{:.4f})'.format(final_ndcg - 0.0730) if final_ndcg >= 0.0730 else 'FAIL'}")
print("=" * 75)
