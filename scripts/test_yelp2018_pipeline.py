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
print(f"Active Device: {device}")

sys.path.append('.')
from src.dataset import DataLoaderHandler
from src.model import DiffGRModel
from src.metrics import evaluate_model, compute_power_law_exponent, compute_spearman_rho, compute_recall_ndcg

data = DataLoaderHandler(data_dir='Data', dataset_name='yelp2018', min_interactions=5, seed=42)
print(f"Loaded Yelp2018: {data.n_users} users, {data.n_items} items")

# Test bipartite graph propagation on Yelp2018
print("Building bipartite graph propagation matrix R and C_norm...")
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
print("Graph propagation matrix ready.")

# Evaluate pure graph propagation on 2000 test users to establish strong baseline
eval_users = [u for u in range(2000) if len(data.test_dict[u]) > 0]
R_sub = R[eval_users]
S_graph = (R_sub @ C_norm).toarray()

train_items = [data.train_dict[u] + data.val_dict[u] for u in eval_users]
test_items = [data.test_dict[u] for u in eval_users]

recalls, ndcgs = compute_recall_ndcg(S_graph, train_items, test_items, k=20)
print(f"Pure 2-Hop Graph Diffusion on 2000 users: Recall@20 = {np.mean(recalls):.4f}, NDCG@20 = {np.mean(ndcgs):.4f}")
print("Paper Target: Recall@20 >= 0.0844, NDCG@20 >= 0.0730")
