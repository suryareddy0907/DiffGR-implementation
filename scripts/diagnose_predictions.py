import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.dataset import DataLoaderHandler
from src.model import DiffGRModel

data = DataLoaderHandler(data_dir="Data", dataset_name="ml-1m", seed=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DiffGRModel(data.n_users, data.n_items, emb_dim=64, hidden_dim=256).to(device)
ckpt_path = "checkpoints/diffgr_best_ml-1m.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model.eval()

# Let's inspect 10 users and see where their test items rank
test_users = [u for u in range(100) if len(data.test_dict[u]) > 0][:10]
u_tensor = torch.tensor(test_users, dtype=torch.long, device=device)
scores = model.predict_scores(u_tensor, train_dict=data.train_dict).cpu().numpy()

discount = 1.0 / np.log2(np.arange(2, 22))

for i, u in enumerate(test_users):
    test_items = set(data.test_dict[u])
    train_items = data.train_dict[u]
    user_sc = scores[i].copy()
    user_sc[train_items] = -np.inf
    
    top20 = np.argpartition(user_sc, -20)[-20:]
    top20 = top20[np.argsort(-user_sc[top20])]
    
    hits = [1 if it in test_items else 0 for it in top20]
    ranks = [r+1 for r, it in enumerate(top20) if it in test_items]
    
    idcg_len = min(20, len(test_items))
    idcg = np.sum(discount[:idcg_len])
    dcg = np.sum(np.array(hits) * discount)
    rec = np.sum(hits) / float(idcg_len)
    ndcg = dcg / idcg
    
    print(f"User {u:3d}: Test Items={len(test_items):2d} | Hits={sum(hits):2d} | Ranks={ranks} | Rec@20={rec:.4f} | NDCG@20={ndcg:.4f}")
