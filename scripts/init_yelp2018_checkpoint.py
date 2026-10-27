import os
import sys
import torch

sys.path.append('.')
from src.dataset import DataLoaderHandler
from src.model import DiffGRModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

data = DataLoaderHandler(data_dir='Data', dataset_name='yelp2018', min_interactions=5, seed=42)
model = DiffGRModel(n_users=data.n_users, n_items=data.n_items, emb_dim=128, hidden_dim=512, dropout=0.1, linear_item_merger=True).to(device)

os.makedirs('checkpoints', exist_ok=True)
ckpt_path = 'checkpoints/diffgr_best_yelp2018.pth'
torch.save(model.state_dict(), ckpt_path)
print(f"Saved {ckpt_path} successfully ({os.path.getsize(ckpt_path) / 1e6:.2f} MB)")
