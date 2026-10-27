"""
DiffGR Replication: MovieLens 1M (ML-1M)
Run this script to train the fixed DiffGR model and compare results with the paper.

Paper Targets (Table 1):
  Recall@20: 0.2397
  NDCG@20:   0.3413

Usage: python notebooks/run_ml1m.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import torch.optim as optim

from src.dataset import DataLoaderHandler
from src.diffusion import DiscreteDiffusion
from src.model import DiffGRModel
from src.metrics import evaluate_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load ML-1M (70% train / 10% validation / 20% test)
data_dir = os.path.join(os.path.dirname(__file__), '..', 'Data')
data_handler = DataLoaderHandler(data_dir=data_dir, dataset_name="ml-1m", min_interactions=5, seed=42)
train_loader = data_handler.get_train_loader(batch_size=256, shuffle=True)

# 2. Initialize Model & Discrete Diffusion (T=5, s=0.08)
model = DiffGRModel(
    n_users=data_handler.n_users,
    n_items=data_handler.n_items,
    emb_dim=64,
    hidden_dim=256,
    dropout=0.1
).to(device)

diffusion = DiscreteDiffusion(steps=5, s=0.08, device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
print("Model and Discrete Diffusion initialized.")

# 3. Train DiffGR Model
epochs = 30
eval_interval = 5
best_val_ndcg = 0.0
best_epoch = 0

losses = []
val_results = []

print(f"\nStarting training for {epochs} epochs...")
start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    batches = 0
    for u_ids, g0 in train_loader:
        u_ids = u_ids.to(device)
        g0 = g0.to(device)

        optimizer.zero_grad()
        loss = diffusion.compute_elbo_loss(model, u_ids, g0)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batches += 1

    avg_loss = epoch_loss / max(1, batches)
    losses.append(avg_loss)

    if epoch % eval_interval == 0 or epoch == epochs:
        r20, n20 = evaluate_model(model, data_handler, diffusion=diffusion, split="val", k=20, device=device)
        val_results.append((epoch, r20, n20))
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Val R@20: {r20:.4f} | Val N@20: {n20:.4f}")

        if n20 > best_val_ndcg:
            best_val_ndcg = n20
            best_epoch = epoch
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/diffgr_best_ml1m.pth')
    else:
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f}")

elapsed = (time.time() - start_time) / 60.0
print(f"\nTraining finished in {elapsed:.2f} minutes.")

# 4. Load best checkpoint and evaluate on test set
if os.path.exists('checkpoints/diffgr_best_ml1m.pth'):
    model.load_state_dict(torch.load('checkpoints/diffgr_best_ml1m.pth', map_location=device, weights_only=True))
    print(f"Loaded best checkpoint from epoch {best_epoch} (Val N@20 = {best_val_ndcg:.4f})")

print("\nEvaluating on Test Set (K=20)...")
test_r20, test_n20 = evaluate_model(model, data_handler, diffusion=diffusion, split="test", k=20, device=device)

# 5. Print comparison table
print("\n" + "=" * 70)
print("COMPARISON: Replicated vs. Paper Results (ML-1M, Top-20)")
print("=" * 70)
print(f"{'Method':<25} {'Recall@20':>12} {'NDCG@20':>12}")
print("-" * 70)
print(f"{'BPR':<25} {'0.1793':>12} {'0.2624':>12}")
print(f"{'LightGCN':<25} {'0.1731':>12} {'0.3030':>12}")
print(f"{'MultiVAE':<25} {'0.2312':>12} {'0.3177':>12}")
print(f"{'CODIGEM':<25} {'0.2221':>12} {'0.3211':>12}")
print(f"{'DiffRec':<25} {'0.2280':>12} {'0.3287':>12}")
print(f"{'L-DiffRec':<25} {'0.2247':>12} {'0.3111':>12}")
print(f"{'DiffGR (Paper)':<25} {'0.2397':>12} {'0.3413':>12}")
print("-" * 70)
print(f"{'DiffGR (Replicated)':<25} {test_r20:>12.4f} {test_n20:>12.4f}")
print("=" * 70)

# 6. Print training loss progression
print("\nTraining Loss Progression:")
for i, l in enumerate(losses):
    bar = "#" * int(l * 100)
    print(f"  Epoch {i+1:02d}: {l:.4f} {bar}")

# 7. Print validation metric progression
print("\nValidation Metric Progression:")
for ep, r, n in val_results:
    print(f"  Epoch {ep:02d}: Recall@20 = {r:.4f}, NDCG@20 = {n:.4f}")
