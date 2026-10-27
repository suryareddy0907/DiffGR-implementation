"""
Training and Evaluation script for DiffGR.
Replicates experimental setup from Section 3 of the paper:
- Diffusion steps T = 5
- Cosine schedule parameter s in {0.8, 0.08, 0.008}
- Dropout in {0.1, 0.3, 0.5}
- Learning rate in {1e-2, 1e-3, 1e-4}
- Metrics: Recall@20 and NDCG@20
"""

import os
import argparse
import random
import time
import numpy as np
import torch
import torch.optim as optim

from src.dataset import DataLoaderHandler
from src.diffusion import DiscreteDiffusion
from src.model import DiffGRModel
from src.metrics import evaluate_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="DiffGR Training and Evaluation")
    parser.add_argument('--dataset', type=str, default='ml-1m', choices=['ml-1m', 'pinterest', 'yelp2018'])
    parser.add_argument('--data_dir', type=str, default='Data')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--steps', type=int, default=5, help="Diffusion steps T (paper: T=5)")
    parser.add_argument('--s', type=float, default=0.08, help="Cosine schedule offset s (paper: {0.8, 0.08, 0.008})")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout ratio in {0.1, 0.3, 0.5}")
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_interval', type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    print("=" * 60)
    print(f"DiffGR Replication: Dataset = {args.dataset.upper()} | Device = {args.device}")
    print(f"Hyperparameters: lr={args.lr}, T={args.steps}, s={args.s}, dropout={args.dropout}, emb_dim={args.emb_dim}")
    print("=" * 60)

    # 1. Load Dataset
    data_handler = DataLoaderHandler(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        min_interactions=5,
        seed=args.seed
    )
    train_loader = data_handler.get_train_loader(batch_size=args.batch_size, shuffle=True)

    # 2. Build Model & Diffusion
    model = DiffGRModel(
        n_users=data_handler.n_users,
        n_items=data_handler.n_items,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(args.device)

    diffusion = DiscreteDiffusion(steps=args.steps, s=args.s, device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs('checkpoints', exist_ok=True)
    best_val_ndcg = 0.0
    best_epoch = 0
    checkpoint_path = os.path.join('checkpoints', f'diffgr_best_{args.dataset}.pth')

    print("\nStarting Training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for u_ids, g0 in train_loader:
            u_ids = u_ids.to(args.device)
            g0 = g0.to(args.device)

            optimizer.zero_grad()
            loss = diffusion.compute_elbo_loss(model, u_ids, g0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        # Evaluation
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_recall, val_ndcg = evaluate_model(model, data_handler, split='val', k=20, device=args.device)
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Loss: {avg_loss:.4f} | Val R@20: {val_recall:.4f} | Val N@20: {val_ndcg:.4f}")

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
        else:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes.")

    # 3. Final Test Evaluation using Best Checkpoint
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.device, weights_only=True))
        print(f"Loaded best checkpoint from epoch {best_epoch} (Val N@20 = {best_val_ndcg:.4f})")

    print("\nRunning Final Test Evaluation (K=20)...")
    test_recall, test_ndcg = evaluate_model(model, data_handler, split='test', k=20, device=args.device)

    print("=" * 60)
    print(f"RESULTS FOR {args.dataset.upper()} (Top-20):")
    print(f"Recall@20: {test_recall:.4f}")
    print(f"NDCG@20:   {test_ndcg:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
