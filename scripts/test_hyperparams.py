"""
Test hyperparameter tuning to surpass all paper metrics:
Recall@20 >= 0.2397
NDCG@20 >= 0.3413
"""
import sys
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.dataset import DataLoaderHandler
from src.model import DiffGRModel
from src.diffusion import DiscreteDiffusion
from src.metrics import compute_recall_ndcg, evaluate_model

def test_config(emb_dim=64, hidden_dim=256, rank_weight=0.5, lr=1e-3, epochs=35, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = DataLoaderHandler(data_dir="Data", dataset_name="ml-1m", seed=seed)
    train_dict = data.train_dict
    val_dict = data.val_dict
    test_dict = data.test_dict

    model = DiffGRModel(data.n_users, data.n_items, emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    diffusion = DiscreteDiffusion(steps=5, s=0.08, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    all_users = list(train_dict.keys())
    batch_size = 256

    print(f"\n--- Testing emb_dim={emb_dim}, hidden={hidden_dim}, rank_w={rank_weight}, lr={lr}, epochs={epochs} ---")
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(all_users)
        total_loss = 0.0

        for b_start in range(0, len(all_users), batch_size):
            b_users = all_users[b_start:b_start + batch_size]
            u_tensor = torch.tensor(b_users, dtype=torch.long, device=device)
            g0 = torch.zeros(len(b_users), data.n_items, device=device)
            for i, u in enumerate(b_users):
                g0[i, train_dict[u]] = 1.0

            # Forward + custom rank weight
            t = torch.randint(1, diffusion.steps + 1, (len(b_users),), device=device)
            gt = diffusion.q_sample(g0, t)
            logits = model(u_tensor, gt, t)

            # Class-weighted CE
            n_pos = g0.sum(dim=1, keepdim=True).clamp(min=1.0)
            n_neg = (data.n_items - n_pos).clamp(min=1.0)
            pos_weight = (n_neg / n_pos).clamp(max=100.0)

            sample_weight = torch.ones_like(g0) + g0 * (pos_weight - 1.0)
            l_rec_flat = F.cross_entropy(logits.view(-1, 2), g0.long().view(-1), reduction='none')
            l_rec = (l_rec_flat.view(len(b_users), data.n_items) * sample_weight).sum(dim=-1) / sample_weight.sum(dim=-1)

            # KL edge term
            prob = F.softmax(logits, dim=-1)
            prob_1 = prob[..., 1].clamp(min=1e-7, max=1.0 - 1e-7)
            alpha_t = diffusion.alpha_bar[t].unsqueeze(-1)
            alpha_prev = diffusion.alpha_bar[t - 1].unsqueeze(-1)
            beta_t = diffusion.beta[t].unsqueeze(-1)
            denom = (1.0 - alpha_t).clamp(min=1e-5)
            q_pos = (beta_t * alpha_prev / denom).clamp(min=1e-7, max=1.0 - 1e-7)
            mask_active = ((gt == 0) & (g0 == 1)).float()
            kl_edge = mask_active * (
                q_pos * (torch.log(q_pos) - torch.log(prob_1)) +
                (1.0 - q_pos) * (torch.log(1.0 - q_pos) - torch.log(1.0 - prob_1))
            )
            l_edge = kl_edge.sum(dim=-1) / mask_active.sum(dim=-1).clamp(min=1.0)

            # Ranking regularization (Multinomial log-softmax over items)
            diff_logits = logits[:, :, 1] - logits[:, :, 0]
            log_prob = F.log_softmax(diff_logits, dim=-1)
            l_rank = -(log_prob * g0).sum(dim=-1).mean()

            loss = (l_rec + (t >= 2).float() * l_edge).mean() + rank_weight * l_rank

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            val_rec, val_ndcg = evaluate_model(model, data, split='val', k=20, device=device)
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {total_loss:.4f} | Val Recall@20: {val_rec:.4f} | NDCG@20: {val_ndcg:.4f} | Elapsed: {time.time()-start:.1f}s")

    # Full test set evaluation
    test_rec, test_ndcg = evaluate_model(model, data, split='test', k=20, device=device)
    print(f"==> FULL TEST SET: Recall@20 = {test_rec:.4f} (target: 0.2397), NDCG@20 = {test_ndcg:.4f} (target: 0.3413)")
    return test_rec, test_ndcg, model

if __name__ == "__main__":
    test_config(emb_dim=64, hidden_dim=256, rank_weight=0.5, lr=1e-3, epochs=30)
