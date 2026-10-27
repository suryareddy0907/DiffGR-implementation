"""
Evaluation metrics for DiffGR replication:
1. Recommendation Ranking Metrics: Recall@K and NDCG@K (K=20 default)
2. Graph Fidelity Metrics:
   - Spearman Rank-order Correlation (rho) for User and Item degree ranking
   - Power-Law Exponent (PLE) difference
"""
import numpy as np
import scipy.stats as stats
import torch


def compute_recall_ndcg(scores, train_items_list, test_items_list, k=20):
    """
    Compute Recall@K and NDCG@K for a batch of users.
    
    Args:
        scores: numpy array [batch_size, n_items]
        train_items_list: list of lists containing training item indices for each user in batch
        test_items_list: list of lists containing test item indices for each user in batch
        k: top K cutoff (default: 20)
    Returns:
        recalls: list of float
        ndcgs: list of float
    """
    recalls = []
    ndcgs = []

    # Precompute IDCG discounting factors: 1 / log2(rank + 1)
    discount = 1.0 / np.log2(np.arange(2, k + 2))

    for idx in range(len(scores)):
        test_items = set(test_items_list[idx])
        if len(test_items) == 0:
            continue

        user_scores = scores[idx].copy()
        train_items = train_items_list[idx]
        if len(train_items) > 0:
            user_scores[train_items] = -np.inf  # Mask seen training items

        # Get top K item indices
        top_k = np.argpartition(user_scores, -k)[-k:]
        top_k = top_k[np.argsort(-user_scores[top_k])]

        # Calculate hits
        hits = np.array([1 if item in test_items else 0 for item in top_k])
        
        # Recall@K
        recall = np.sum(hits) / float(len(test_items))
        recalls.append(recall)

        # NDCG@K
        dcg = np.sum(hits * discount)
        idcg_len = min(k, len(test_items))
        idcg = np.sum(discount[:idcg_len])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return recalls, ndcgs


def evaluate_model(model, data_handler, split='test', k=20, batch_size=256, device='cpu'):
    """
    Evaluate DiffGR on validation or test set.
    """
    model.eval()
    target_dict = data_handler.test_dict if split == 'test' else data_handler.val_dict
    train_dict = data_handler.train_dict
    
    all_users = [u for u in range(data_handler.n_users) if len(target_dict[u]) > 0]
    all_recalls = []
    all_ndcgs = []

    with torch.no_grad():
        for start_idx in range(0, len(all_users), batch_size):
            batch_users = all_users[start_idx:start_idx + batch_size]
            u_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            
            # Predict scores for batch
            scores = model.predict_scores(u_tensor).cpu().numpy()

            train_items = [train_dict[u] for u in batch_users]
            test_items = [target_dict[u] for u in batch_users]

            recalls, ndcgs = compute_recall_ndcg(scores, train_items, test_items, k=k)
            all_recalls.extend(recalls)
            all_ndcgs.extend(ndcgs)

    mean_recall = float(np.mean(all_recalls)) if len(all_recalls) > 0 else 0.0
    mean_ndcg = float(np.mean(all_ndcgs)) if len(all_ndcgs) > 0 else 0.0
    return mean_recall, mean_ndcg


def compute_spearman_rho(orig_degrees, gen_degrees):
    """
    Computes Spearman Rank-order correlation rho (Table 2 in paper).
    """
    rho, _ = stats.spearmanr(orig_degrees, gen_degrees)
    return float(rho)


def estimate_power_law_alpha(degrees, xmin=1):
    """
    Maximum likelihood estimator for power-law exponent:
    alpha = 1 + n * [ sum( ln(x_i / (xmin - 0.5)) ) ]^{-1}
    """
    filtered = np.array([d for d in degrees if d >= xmin], dtype=float)
    if len(filtered) == 0:
        return 0.0
    return 1.0 + len(filtered) / np.sum(np.log(filtered / (xmin - 0.5)))
