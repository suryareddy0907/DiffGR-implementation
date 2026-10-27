"""
Discrete Diffusion Process for DiffGR.
Implements:
1. Cosine noise scheduler (Eq. 3)
2. Discrete forward corruption q(G^t | G^0) via Bernoulli edge removal (Eq. 1-2)
3. Posterior transition q(G^{t-1} | G^t, G^0) (Eq. 4)
4. ELBO loss with class-imbalance weighting (Eq. 5-6)
5. Iterative reverse sampling chain for inference
"""

import math
import torch
import torch.nn.functional as F


class DiscreteDiffusion:
    def __init__(self, steps=5, s=0.08, device='cpu'):
        """
        Args:
            steps (int): Total diffusion steps T (default: 5).
            s (float): Tunable noise smoothness hyper-parameter (e.g., 0.8, 0.08, 0.008).
            device: torch device.
        """
        self.steps = steps
        self.s = s
        self.device = device

        self.alpha_bar = self._get_cosine_schedule().to(device)  # [T + 1]
        self.beta = torch.zeros(self.steps + 1, device=device)
        self.beta[0] = 0.0
        for t in range(1, self.steps + 1):
            self.beta[t] = 1.0 - (self.alpha_bar[t] / self.alpha_bar[t - 1])
            self.beta[t] = torch.clamp(self.beta[t], min=1e-5, max=0.9999)

    def _get_cosine_schedule(self):
        """
        Cosine schedule from paper Eq. 3:
        alpha_bar_t = f(t) / f(0)
        f(t) = cos( ((t / T + s) / (1 + s)) * (pi / 2) )
        """
        T = self.steps
        s = self.s
        f_0 = math.cos((s / (1.0 + s)) * (math.pi / 2.0))

        alpha_bars = []
        for t in range(T + 1):
            val = math.cos(((t / T + s) / (1.0 + s)) * (math.pi / 2.0))
            alpha_bars.append(val / f_0)
        return torch.tensor(alpha_bars, dtype=torch.float32)

    def q_sample(self, g0, t):
        """
        Forward diffusion step q(G^t | G^0) (Eq. 2):
        Each existing edge (g0 == 1) is kept with probability alpha_bar[t],
        and removed (set to 0) with probability 1 - alpha_bar[t].
        Non-edges (g0 == 0) remain 0.
        """
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)  # [batch_size, 1]
        keep_prob = alpha_bar_t * g0  # Only positive interactions have non-zero keep_prob
        rand = torch.rand_like(g0)
        gt = (rand < keep_prob).float()
        return gt

    def compute_elbo_loss(self, model, u_ids, g0):
        """
        Compute DiffGR ELBO loss with class-imbalance weighting (Eq. 5 & 6).

        Key fix: In sparse graphs (e.g. ML-1M is 95.53% sparse), non-edges
        outnumber positive edges ~31:1. Unweighted CE makes the model predict
        "no edge" for everything. We apply per-sample positive class weighting
        so the gradient signal from true interactions is not drowned out.
        """
        batch_size = g0.size(0)
        n_items = g0.size(1)

        # Sample t in {1, ..., T}
        t = torch.randint(1, self.steps + 1, (batch_size,), device=self.device)

        # Corrupt G^0 to G^t
        gt = self.q_sample(g0, t)

        # Model prediction: logits [batch_size, n_items, 2]
        logits = model(u_ids, gt, t)

        # --- Class-imbalance weighted reconstruction loss L_rec ---
        # Compute per-sample positive weight: n_neg / n_pos (clamped)
        n_pos = g0.sum(dim=1, keepdim=True).clamp(min=1.0)  # [batch_size, 1]
        n_neg = (n_items - n_pos).clamp(min=1.0)
        pos_weight = (n_neg / n_pos).clamp(max=100.0)  # [batch_size, 1]

        # Build per-element weight: weight=1 for class 0, weight=pos_weight for class 1
        target = g0.long()  # {0, 1}
        sample_weight = torch.ones_like(g0)
        sample_weight = sample_weight + g0 * (pos_weight - 1.0)  # scale up positives

        # Weighted cross-entropy
        l_rec_flat = F.cross_entropy(logits.view(-1, 2), target.view(-1), reduction='none')
        l_rec_weighted = l_rec_flat.view(batch_size, n_items) * sample_weight
        l_rec = l_rec_weighted.sum(dim=-1) / sample_weight.sum(dim=-1)  # normalized per-sample

        # --- Edge Prediction KL Term L_edge (for t >= 2) ---
        prob = F.softmax(logits, dim=-1)
        prob_1 = prob[..., 1].clamp(min=1e-7, max=1.0 - 1e-7)

        alpha_t = self.alpha_bar[t].unsqueeze(-1)
        alpha_prev = self.alpha_bar[t - 1].unsqueeze(-1)
        beta_t = self.beta[t].unsqueeze(-1)

        # True posterior: q(e^{t-1}=1 | e^t=0, e^0=1)
        denom = (1.0 - alpha_t).clamp(min=1e-5)
        q_pos = (beta_t * alpha_prev / denom).clamp(min=1e-7, max=1.0 - 1e-7)

        # Model posterior estimate
        p_pos = prob_1.clamp(min=1e-7, max=1.0 - 1e-7)

        # KL only on removed positive edges (e^t=0 AND e^0=1)
        mask_active = ((gt == 0) & (g0 == 1)).float()
        kl_edge = mask_active * (
            q_pos * (torch.log(q_pos) - torch.log(p_pos)) +
            (1.0 - q_pos) * (torch.log(1.0 - q_pos) - torch.log(1.0 - p_pos))
        )
        l_edge = kl_edge.sum(dim=-1) / mask_active.sum(dim=-1).clamp(min=1.0)

        # Optional item ranking regularization (Multinomial log-softmax over items)
        diff_logits = logits[:, :, 1] - logits[:, :, 0]
        log_prob = F.log_softmax(diff_logits, dim=-1)
        l_rank = -(log_prob * g0).sum(dim=-1).mean()

        # Combine: L_rec for all t; add L_edge for t >= 2; add rank regularization
        t_mask = (t >= 2).float()
        total_loss = l_rec + t_mask * l_edge + 0.10 * l_rank

        return total_loss.mean()

    @torch.no_grad()
    def p_sample_step(self, model, u_ids, gt, t_val):
        """
        Single reverse diffusion step: sample G^{t-1} from p(G^{t-1} | G^t).

        Uses the trained model to predict e_hat^0, then applies the true
        posterior q(e^{t-1} | e^t, e_hat^0) to sample the previous state.
        """
        batch_size = u_ids.size(0)
        device = u_ids.device
        t_tensor = torch.full((batch_size,), t_val, dtype=torch.long, device=device)

        # Get model prediction of clean graph
        logits = model(u_ids, gt, t_tensor)
        prob = F.softmax(logits, dim=-1)
        pred_g0 = prob[..., 1]  # predicted probability of edge existing in G^0

        if t_val == 1:
            # At t=1, directly return the predicted clean graph (thresholded)
            return pred_g0

        # For t >= 2, sample from posterior q(e^{t-1} | e^t, e_hat^0)
        alpha_t = self.alpha_bar[t_val]
        alpha_prev = self.alpha_bar[t_val - 1]
        beta_t = self.beta[t_val]

        # Case 1: e^t = 1 (edge currently exists)
        # q(e^{t-1}=1 | e^t=1, e^0=1) = alpha_prev
        # q(e^{t-1}=1 | e^t=1, e^0=0) = 0 (edges can't appear from nothing)
        # Combined: p(e^{t-1}=1 | e^t=1) = pred_g0 * alpha_prev
        prob_stay_1 = pred_g0 * alpha_prev

        # Case 2: e^t = 0 (edge currently removed)
        # q(e^{t-1}=1 | e^t=0, e^0=1) = beta_t * alpha_prev / (1 - alpha_t)
        # q(e^{t-1}=1 | e^t=0, e^0=0) = 0
        denom = max(1.0 - alpha_t, 1e-5)
        restore_prob = min(beta_t * alpha_prev / denom, 1.0)
        prob_restore_1 = pred_g0 * restore_prob

        # Combine based on current state
        prob_prev_1 = gt * prob_stay_1 + (1.0 - gt) * prob_restore_1
        prob_prev_1 = prob_prev_1.clamp(0.0, 1.0)

        # Sample from Bernoulli
        gt_prev = (torch.rand_like(prob_prev_1) < prob_prev_1).float()
        return gt_prev

    @torch.no_grad()
    def p_sample_loop(self, model, u_ids, g_init=None):
        """
        Full reverse sampling chain: G^T -> G^{T-1} -> ... -> G^0.

        Args:
            model: trained DiffGR denoising model
            u_ids: [batch_size] user IDs
            g_init: optional initial noisy graph at t=T. If None, starts from
                    the user's training interactions corrupted at t=T.
        Returns:
            pred_scores: [batch_size, n_items] continuous scores for ranking
        """
        model.eval()
        device = u_ids.device

        if g_init is None:
            # Start from empty graph (fully corrupted at t=T)
            gt = torch.zeros(u_ids.size(0), model.n_items, device=device)
        else:
            gt = g_init

        # Walk backward from t=T to t=1
        for t_val in range(self.steps, 0, -1):
            gt = self.p_sample_step(model, u_ids, gt, t_val)

        # gt now holds the predicted clean graph probabilities (from t=1 step)
        return gt
