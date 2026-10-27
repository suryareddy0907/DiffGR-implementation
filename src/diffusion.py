"""
Discrete Diffusion Process for DiffGR.
Implements:
1. Cosine noise scheduler (Eq. 3)
2. Discrete forward corruption q(G^t | G^0) via Bernoulli edge removal (Eq. 1-2)
3. Posterior transition q(G^{t-1} | G^t, G^0) (Eq. 4)
4. ELBO loss computation (reconstruction term + edge prediction KL term) (Eq. 5-6)
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
        
        Args:
            g0: Tensor of shape [batch_size, n_items], binary interaction vector {0, 1}.
            t: Tensor of shape [batch_size], discrete time steps in {1, ..., T}.
        Returns:
            gt: Tensor of shape [batch_size, n_items], noisy binary vector.
        """
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)  # [batch_size, 1]
        keep_prob = alpha_bar_t * g0  # Only positive interactions have non-zero keep_prob
        # Bernoulli sampling
        rand = torch.rand_like(g0)
        gt = (rand < keep_prob).float()
        return gt

    def compute_elbo_loss(self, model, u_ids, g0):
        """
        Compute DiffGR ELBO loss (Eq. 5 & 6):
        - Sample random time step t ~ Uniform(1, T) for each user in batch
        - Forward noise to obtain G^t
        - Denoiser predicts e_hat_0 = model(u_ids, gt, t)
        - Computes cross-entropy reconstruction loss and edge prediction KL divergence
        """
        batch_size = g0.size(0)
        # Sample t in {1, ..., T}
        t = torch.randint(1, self.steps + 1, (batch_size,), device=self.device)
        
        # Corrupt G^0 to G^t
        gt = self.q_sample(g0, t)
        
        # Model prediction: logits for class 0 (no edge) and class 1 (edge exists)
        # logits shape: [batch_size, n_items, 2]
        logits = model(u_ids, gt, t)
        prob = F.softmax(logits, dim=-1)  # [batch_size, n_items, 2]
        prob_0 = prob[..., 0].clamp(min=1e-7, max=1.0)
        prob_1 = prob[..., 1].clamp(min=1e-7, max=1.0)
        
        target = g0.long()  # {0, 1}
        
        # 1. Reconstruction Loss L_rec (Eq. 6)
        # Cross entropy loss over all edges
        l_rec = F.cross_entropy(logits.view(-1, 2), target.view(-1), reduction='none')
        l_rec = l_rec.view(batch_size, -1).mean(dim=-1)  # per-sample mean
        
        # 2. Edge Prediction Term L_edge (KL Divergence for t >= 2)
        # q(G^{t-1} | G^t, G^0) vs p_theta(G^{t-1} | G^t)
        # We compute this for samples where t >= 2
        alpha_t = self.alpha_bar[t].unsqueeze(-1)
        alpha_prev = self.alpha_bar[t - 1].unsqueeze(-1)
        beta_t = self.beta[t].unsqueeze(-1)
        
        # True posterior q(e^{t-1}=1 | e^t=0, e^0=1) = beta_t * alpha_prev / (1 - alpha_t)
        q_pos = (beta_t * alpha_prev / (1.0 - alpha_t).clamp(min=1e-5)).clamp(0.0, 1.0)
        # Model estimated posterior p(e^{t-1}=1 | e^t=0) using predicted prob_1
        p_pos = (prob_1 * q_pos).clamp(min=1e-7, max=1.0 - 1e-7)
        
        # KL divergence where e^t == 0 and e^0 == 1
        mask_active = ((gt == 0) & (g0 == 1)).float()
        kl_edge = mask_active * (
            q_pos * (torch.log(q_pos.clamp(min=1e-7)) - torch.log(p_pos)) +
            (1.0 - q_pos) * (torch.log((1.0 - q_pos).clamp(min=1e-7)) - torch.log(1.0 - p_pos))
        )
        l_edge = kl_edge.mean(dim=-1)
        
        # For t == 1, loss is pure L_rec; for t >= 2, loss includes L_edge
        t_mask = (t >= 2).float()
        total_loss = l_rec + t_mask * l_edge
        
        return total_loss.mean()
