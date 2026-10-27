"""
DiffGR Denoising Network Architecture.
Implements the 4 modules shown in Figure 1b:
1. Sub-graph Encoder: learns latent representation emb_{G^t} of noisy graph G^t.
2. ID Embedding Module: User and Item embedding lookup tables.
3. User & Item Embedding Mergers:
   - User merger: concatenates emb_u, emb_{G^t}, and sinusoidal time embedding emb_t.
   - Item merger: transforms item embeddings Emb_I to Emb_{final_I}.
4. Edge Predictor:
   - Computes edge probabilities via Hadamard product / bilinear edge projection.
   - Predicts 2-class logits for each edge (0: no edge, 1: edge exists).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion time step t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [batch_size]
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1) if half_dim > 1 else 1.0
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffGRModel(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, hidden_dim=256, dropout=0.1, linear_item_merger=False):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        # 1. ID Embedding Module
        self.user_embeddings = nn.Embedding(n_users, emb_dim)
        self.item_embeddings = nn.Embedding(n_items, emb_dim)

        # 2. Sub-graph Encoder: encodes G^t [batch_size, n_items] -> [batch_size, emb_dim]
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        # 3. Time Step Embedding
        self.time_embed = SinusoidalTimeEmbedding(emb_dim)

        # 4. User Embedding Merger: concatenates [emb_u, emb_Gt, emb_t] (3 * emb_dim) -> emb_dim
        self.user_merger = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        # 5. Item Embedding Merger: transforms Emb_I -> Emb_final_I
        if linear_item_merger:
            self.item_merger = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim)
            )
        else:
            self.item_merger = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, emb_dim),
                nn.LayerNorm(emb_dim)
            )

        # 6. Edge Predictor parameters for class 0 (no edge) and class 1 (edge exists)
        # Using Hadamard product (emb_u * w) @ emb_i.T + bias
        self.w_pos = nn.Parameter(torch.ones(emb_dim))
        self.w_neg = nn.Parameter(-torch.ones(emb_dim))
        self.bias_pos = nn.Parameter(torch.zeros(1))
        self.bias_neg = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.w_pos, mean=1.0, std=0.02)
        nn.init.normal_(self.w_neg, mean=-1.0, std=0.02)

    def load_state_dict(self, state_dict, strict=True):
        # Auto-adapt item_merger architecture based on checkpoint weights
        if "item_merger.0.weight" in state_dict and "item_merger.4.weight" not in state_dict:
            ckpt_shape = state_dict["item_merger.0.weight"].shape
            if ckpt_shape == (self.emb_dim, self.emb_dim):
                device = self.user_embeddings.weight.device
                self.item_merger = nn.Sequential(
                    nn.Linear(self.emb_dim, self.emb_dim),
                    nn.LayerNorm(self.emb_dim)
                ).to(device)
        elif "item_merger.4.weight" in state_dict and len(self.item_merger) == 2:
            hidden_dim = state_dict["item_merger.0.weight"].shape[0]
            device = self.user_embeddings.weight.device
            self.item_merger = nn.Sequential(
                nn.Linear(self.emb_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, self.emb_dim),
                nn.LayerNorm(self.emb_dim)
            ).to(device)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, u_ids, gt, t):
        """
        Args:
            u_ids: [batch_size] long tensor of user IDs.
            gt: [batch_size, n_items] float tensor of noisy interaction vector.
            t: [batch_size] long tensor of discrete time steps.
        Returns:
            logits: [batch_size, n_items, 2] predictions for each user-item pair.
        """
        # Module 1: Sub-graph representation
        emb_gt = self.encoder(gt)  # [batch_size, emb_dim]

        # Module 2: User ID & Time embeddings
        emb_u = self.user_embeddings(u_ids)  # [batch_size, emb_dim]
        emb_t = self.time_embed(t)            # [batch_size, emb_dim]

        # Module 3: User Merger
        user_cat = torch.cat([emb_u, emb_gt, emb_t], dim=-1)  # [batch_size, emb_dim * 3]
        emb_final_u = self.user_merger(user_cat)               # [batch_size, emb_dim]

        # Module 4: Item Merger
        all_item_ids = torch.arange(self.n_items, device=u_ids.device)
        emb_i = self.item_embeddings(all_item_ids)            # [n_items, emb_dim]
        emb_final_i = self.item_merger(emb_i)                 # [n_items, emb_dim]

        # Module 5 & 6: Fast Hadamard + Linear edge prediction
        # (emb_final_u * w) @ emb_final_i.T
        logits_neg = torch.matmul(emb_final_u * self.w_neg, emb_final_i.T) + self.bias_neg
        logits_pos = torch.matmul(emb_final_u * self.w_pos, emb_final_i.T) + self.bias_pos

        logits = torch.stack([logits_neg, logits_pos], dim=-1)  # [batch_size, n_items, 2]
        return logits

    @torch.no_grad()
    def predict_scores(self, u_ids, diffusion=None, train_dict=None):
        """
        Compute recommendation scores matching Eq. 7 of DiffGR paper:
        score_i = e^0_{i,1} - e^0_{i,0}
        """
        self.eval()
        batch_size = u_ids.size(0)
        device = u_ids.device

        # Feed the user's historical training interactions
        gt = torch.zeros(batch_size, self.n_items, device=device)
        if train_dict is not None:
            for i, uid in enumerate(u_ids.cpu().tolist()):
                items = train_dict.get(uid, [])
                if len(items) > 0:
                    gt[i, items] = 1.0

        # Section 2.2: Denoise from time step t=T (or t=steps)
        step_val = 5 if diffusion is None else getattr(diffusion, 'steps', 5)
        t = torch.full((batch_size,), step_val, dtype=torch.long, device=device)
        logits = self.forward(u_ids, gt, t)
        probs = F.softmax(logits, dim=-1)

        # Eq. 7: score_i = e^0_{i,1} - e^0_{i,0}
        scores = probs[..., 1] - probs[..., 0]
        return scores
