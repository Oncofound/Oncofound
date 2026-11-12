"""cls_model.py - gated-attention classifier (matching proven results_0807 architecture)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPooler(nn.Module):
    def __init__(self, M: int, L: int):
        super().__init__()
        self.V = nn.Sequential(nn.LayerNorm(M, eps=1e-6), nn.Linear(M, L), nn.Tanh())
        self.U = nn.Sequential(nn.LayerNorm(M, eps=1e-6), nn.Linear(M, L), nn.Sigmoid())
        self.w = nn.Linear(L, 1)

    def forward(self, H: torch.Tensor):
        A = self.w(self.V(H) * self.U(H)).transpose(1, 2)   # (B, 1, N)
        A = F.softmax(A, dim=2)
        Z = torch.bmm(A, H).squeeze(1)
        return Z


class GateFusionClassifier(nn.Module):
    """
    Gated-attention pooling over ALL patches (no forced global/local split),
    followed by a 2-layer MLP head. Replaces the old MeanPoolClassifier
    that achieved good results on cls.
    """
    def __init__(self, emb_dim: int, hidden_dim: int, n_cls: int, dropout: float):
        super().__init__()
        self.attn = GatedAttentionPooler(M=emb_dim, L=hidden_dim)
        self.post = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_cls),
        )

    def forward(self, x: torch.Tensor):
        Z = self.attn(x)
        Z = self.post(Z)
        return self.head(Z)
