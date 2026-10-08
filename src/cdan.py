"""CDAN (Conditional Domain Adversarial Network, Long et al. 2018).

Marginal alignment (MMD/CORAL) added nothing on these pairs because we matched
the label marginals by construction — the residual gap is conditional, P(y|x).
CDAN aligns the JOINT distribution of features and classifier predictions by
feeding a domain discriminator the multilinear map  f ⊗ g  (features outer
predictions), trained adversarially via a gradient-reversal layer.

CDAN+E additionally down-weights uncertain (high-entropy) examples.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coeff * grad, None


def grad_reverse(x, coeff=1.0):
    return GradientReversal.apply(x, coeff)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def multilinear_map(feature: torch.Tensor, softmax_out: torch.Tensor) -> torch.Tensor:
    """f ⊗ g flattened -> (B, feat_dim * num_classes)."""
    b = feature.size(0)
    op = torch.bmm(softmax_out.unsqueeze(2), feature.unsqueeze(1))  # (B, C, F)
    return op.view(b, -1)


def entropy_weights(softmax_out: torch.Tensor) -> torch.Tensor:
    """CDAN+E: weight = 1 + exp(-entropy), normalised per batch."""
    eps = 1e-6
    ent = -(softmax_out * torch.log(softmax_out + eps)).sum(1)
    w = 1.0 + torch.exp(-ent)
    return w / w.mean().clamp_min(eps)


def cdan_loss(disc, feat_s, logit_s, feat_t, logit_t, coeff=1.0, use_entropy=True):
    """Conditional adversarial domain loss (encoder side, via gradient reversal)."""
    def probs2(logit):                      # 1-logit sigmoid -> 2-class softmax form
        p = torch.sigmoid(logit).unsqueeze(1)
        return torch.cat([1 - p, p], dim=1)

    g_s, g_t = probs2(logit_s), probs2(logit_t)
    T_s = multilinear_map(feat_s, g_s)
    T_t = multilinear_map(feat_t, g_t)
    T = torch.cat([T_s, T_t], 0)
    d = disc(grad_reverse(T, coeff))
    dom = torch.cat([torch.ones(len(T_s)), torch.zeros(len(T_t))]).to(T.device)

    if use_entropy:
        w = torch.cat([entropy_weights(g_s.detach()), entropy_weights(g_t.detach())])
        loss = nn.functional.binary_cross_entropy_with_logits(d, dom, weight=w)
    else:
        loss = nn.functional.binary_cross_entropy_with_logits(d, dom)
    return loss
