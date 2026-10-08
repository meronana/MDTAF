"""Domain-adaptation alignment losses, computed on encoder features z.

All take (z_source, z_target) -> scalar, and are minimised jointly with the
supervised source loss. They align the marginal feature distribution P(z), i.e.
they address COVARIATE shift (which we verified exists for these pairs).

  mmd    : multi-kernel RBF Maximum Mean Discrepancy (Long et al., DAN).
  coral  : second-order (covariance) alignment (Sun & Saenko, Deep CORAL).
  mda    : "minimum-discrepancy alignment" here = mean (first-order) + CORAL
           (second-order), a light MMD-free moment matcher.

Kept dependency-free and differentiable so the encoder receives gradients.
"""
from __future__ import annotations

import torch


def _pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2 = (a * a).sum(1, keepdim=True)
    b2 = (b * b).sum(1, keepdim=True)
    return (a2 + b2.t() - 2.0 * a @ b.t()).clamp_min(0.0)


def mmd_rbf(zs: torch.Tensor, zt: torch.Tensor,
            bandwidths=(0.5, 1, 2, 4, 8)) -> torch.Tensor:
    """Multi-kernel RBF MMD^2, bandwidths scaled by the median heuristic."""
    zz = torch.cat([zs, zt], 0)
    d2 = _pairwise_sq_dists(zz, zz)
    med = d2.detach()[d2.detach() > 0].median().clamp_min(1e-8)
    n = zs.size(0)
    k = torch.zeros_like(d2)
    for mul in bandwidths:
        k = k + torch.exp(-d2 / (med * mul))
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    m = zt.size(0)
    # unbiased-ish: exclude diagonals on the within-domain blocks
    kxx = (kxx.sum() - kxx.diag().sum()) / max(n * (n - 1), 1)
    kyy = (kyy.sum() - kyy.diag().sum()) / max(m * (m - 1), 1)
    return kxx + kyy - 2.0 * kxy.mean()


def coral(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """Deep CORAL: squared Frobenius distance between feature covariances."""
    d = zs.size(1)
    zs = zs - zs.mean(0, keepdim=True)
    zt = zt - zt.mean(0, keepdim=True)
    cs = (zs.t() @ zs) / max(zs.size(0) - 1, 1)
    ct = (zt.t() @ zt) / max(zt.size(0) - 1, 1)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def mda(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """First-order (mean) + second-order (CORAL) moment alignment."""
    mean_gap = ((zs.mean(0) - zt.mean(0)) ** 2).mean()
    return mean_gap + coral(zs, zt)


DA_LOSSES = {"mmd": mmd_rbf, "coral": coral, "mda": mda,
             "none": lambda zs, zt: zs.new_zeros(())}


def get_da_loss(name: str):
    if name not in DA_LOSSES:
        raise ValueError(f"unknown DA loss '{name}', choose from {list(DA_LOSSES)}")
    return DA_LOSSES[name]
