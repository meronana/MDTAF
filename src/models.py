"""Encoder + head. Deliberately mirrors the eventual Graphormer setup:

    features -> Encoder -> z -> Head -> logit

so the DA losses (MMD / CORAL / MDA) can be attached to `z` and the encoder can
later be swapped for Graphormer without touching the training loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_dim: int, dims=(512, 256), dropout: float = 0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class Model(nn.Module):
    """Encoder + single binary head. Returns (logit, z) — z is the DA hook."""

    def __init__(self, in_dim: int, enc_dims=(512, 256), dropout=0.3):
        super().__init__()
        self.encoder = Encoder(in_dim, enc_dims, dropout)
        self.head = Head(self.encoder.out_dim, dropout=dropout)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z
