"""Partially-unfrozen Graphormer for domain adaptation.

The frozen-embedding experiments showed MMD/CORAL cannot align when only a
shallow MLP is trainable. Here the TOP-k Graphormer layers (+ final norm) are
unfrozen so the alignment loss reshapes the actual representation, while the
lower layers stay frozen (cheap + keeps pretrained structure).

    Graphormer[0:12-k] frozen ─▶ Graphormer[12-k:12] trainable ─▶ graph_rep
        ─▶ proj (trainable) ─▶ z  ──(DA loss here)──▶ head ─▶ logit
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .graphormer_encoder import load_encoder


class GraphormerFT(nn.Module):
    def __init__(self, k_unfrozen: int = 2, proj_dim: int = 256, dropout: float = 0.3,
                 device="cpu"):
        super().__init__()
        enc, cfg = load_encoder(device)   # returns frozen encoder (all requires_grad False)
        self.enc = enc
        self.cfg = cfg
        emb = cfg.embedding_dim

        # unfreeze top-k transformer layers + final layer norm
        n_layers = len(self.enc.layers)
        for li in range(max(n_layers - k_unfrozen, 0), n_layers):
            for p in self.enc.layers[li].parameters():
                p.requires_grad_(True)
        if getattr(self.enc, "final_layer_norm", None) is not None:
            for p in self.enc.final_layer_norm.parameters():
                p.requires_grad_(True)

        self.proj = nn.Sequential(
            nn.Linear(emb, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(proj_dim, 1)

    def graph_rep(self, batch):
        _, rep = self.enc(
            batch["input_nodes"], batch["input_edges"], batch["attn_bias"],
            batch["in_degree"], batch["out_degree"], batch["spatial_pos"],
            batch["attn_edge_type"],
        )
        return rep                      # (B, emb)

    def forward(self, batch):
        z = self.proj(self.graph_rep(batch))
        return self.head(z).squeeze(-1), z

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
