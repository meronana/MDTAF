"""Load the pretrained PCQM4Mv2 Graphormer graph-encoder from the HF checkpoint
into the vendored nn.Module, and expose a frozen graph-embedding extractor.

The checkpoint (clefourrier/graphormer-base-pcqm4mv2) stores the encoder under
the prefix `encoder.graph_encoder.*`; we strip that and load the rest.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch

from .graphormer_vendored import GraphormerGraphEncoder

HF_BASE = "https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2/resolve/main"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


def _download(fname):
    import urllib.request
    os.makedirs(CACHE_DIR, exist_ok=True)
    dst = os.path.join(CACHE_DIR, fname)
    if not os.path.exists(dst):
        print(f"downloading {fname} ...")
        urllib.request.urlretrieve(f"{HF_BASE}/{fname}", dst)
    return dst


def load_config():
    with open(_download("config.json")) as f:
        c = json.load(f)
    # fill attributes the vendored module reads but config.json may not name
    c.setdefault("num_hidden_layers", c.get("num_layers", 12))
    c.setdefault("traceable", False)
    c.setdefault("kdim", None)
    c.setdefault("vdim", None)
    c.setdefault("embed_scale", None)
    c.setdefault("q_noise", 0.0)
    c.setdefault("qn_block_size", 8)
    return SimpleNamespace(**c)


def load_encoder(device="cpu"):
    cfg = load_config()
    enc = GraphormerGraphEncoder(cfg)
    ckpt = _download("pytorch_model.bin")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    prefix = "encoder.graph_encoder."
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = enc.load_state_dict(sub, strict=False)
    print(f"[graphormer] loaded {len(sub)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing (first 5):", missing[:5])
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, cfg


@torch.no_grad()
def embed(enc, batch, device="cpu"):
    """Return graph-token embeddings (B, embedding_dim) for a collated batch."""
    batch = {k: v.to(device) for k, v in batch.items()}
    inner_states, graph_rep = enc(
        batch["input_nodes"], batch["input_edges"], batch["attn_bias"],
        batch["in_degree"], batch["out_degree"], batch["spatial_pos"],
        batch["attn_edge_type"],
    )
    return graph_rep.detach().cpu()
