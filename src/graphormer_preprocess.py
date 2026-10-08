"""Graphormer input preprocessing without the Cython `algos_graphormer` module.

Reproduces transformers' preprocess_item / GraphormerDataCollator, but computes
shortest paths with scipy (Floyd-Warshall) and reconstructs multi-hop edge
inputs in pure Python. Uses OGB `smiles2graph` so node/edge features match the
PCQM4Mv2 pretraining exactly.
"""
from __future__ import annotations

import numpy as np
import torch
from ogb.utils import smiles2graph
from scipy.sparse.csgraph import shortest_path

UNREACHABLE = 510


def convert_to_single_emb(x, offset: int = 512):
    """Offset each feature column into its own embedding range (as in HF/OGB)."""
    x = np.asarray(x)
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    return x + feature_offset


def _node_path(pred_row, i, j):
    """Reconstruct node sequence i->j from a scipy predecessor row (pred[i])."""
    if i == j:
        return [i]
    seq = [j]
    while seq[-1] != i:
        p = pred_row[seq[-1]]
        if p < 0:
            return None
        seq.append(int(p))
    return seq[::-1]


def preprocess_smiles(smiles: str, multi_hop_max_dist: int = 5):
    g = smiles2graph(smiles)
    n = int(g["num_nodes"])
    if n == 0:
        return None
    node_feat = np.asarray(g["node_feat"], dtype=np.int64)          # (n, 9)
    edge_index = np.asarray(g["edge_index"], dtype=np.int64)        # (2, E)
    edge_attr = np.asarray(g["edge_feat"], dtype=np.int64)          # (E, 3)
    if edge_attr.ndim == 1:
        edge_attr = edge_attr[:, None]

    input_nodes = convert_to_single_emb(node_feat) + 1              # +1 padding shift

    attn_edge_type = np.zeros((n, n, edge_attr.shape[-1]), dtype=np.int64)
    if edge_index.shape[1] > 0:
        attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1

    adj = np.zeros((n, n), dtype=bool)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = True

    dist, pred = shortest_path(adj.astype(np.int8), method="FW",
                               directed=False, return_predecessors=True)
    dist[np.isinf(dist)] = UNREACHABLE
    sp = dist.astype(np.int64)
    sp[sp > UNREACHABLE] = UNREACHABLE
    max_dist = int(sp[sp < UNREACHABLE].max()) if (sp < UNREACHABLE).any() else 1
    max_dist = max(min(max_dist, multi_hop_max_dist), 1)   # model only uses <= multi_hop_max_dist

    # multi-hop edge input: edge features along the shortest path of each pair
    edge_dim = attn_edge_type.shape[-1]
    input_edges = np.zeros((n, n, max_dist, edge_dim), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j or sp[i, j] >= UNREACHABLE:
                continue
            seq = _node_path(pred[i], i, j)
            if seq is None:
                continue
            for k in range(len(seq) - 1):
                if k >= max_dist:
                    break
                input_edges[i, j, k] = attn_edge_type[seq[k], seq[k + 1]]

    return {
        "input_nodes": input_nodes,                                    # (n, 9)
        "in_degree": adj.sum(axis=1).astype(np.int64) + 1,             # (n,)
        "out_degree": adj.sum(axis=0).astype(np.int64) + 1,            # (n,)
        "spatial_pos": sp + 1,                                         # (n, n)
        "attn_edge_type": attn_edge_type,                             # (n, n, edge_dim)
        "input_edges": input_edges + 1,                               # (n, n, max_dist, edge_dim)
        "num_nodes": n,
    }


def collate(items, spatial_pos_max: int = 20):
    """Pad a list of preprocessed items into Graphormer batch tensors."""
    items = [x for x in items if x is not None]
    B = len(items)
    max_n = max(x["num_nodes"] for x in items)
    node_dim = items[0]["input_nodes"].shape[1]
    edge_dim = items[0]["attn_edge_type"].shape[2]
    max_d = max(x["input_edges"].shape[2] for x in items)

    input_nodes = np.zeros((B, max_n, node_dim), np.int64)
    in_deg = np.zeros((B, max_n), np.int64)
    out_deg = np.zeros((B, max_n), np.int64)
    spatial = np.zeros((B, max_n, max_n), np.int64)
    edge_type = np.zeros((B, max_n, max_n, edge_dim), np.int64)
    input_edges = np.zeros((B, max_n, max_n, max_d, edge_dim), np.int64)
    attn_bias = np.zeros((B, max_n + 1, max_n + 1), np.float32)

    for b, x in enumerate(items):
        n = x["num_nodes"]
        input_nodes[b, :n] = x["input_nodes"]
        in_deg[b, :n] = x["in_degree"]
        out_deg[b, :n] = x["out_degree"]
        spatial[b, :n, :n] = x["spatial_pos"]
        edge_type[b, :n, :n] = x["attn_edge_type"]
        d = x["input_edges"].shape[2]
        input_edges[b, :n, :n, :d] = x["input_edges"]
        # mask out pairs farther than spatial_pos_max (graph token is row/col 0)
        sp = x["spatial_pos"]
        far = sp >= spatial_pos_max
        ab = np.zeros((n + 1, n + 1), np.float32)
        ab[1:, 1:][far] = float("-inf")
        attn_bias[b, :n + 1, :n + 1] = ab

    t = lambda a: torch.from_numpy(a)
    return {
        "input_nodes": t(input_nodes), "in_degree": t(in_deg), "out_degree": t(out_deg),
        "spatial_pos": t(spatial), "attn_edge_type": t(edge_type),
        "input_edges": t(input_edges), "attn_bias": t(attn_bias),
    }
