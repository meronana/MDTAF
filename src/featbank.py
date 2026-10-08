"""Unified feature loader so baseline/DA scripts can switch ECFP <-> Graphormer
with a single flag. Both return (X, y) aligned to rows that featurised OK.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .features import featurize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "data", "final")
FEAT = os.path.join(ROOT, "features")

_BANK = None


def _load_bank():
    global _BANK
    if _BANK is None:
        z = np.load(os.path.join(FEAT, "graphormer_bank.npz"), allow_pickle=True)
        _BANK = {str(k): v for k, v in zip(z["keys"], z["X"])}
    return _BANK


def load_split(task: str, split: str, kind: str = "ecfp"):
    df = pd.read_csv(os.path.join(FINAL, f"{task}_{split}.csv"))
    smiles = df["smiles"].astype(str).tolist()
    y = df["Y"].values.astype("float32")
    if kind == "ecfp":
        X, keep = featurize(smiles, cache_path=os.path.join(FEAT, f"{task}_{split}.npz"))
        return X, y[keep]
    elif kind == "graphormer":
        bank = _load_bank()
        keep = np.array([s in bank for s in smiles], dtype=bool)
        X = np.stack([bank[s] for s in np.asarray(smiles)[keep]]).astype("float32")
        return X, y[keep]
    raise ValueError(f"unknown feature kind '{kind}'")
