"""Featurisation. Swappable: today ECFP4+physchem, later Graphormer embeddings.

Everything downstream consumes an (N, D) float32 matrix, so replacing this
module with a Graphormer feature bank does not touch the DA code.
"""
from __future__ import annotations

import os

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors

RDLogger.DisableLog("rdApp.*")

N_BITS = 2048
PHYS_FNS = [
    Descriptors.MolWt, Crippen.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.FractionCSP3,
    Descriptors.NumAromaticRings, Descriptors.HeavyAtomCount,
    Descriptors.RingCount,
]


def featurize_one(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS)
    bits = np.zeros((N_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, bits)
    phys = np.array([f(mol) for f in PHYS_FNS], dtype=np.float32)
    phys = np.nan_to_num(phys, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([bits, phys])


def featurize(smiles_list, cache_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, keep_mask). keep_mask marks SMILES that parsed."""
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["X"], z["keep"]
    feats, keep = [], []
    for s in smiles_list:
        v = featurize_one(s)
        keep.append(v is not None)
        if v is not None:
            feats.append(v)
    X = np.stack(feats).astype(np.float32) if feats else np.zeros((0, N_BITS + len(PHYS_FNS)), "float32")
    keep = np.array(keep, dtype=bool)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X, keep=keep)
    return X, keep
