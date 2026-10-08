"""Compute RDKit descriptors for every CV molecule and save to features/rdkit_descriptors.npz.

In the QSAR baseline (15), RDKit descriptors were consistently better than Graphormer embeddings,
so this prepares for repeating the DA experiments on this representation to check whether the conclusions depend on it.

Contents
- keys : SMILES (exactly as written in the CV csv — same convention as the existing embedding npz)
- X    : descriptor matrix (zero-variance columns removed, NaN/inf set to 0)
- names: names of the remaining descriptors

No standardisation here (it must use per-fold train statistics to avoid leakage, so training code handles it).
"""
import os
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'cv_datasets')
OUT = os.path.join(ROOT, 'features', 'rdkit_descriptors.npz')
PROPERTIES = ['fu', 'clearance', 'half_life']
N_FOLDS = 5
DESC = list(Descriptors.descList)


def all_smiles():
    s = set()
    for prop in PROPERTIES:
        for fold in range(N_FOLDS):
            for split in ('source', 'target_train', 'target_test'):
                p = os.path.join(DATA_DIR, f'fold_{fold}', f'{prop}_{split}.csv')
                s |= set(pd.read_csv(p, low_memory=False)['smiles'].astype(str))
    return sorted(s)


def main():
    smis = all_smiles()
    print(f'{len(smis)} unique molecules, {len(DESC)} descriptors — starting', flush=True)

    keys, rows, t0 = [], [], time.time()
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        v = []
        for _, fn in DESC:
            try:
                v.append(fn(m))
            except Exception:
                v.append(np.nan)
        keys.append(s)
        rows.append(v)
        if (i + 1) % 5000 == 0:
            print(f'  {i+1}/{len(smis)}  ({time.time()-t0:.0f}s)', flush=True)

    X = np.nan_to_num(np.asarray(rows, dtype='float64'), nan=0.0, posinf=0.0, neginf=0.0)
    names = np.array([n for n, _ in DESC])
    keep = X.std(axis=0) > 0                      # drop zero-variance descriptors
    X, names = X[:, keep].astype('float32'), names[keep]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, keys=np.array(keys), X=X, names=names)
    print(f'\nSaved: {OUT}')
    print(f'  {X.shape[0]} molecules × {X.shape[1]} descriptors ({int((~keep).sum())} zero-variance removed)')
    print(f'  total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
