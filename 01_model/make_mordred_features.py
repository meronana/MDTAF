"""Compute Mordred descriptors for every CV molecule and save to features/mordred_descriptors.npz.

PKSmart used Morgan FP + Mordred descriptors, so this prepares the same representation for the comparison group.
Computes 1613 descriptors, far more than the 203 RDKit descriptors.
Missing and constant columns are dropped; no standardisation here (it must use per-fold train statistics to avoid leakage).
"""
import os
import time

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'cv_datasets')
OUT = os.path.join(ROOT, 'features', 'mordred_descriptors.npz')


def main():
    smis = set()
    for prop in ['fu', 'clearance', 'half_life']:
        for fold in range(5):
            for split in ('source', 'target_train', 'target_test'):
                p = os.path.join(DATA_DIR, f'fold_{fold}', f'{prop}_{split}.csv')
                smis |= set(pd.read_csv(p, low_memory=False)['smiles'].astype(str))
    smis = sorted(smis)

    calc = Calculator(descriptors, ignore_3D=True)
    print(f'{len(smis)} molecules × {len(calc.descriptors)} descriptors', flush=True)

    mols, keys = [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
            keys.append(s)

    # Computing everything at once makes an object-dtype DataFrame of several GB and runs out of memory.
    # Compute in chunks and convert to float immediately while accumulating.
    t0, CH, parts = time.time(), 2000, []
    for i in range(0, len(mols), CH):
        df = calc.pandas(mols[i:i + CH], nproc=1, quiet=True)
        parts.append(np.nan_to_num(df.apply(pd.to_numeric, errors='coerce').to_numpy(dtype='float32'),
                                   nan=0.0, posinf=0.0, neginf=0.0))
        del df
        print(f'  {min(i + CH, len(mols))}/{len(mols)}  ({time.time() - t0:.0f}s)', flush=True)
    X = np.vstack(parts).astype('float64')
    del parts

    names = np.array([str(d) for d in calc.descriptors])
    keep = X.std(axis=0) > 0
    X, names = X[:, keep].astype('float32'), names[keep]

    np.savez_compressed(OUT, keys=np.array(keys), X=X, names=names)
    print(f'\nSaved: {OUT}')
    print(f'  {X.shape[0]} molecules × {X.shape[1]} descriptors ({int((~keep).sum())} constant removed)')
    print(f'  total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
