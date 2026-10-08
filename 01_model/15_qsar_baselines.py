"""Standard QSAR baseline comparison.

Reference group for answering "is Graphormer embedding + DA framework better than plain QSAR?".
Uses the same scaffold 5-fold and the same target split, so results compare directly with 01-14.

Representations
- ECFP4   : Morgan radius 2, 2048 bits
- RDKit   : all RDKit descriptors (zero-variance removed + standardised)
- Graphormer: the existing 768-d embedding (to see the difference when the same model sits on another representation)

Models
- RF  : RandomForestRegressor(500)
- XGB : XGBRegressor(500, lr 0.05, depth 6)
- Ridge

Training settings
- target-only : trained on target train only (same condition as the 01 baseline)
- pooled      : source + target simply concatenated (no domain distinction — the simplest transfer)

Output: results/qsar_baselines/runs.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'cv_datasets')
OUT_DIR = os.path.join(ROOT, 'results', 'qsar_baselines')
os.makedirs(OUT_DIR, exist_ok=True)

PROPERTIES = ['fu', 'clearance', 'half_life']
N_FOLDS = 5
SEEDS = [0, 1, 2]
MAX_SOURCE = 5000           # cap on source size for pooled training (use all if smaller)

_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_emb = np.load(os.path.join(ROOT, 'features', 'graphormer_cv_embeddings.npz'), allow_pickle=True)
_X, _K = _emb['X'], _emb['keys']
SMILES_TO_EMB = {s: _X[i] for i, s in enumerate(_K)}
_DESC = [(n, f) for n, f in Descriptors.descList]


_feat_cache = {}


def _one(smi, kind):
    key = (smi, kind)
    if key not in _feat_cache:
        m = Chem.MolFromSmiles(str(smi))
        if kind == 'ECFP4':
            v = np.asarray(_fpgen.GetFingerprint(m), dtype='float32')
        else:
            v = []
            for _, fn in _DESC:
                try:
                    x = fn(m)
                except Exception:
                    x = np.nan
                v.append(x)
            v = np.asarray(v, dtype='float32')
        _feat_cache[key] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return _feat_cache[key]


def featurize(smiles, kind):
    """Per-molecule cache — the source is identical across folds, so it is not recomputed."""
    if kind == 'Graphormer':
        return np.stack([SMILES_TO_EMB[s] for s in smiles]).astype('float32')
    return np.vstack([_one(s, kind) for s in smiles])


def load_split(prop, fold, split):
    f = 'source' if split == 'source' else f'target_{split}'
    df = pd.read_csv(os.path.join(DATA_DIR, f'fold_{fold}', f'{prop}_{f}.csv'), low_memory=False)
    keep = [(s, v) for s, v in zip(df['smiles'].astype(str), df['value']) if s in SMILES_TO_EMB]
    return [s for s, _ in keep], np.array([v for _, v in keep], dtype='float32')


def make_model(name, seed):
    if name == 'RF':
        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=2)
    if name == 'XGB':
        return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0)
    return Ridge(alpha=1.0, random_state=seed)


def metrics(y, p):
    p = np.nan_to_num(p, nan=0.0, posinf=1e6, neginf=-1e6)
    return {'r2': float(r2_score(y, p)), 'rho': float(spearmanr(y, p)[0]),
            'rmse': float(np.sqrt(np.mean((y - p) ** 2))), 'mae': float(np.mean(np.abs(y - p)))}


def main():
    reps = ['ECFP4', 'RDKit', 'Graphormer']
    models = ['RF', 'XGB', 'Ridge']
    rows, t0 = [], time.time()
    runs_path = os.path.join(OUT_DIR, 'runs.csv')

    for prop in PROPERTIES:
        for fold in range(N_FOLDS):
            tr_s, tr_y = load_split(prop, fold, 'train')
            te_s, te_y = load_split(prop, fold, 'test')
            sr_s, sr_y = load_split(prop, fold, 'source')
            if len(sr_s) > MAX_SOURCE:
                idx = np.random.RandomState(0).choice(len(sr_s), MAX_SOURCE, replace=False)
                sr_s, sr_y = [sr_s[i] for i in idx], sr_y[idx]

            feats = {}
            for rep in reps:
                Xtr, Xte, Xsr = (featurize(x, rep) for x in (tr_s, te_s, sr_s))
                if rep == 'RDKit':                      # descriptors differ widely in scale, so standardise
                    sc = StandardScaler().fit(np.vstack([Xtr, Xsr]))
                    Xtr, Xte, Xsr = sc.transform(Xtr), sc.transform(Xte), sc.transform(Xsr)
                feats[rep] = (Xtr, Xte, Xsr)

            for rep in reps:
                Xtr, Xte, Xsr = feats[rep]
                for mdl in models:
                    for setting in ('target-only', 'pooled'):
                        X = Xtr if setting == 'target-only' else np.vstack([Xtr, Xsr])
                        y = tr_y if setting == 'target-only' else np.r_[tr_y, sr_y]
                        for seed in SEEDS:
                            if mdl == 'Ridge' and seed > 0:
                                continue                # deterministic model, one seed is enough
                            m = make_model(mdl, seed).fit(X, y)
                            rows.append({'property': prop, 'fold': fold, 'rep': rep, 'model': mdl,
                                         'setting': setting, 'seed': seed, 'n_train': len(X),
                                         **metrics(te_y, m.predict(Xte))})
            pd.DataFrame(rows).to_csv(runs_path, index=False)
            print(f'  {prop:<10} fold {fold}  ({time.time() - t0:.0f}s)', flush=True)

    runs = pd.DataFrame(rows)
    summ = runs.groupby(['property', 'rep', 'model', 'setting'])[['r2', 'rho', 'rmse', 'mae']].mean()
    summ.to_csv(os.path.join(OUT_DIR, 'summary.csv'), encoding='utf-8-sig')
    pd.set_option('display.width', 200)
    print('\n== mean over folds and seeds')
    print(summ.round(4).to_string())
    print(f'\nTotal {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
