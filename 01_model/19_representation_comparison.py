"""Representation ablation: starting from Full (all four representations), remove one at a time to measure each one's contribution.

Representations
- MorganFP   : Morgan/ECFP4 fingerprint, 2048 bits (structural patterns)
- Mordred    : 1445 Mordred descriptors (physicochemical)
- RDKit      : 203 RDKit descriptors (physicochemical, largely overlapping with Mordred)
- Graphormer : 768-d graph embedding pretrained on PCQM4Mv2 (learned structural representation)

Compared settings
- each representation alone (4)
- Full = all four representations (4464-d)
- Full − X : remove one at a time (leave-one-out) → unique contribution of that representation
- MorganFP+Mordred : the combination used by PKSmart

Runs on GPU (XGBoost device='cuda'). Descriptors are read from precomputed npz files, so the Mordred package is not needed.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
from xgboost import XGBRegressor, XGBRFRegressor
warnings.filterwarnings('ignore'); RDLogger.DisableLog('rdApp.*')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data', 'cv_datasets')
OUT = os.path.join(ROOT, 'results', 'representation')
os.makedirs(OUT, exist_ok=True)
PROPS = ['fu', 'clearance', 'half_life']; SEEDS = [0, 1, 2]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('xgboost device:', DEV, flush=True)

def load_npz(name):
    d = np.load(os.path.join(ROOT, 'features', name), allow_pickle=True)
    X, k = d['X'], d['keys']          # lazy → extract before the loop
    return {s: X[i] for i, s in enumerate(k)}

TAB = {'Graphormer': load_npz('graphormer_cv_embeddings.npz'),
       'RDKit': load_npz('rdkit_descriptors.npz'),
       'Mordred': load_npz('mordred_descriptors.npz')}
_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_fpc = {}
def ecfp(s):
    if s not in _fpc:
        _fpc[s] = np.asarray(_fp.GetFingerprint(Chem.MolFromSmiles(str(s))), dtype='float32')
    return _fpc[s]

def feat(smiles, kind):
    if kind == 'MorganFP': return np.vstack([ecfp(s) for s in smiles])
    return np.stack([TAB[kind][s] for s in smiles]).astype('float32')

def load(prop, fold, split):
    d = pd.read_csv(os.path.join(DATA, f'fold_{fold}', f'{prop}_target_{split}.csv'))
    ok = set(TAB['Mordred']) & set(TAB['RDKit']) & set(TAB['Graphormer'])
    keep = [(s, v) for s, v in zip(d['smiles'].astype(str), d['value']) if s in ok]
    return [s for s, _ in keep], np.array([v for _, v in keep], dtype='float32')

ALL4 = ['MorganFP', 'Mordred', 'RDKit', 'Graphormer']
REPS = {k: [k] for k in ALL4}                                   # single
REPS['Full'] = ALL4                                             # all
for k in ALL4:
    REPS[f'Full − {k}'] = [x for x in ALL4 if x != k]           # leave-one-out
REPS['MorganFP+Mordred'] = ['MorganFP', 'Mordred']              # PKSmart combination

def scale(Xtr, Xte):
    # Some descriptor columns reach 10^30, so quantile scaling is the safe choice
    c = lambda A: np.nan_to_num(A.astype('float64'), nan=0., posinf=0., neginf=0.)
    lo, hi = np.percentile(c(Xtr), [1, 99], axis=0)
    sp = np.maximum(hi - lo, 1e-8)
    f = lambda A: np.clip((c(A) - lo) / sp, -5, 5).astype('float32')
    return f(Xtr), f(Xte)

rows, t0 = [], time.time()
for prop in PROPS:
    for fold in range(5):
        tr_s, tr_y = load(prop, fold, 'train'); te_s, te_y = load(prop, fold, 'test')
        parts_tr = {k: feat(tr_s, k) for k in ('MorganFP', 'RDKit', 'Mordred', 'Graphormer')}
        parts_te = {k: feat(te_s, k) for k in ('MorganFP', 'RDKit', 'Mordred', 'Graphormer')}
        for name, ks in REPS.items():
            Xtr = np.hstack([parts_tr[k] for k in ks]); Xte = np.hstack([parts_te[k] for k in ks])
            if any(k in ('RDKit', 'Mordred') for k in ks):
                Xtr, Xte = scale(Xtr, Xte)
            for mdl in ('RF', 'XGB'):
                for seed in SEEDS:
                    M = (XGBRFRegressor(n_estimators=300, max_depth=8, max_bin=64, subsample=0.8,
                                        colsample_bynode=0.3, random_state=seed, tree_method='hist', device=DEV, verbosity=0)
                         if mdl == 'RF' else
                         XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, max_bin=64,
                                      subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                      tree_method='hist', device=DEV, verbosity=0))
                    p = M.fit(Xtr, tr_y).predict(Xte)
                    rows.append({'property': prop, 'fold': fold, 'rep': name, 'model': mdl, 'seed': seed,
                                 'n_feat': Xtr.shape[1], 'r2': r2_score(te_y, p),
                                 'r': pearsonr(te_y, p)[0], 'rho': spearmanr(te_y, p)[0],
                                 'mae': np.mean(np.abs(te_y - p)),
                                 'rmse': np.sqrt(np.mean((te_y - p) ** 2))})
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'runs.csv'), index=False)
    print(f'  {prop} done ({time.time()-t0:.0f}s)', flush=True)

d = pd.DataFrame(rows)
print('\n== R² (3 seeds × 5 folds, target-only)')
print(d.groupby(['property', 'rep']).r2.mean().unstack('property')[PROPS].round(4).to_string())
print(f'\nTotal {time.time()-t0:.0f}s')
