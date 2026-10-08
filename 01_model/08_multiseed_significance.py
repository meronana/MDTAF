"""Multi-seed significance test of DA methods against the baseline.

- Each method uses the λ selected by val R² in 07 (results/comparison/best_performance.json)
- Outer folds (scaffold 5-fold) and the inner scaffold val split are fixed; only model init and batch order change with the seed
- Models and training functions are taken by executing each notebook's code cells as-is (no duplicated definitions)
- Test: average over seeds within each fold, then a paired comparison with the baseline (n=5 folds)
    * paired t-test (primary) + Wilcoxon signed-rank (reference; minimum p=0.0625 at n=5)
    * Holm correction over the 15 property×method comparisons
    * Secondary: fraction of the 50 (fold, seed) pairs that beat the baseline

Output: results/significance/{runs.csv, summary.csv, predictions.csv}  (path can be changed with SIG_OUT_DIR)
- predictions.csv: per-compound test predictions for every run (y_true and y_pred in per-fold z-score units)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_rel, wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT, '01_model')
OUT_DIR = os.environ.get('SIG_OUT_DIR', os.path.join(ROOT, 'results', 'significance'))
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, MODEL_DIR)
from train_utils import split_train_val, fit_early_stopping, target_smiles  # noqa: E402

PROPERTIES = ['clearance', 'half_life', 'fu']
N_FOLDS = 5
SEEDS = list(range(int(os.environ.get('DA_SEEDS', 10))))
# Batch size is hard-coded to 32 in the notebook code, so it is injected here to change it.
# A larger batch means fewer steps per epoch, so max_epochs and patience are scaled up to keep the total number of steps.
BATCH = int(os.environ.get('DA_BATCH', 32))
_SCALE = BATCH / 32
MAX_EPOCHS, PATIENCE = int(80 * _SCALE), max(int(15 * _SCALE), 15)

NOTEBOOKS = {
    'Baseline': '01_baseline_target_only.ipynb',
    'MMD': '02_domain_adaptation_mmd_tuning.ipynb',
    'CORAL': '03_domain_adaptation_coral_tuning.ipynb',
    'DANN': '04_domain_adaptation_dann_tuning.ipynb',
    'CDAN': '05_domain_adaptation_cdan_tuning.ipynb',
    'Importance Weighting': '06_domain_adaptation_importance_weighting_tuning.ipynb',
}


def load_namespace(nb_file):
    """Namespace obtained by running only the code cells before the training-loop cell (config, model, loader, training functions)."""
    nb = json.load(open(os.path.join(MODEL_DIR, nb_file), encoding='utf-8'))
    ns = {}
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        if 'fit_early_stopping(' in src and 'for prop in PROPERTIES' in src:
            break
        for a, b in (('range(0, n, 32)', 'range(0, n, BATCH)'),
                     ('range(0, len(X_batch), 32)', 'range(0, len(X_batch), BATCH)'),
                     ('perm[i:i + 32]', 'perm[i:i + BATCH]')):
            src = src.replace(a, b)
        ns['BATCH'] = BATCH
        exec(src, ns)
    return ns


def metrics_of(m):
    return {'r': m.get('r', m.get('pearson_r')), 'r2': m['r2'],
            'rho': m.get('rho', m.get('spearman_rho')), 'rmse': m['rmse'], 'mae': m['mae']}


def run_one(method, ns, prop, fold, seed, lam, iw_cache):
    dev = ns['DEVICE']
    if method == 'Baseline':
        X_tr, y_tr = ns['load_fold_data'](prop, fold, 'train')
        X_te, y_te = ns['load_fold_data'](prop, fold, 'test')
        ev = lambda m, X, y: ns['evaluate'](m, X, y, dev)[0]
    else:
        X_src, y_src = ns['load_source'](prop, fold)
        X_tr, y_tr = ns['load_target'](prop, fold, 'train')
        X_te, y_te = ns['load_target'](prop, fold, 'test')
        X_src, y_src = np.asarray(X_src, 'float32'), np.asarray(y_src, 'float32')
        ev = lambda m, X, y: ns['evaluate'](m, X, y, dev)

    sm = target_smiles(ns['DATA_DIR'], prop, fold, 'train', ns['smiles_to_emb'])
    X_tr, y_tr, X_va, y_va = split_train_val(np.asarray(X_tr, 'float32'), np.asarray(y_tr, 'float32'), fold, smiles=sm)
    X_te, y_te = np.asarray(X_te, 'float32'), np.asarray(y_te, 'float32')

    torch.manual_seed(seed * 100 + fold)
    np.random.seed(seed * 100 + fold)

    reduction = 'none' if method == 'Importance Weighting' else 'mean'
    criterion = nn.HuberLoss(delta=1.0, reduction=reduction)

    if method == 'Baseline':
        model = ns['DirectRegressionModel']().to(dev)
        step = lambda: ns['train_epoch'](model, X_tr, y_tr, opt, criterion, dev)
    elif method in ('MMD', 'CORAL', 'DANN', 'CDAN'):
        cls = {'MMD': 'DomainAdaptationModel', 'CORAL': 'DomainAdaptationModel',
               'DANN': 'DANNModel', 'CDAN': 'CDANModel'}[method]
        fn = {'MMD': 'train_epoch_mmd', 'CORAL': 'train_epoch_coral',
              'DANN': 'train_epoch_dann', 'CDAN': 'train_epoch_cdan'}[method]
        model = ns[cls]().to(dev)
        step = lambda: ns[fn](model, X_src, y_src, X_tr, y_tr, opt, criterion, lam, dev)
    else:
        if (prop, fold) not in iw_cache:
            iw_cache[(prop, fold)] = ns['compute_importance_weights'](X_src, X_tr)
        w = iw_cache[(prop, fold)]
        model = ns['ImportanceWeightingModel']().to(dev)
        step = lambda: ns['train_epoch_iw'](model, X_src, y_src, w, X_tr, y_tr, opt, criterion, lam, dev)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_epoch, _ = fit_early_stopping(model, step, ev, X_va, y_va,
                                       max_epochs=MAX_EPOCHS, patience=PATIENCE)

    # Per-compound test predictions (eval mode, no randomness → metrics unaffected). DA models return an (output, feature, ...) tuple
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X_te, device=dev))
    y_pred = (out[0] if isinstance(out, tuple) else out).cpu().numpy()
    preds = pd.DataFrame({'smiles': target_smiles(ns['DATA_DIR'], prop, fold, 'test', ns['smiles_to_emb']),
                          'y_true': y_te, 'y_pred': y_pred})
    return {**metrics_of(ev(model, X_te, y_te)), 'best_epoch': best_epoch}, preds


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    best = json.load(open(os.path.join(ROOT, 'results', 'comparison', 'best_performance.json')))
    runs_path = os.path.join(OUT_DIR, 'runs.csv')
    preds_path = os.path.join(OUT_DIR, 'predictions.csv')
    rows, pred_frames = [], []
    t0 = time.time()
    print(f'batch={BATCH}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}  seeds={len(SEEDS)}', flush=True)
    for method, nb_file in NOTEBOOKS.items():
        ns = load_namespace(nb_file)
        iw_cache = {}
        for prop in PROPERTIES:
            lam = None if method == 'Baseline' else best[prop][method]['best_weight']
            for seed in SEEDS:
                for fold in range(N_FOLDS):
                    r, preds = run_one(method, ns, prop, fold, seed, lam, iw_cache)
                    rows.append({'method': method, 'property': prop, 'lambda': lam, 'seed': seed, 'fold': fold, **r})
                    pred_frames.append(preds.assign(method=method, property=prop, seed=seed, fold=fold))
            print(f'{method:<22} {prop:<10} λ={lam} done ({time.time() - t0:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(runs_path, index=False)
        pd.concat(pred_frames, ignore_index=True).to_csv(preds_path, index=False)

    runs = pd.DataFrame(rows)
    summary = []
    for prop in PROPERTIES:
        base = runs[(runs.method == 'Baseline') & (runs.property == prop)]
        base_fold = base.groupby('fold')[['r2', 'rho']].mean()
        base_pair = base.set_index(['seed', 'fold'])['r2']
        for method in NOTEBOOKS:
            d = runs[(runs.method == method) & (runs.property == prop)]
            per_seed = d.groupby('seed')['r2'].mean()  # distribution over seeds of the 5-fold mean R²
            fold_mean = d.groupby('fold')[['r2', 'rho']].mean()
            row = {
                'property': prop, 'method': method, 'lambda': d['lambda'].iloc[0],
                'r2_mean': d['r2'].mean(), 'r2_seed_sd': per_seed.std(),
                'rho_mean': d['rho'].mean(), 'rmse_mean': d['rmse'].mean(),
            }
            if method != 'Baseline':
                diff = fold_mean['r2'] - base_fold['r2']
                row['delta_r2'] = diff.mean()
                row['delta_rho'] = (fold_mean['rho'] - base_fold['rho']).mean()
                row['p_ttest'] = ttest_rel(fold_mean['r2'], base_fold['r2']).pvalue
                row['p_wilcoxon'] = wilcoxon(fold_mean['r2'], base_fold['r2']).pvalue
                pair = d.set_index(['seed', 'fold'])['r2']
                row['win_rate'] = (pair > base_pair.loc[pair.index]).mean()
            summary.append(row)

    summary = pd.DataFrame(summary)
    mask = summary.method != 'Baseline'
    summary.loc[mask, 'p_holm'] = holm(summary.loc[mask, 'p_ttest'])
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
