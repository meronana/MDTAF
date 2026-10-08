"""Transfer routes per endpoint: label-based (10) vs distribution-based DA (08) vs target-only.

Compared settings (all with the same scaffold 5-fold, same inner val split, seeds 0-2):
- frozen            : Baseline of 08 multi-seed (frozen Graphormer + MLP)
- target-only FT    : 09, fine-tuning the top k Graphormer layers (target only)
- pretrain -> FT    : 10, pretraining on in vitro labels then target fine-tuning
- DA (5 methods)    : 08 multi-seed (frozen embeddings + distribution alignment / reweighting)

Test: average over seeds within each fold, then paired t-test (n=5), Holm correction over all comparisons.
Endpoint characteristics: benchmark phase 04 (1-NN label correspondence) and the zero-shot ρ from 10 are shown alongside.

Output: results/transfer_comparison/{summary.csv, fold_means.csv}
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(ROOT, 'results', 'transfer_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

PROPERTIES = ['fu', 'clearance', 'half_life']   # ordered by hypothesised mechanistic distance
SEEDS = [0, 1, 2]
K = 4
DA_METHODS = ['MMD', 'CORAL', 'DANN', 'CDAN', 'Importance Weighting']


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.full_like(p, np.nan), 0.0
    ok = np.where(~np.isnan(p))[0]
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, min(1.0, (len(ok) - rank) * p[i]))
        adj[i] = running
    return adj


def load_runs():
    sig = pd.read_csv(os.path.join(ROOT, 'results', 'significance', 'runs.csv'))
    sig = sig[sig['seed'].isin(SEEDS)]
    frames = [sig.assign(model=sig['method'].replace({'Baseline': 'frozen'}))]

    ft = pd.read_csv(os.path.join(ROOT, 'results', 'finetune', 'runs.csv'))
    ft = ft[(ft['k'] == K) & ft['seed'].isin(SEEDS)]
    frames.append(ft.assign(model='target-only FT'))

    pre = pd.read_csv(os.path.join(ROOT, 'results', 'pretrain_ft', 'runs.csv'))
    pre = pre[pre['seed'].isin(SEEDS)]
    frames.append(pre.assign(model='pretrain -> FT'))

    cols = ['model', 'property', 'seed', 'fold', 'r2', 'rho']
    return pd.concat([f[cols] for f in frames], ignore_index=True), pre


def main():
    runs, pre = load_runs()
    fold_means = runs.groupby(['property', 'model', 'fold'], as_index=False)[['r2', 'rho']].mean()
    n_runs = runs.groupby(['property', 'model']).size()

    nn = json.load(open(os.path.join(ROOT, 'benchmark', 'results', 'phase_04_label_correspondence.json')))
    zero_shot = pre.groupby('property')['zero_shot_rho'].mean()

    rows = []
    for prop in PROPERTIES:
        fm = fold_means[fold_means['property'] == prop].pivot(index='fold', columns='model', values='r2')
        ref_models = {'frozen': 'frozen', 'target-only FT': 'target-only FT'}
        for model in ['frozen', 'target-only FT', 'pretrain -> FT'] + DA_METHODS:
            if model not in fm:
                continue
            row = {'property': prop, 'model': model, 'n_runs': int(n_runs.get((prop, model), 0)),
                   'n_folds': int(fm[model].notna().sum()), 'r2_mean': fm[model].mean()}
            for ref_name, ref in ref_models.items():
                if model == ref or ref not in fm:
                    continue
                pair = fm[[model, ref]].dropna()
                row[f'delta_vs_{ref_name}'] = (pair[model] - pair[ref]).mean()
                row[f'p_vs_{ref_name}'] = ttest_rel(pair[model], pair[ref]).pvalue if len(pair) >= 3 else np.nan
            rows.append(row)
    summary = pd.DataFrame(rows)
    for ref in ['frozen', 'target-only FT']:
        col = f'p_vs_{ref}'
        if col in summary:
            summary[f'p_holm_vs_{ref}'] = holm(summary[col])

    summary['nn_label_rho'] = summary['property'].map({p: nn[p]['spearman_rho'] for p in nn})
    summary['nn_rel_to_within'] = summary['property'].map({p: nn[p]['relative_to_within'] for p in nn})
    summary['zero_shot_rho'] = summary['property'].map(zero_shot)

    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
    fold_means.to_csv(os.path.join(OUT_DIR, 'fold_means.csv'), index=False)
    pd.set_option('display.width', 250)
    print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
