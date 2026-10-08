"""Runner for exploring and evaluating the proposed methods.

1) Tuning : select (λ, τ) of the LCW family by fold-mean val R² (same protocol as 02-06, seed 0)
2) Evaluation : run every method × seed × fold and save to results/proposed/{runs.csv, best_hp.json}

Usage: python run_proposed.py [--seeds 3] [--tune] [--methods A,B]
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proposed_methods as P

OUT_DIR = os.path.join(P.ROOT, 'results', 'proposed')
os.makedirs(OUT_DIR, exist_ok=True)

LAMS = [0.1, 0.3, 1.0]
TAUS = [0.5, 1.0, 2.0]
SIGMA_YS = [0.25, 0.5, 1.0, 4.0]     # 4.0 is effectively marginal MMD (same as the existing MMD)
CDAN_LAMS = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]   # same grid as the existing CDAN
ALL_METHODS = ['TargetOnly', 'CMMD', 'CDAN-mm', 'LCW', 'LCW-rel', 'LCW-corr',
               'IVP', 'PTFT', 'PTFT+LCW', 'PTFT+IVP', 'IVP-shuffled', 'PTFT-shuffled']

_folds = {}


def get_fold(prop, fold, frac=1.0):
    if (prop, fold, frac) not in _folds:
        _folds[(prop, fold, frac)] = P.load_fold(prop, fold, train_frac=frac)
    return _folds[(prop, fold, frac)]


def base_method(m):
    return m.replace('-shuffled', '')


def grid(method):
    if method in P.USES_CMMD:
        return [{'lam': l, 'sigma_y': s} for l, s in itertools.product(LAMS, SIGMA_YS)]
    if method in P.USES_CDAN_MM:
        return [{'lam': l} for l in CDAN_LAMS]
    if method not in P.USES_WEIGHTED_SOURCE:
        return [{}]
    if method == 'LCW-rel':                       # uses relevance only, so τ is meaningless
        return [{'lam': l, 'tau': 1.0} for l in LAMS]
    return [{'lam': l, 'tau': t} for l, t in itertools.product(LAMS, TAUS)]


def tune(methods, props):
    best = {}
    for prop in props:
        best[prop] = {}
        for m in methods:
            cand = grid(base_method(m))
            if cand == [{}]:
                best[prop][m] = {}
                continue
            scores = []
            for hp in cand:
                v = [P.run_method(base_method(m), get_fold(prop, f), prop, f, 0,
                                  {**hp, 'shuffle': m.endswith('-shuffled')})['val_r2']
                     for f in range(P.N_FOLDS)]
                scores.append((float(np.mean(v)), hp))
            scores.sort(key=lambda x: -x[0])
            best[prop][m] = scores[0][1]
            print(f'  [tune] {prop:<10} {m:<12} → {scores[0][1]}  val R²={scores[0][0]:.4f}', flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--methods', default=','.join(ALL_METHODS))
    ap.add_argument('--props', default=','.join(P.PROPERTIES))
    ap.add_argument('--tag', default='')
    ap.add_argument('--train-fracs', default='1.0',
                    help='Comma-separated fractions of target training data, to separate the size confound')
    ap.add_argument('--rep', default='Graphormer', choices=['Graphormer', 'RDKit'],
                    help='Molecular representation. With RDKit the same experiment is repeated on descriptors')
    ap.add_argument('--reuse-hp', default='', help='Comma-separated list of tags. Merge their best_hp and skip tuning')
    args = ap.parse_args()
    d_in = P.set_representation(args.rep)
    print(f'Representation: {args.rep} ({d_in}-d)', flush=True)
    methods = args.methods.split(',')
    props = args.props.split(',')
    seeds = list(range(args.seeds))

    t0 = time.time()
    hp_path = os.path.join(OUT_DIR, f'best_hp{args.tag}.json')
    if args.reuse_hp:
        best = {p: {} for p in props}
        for tag in args.reuse_hp.split(','):
            prev = json.load(open(os.path.join(OUT_DIR, f'best_hp{tag}.json')))
            for p in props:
                best[p].update(prev.get(p, {}))
        missing = [(p, m) for p in props for m in methods if m not in best[p]]
        assert not missing, f'Combinations without hyperparameters to reuse: {missing}'
        print(f'== 1) Reusing hyperparameters ({args.reuse_hp}) — no further search', flush=True)
    else:
        print('== 1) Hyperparameter tuning (val R², seed 0)', flush=True)
        best = tune(methods, props)
    json.dump(best, open(hp_path, 'w'), indent=2)

    print(f'\n== 2) Evaluation ({len(methods)} methods × {len(seeds)} seeds × {P.N_FOLDS} folds × {len(props)} props)', flush=True)
    rows = []
    runs_path = os.path.join(OUT_DIR, f'runs{args.tag}.csv')
    fracs = [float(x) for x in args.train_fracs.split(',')]
    for m in methods:
        for prop in props:
            hp = dict(best[prop][m], shuffle=m.endswith('-shuffled'))
            for frac in fracs:
                for seed in seeds:
                    for fold in range(P.N_FOLDS):
                        r = P.run_method(base_method(m), get_fold(prop, fold, frac), prop, fold, seed, hp)
                        rows.append({'method': m, 'property': prop, 'train_frac': frac, 'seed': seed,
                                     'fold': fold, 'lam': hp.get('lam'), 'tau': hp.get('tau'), **r})
                d = pd.DataFrame(rows)
                d = d[(d.method == m) & (d.property == prop) & (d.train_frac == frac)]
                print(f'  {m:<14} {prop:<10} frac={frac:<5} r2={d.r2.mean():+.4f} '
                      f'n_fit={len(get_fold(prop, 0, frac)["X_fit"])} ({time.time()-t0:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(runs_path, index=False)

    runs = pd.DataFrame(rows)
    runs = runs[runs.train_frac == max(fracs)] if 'train_frac' in runs else runs
    ref = runs[runs.method == 'TargetOnly'].groupby(['property', 'fold'])['r2'].mean()
    print('\n== 3) ΔR² over TargetOnly (fold mean)', flush=True)
    summ = []
    for m in methods:
        for prop in props:
            fm = runs[(runs.method == m) & (runs.property == prop)].groupby('fold')['r2'].mean()
            summ.append({'method': m, 'property': prop, 'r2': fm.mean(),
                         'delta_r2': (fm - ref.loc[prop]).mean() if m != 'TargetOnly' else np.nan})
    s = pd.DataFrame(summ).pivot(index='method', columns='property', values=['r2', 'delta_r2'])
    s.to_csv(os.path.join(OUT_DIR, f'summary{args.tag}.csv'))
    print(s.round(4).to_string(), flush=True)
    print(f'\nTotal {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
