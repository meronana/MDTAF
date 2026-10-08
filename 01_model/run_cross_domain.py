"""Cross-domain transfer experiment: measure the transfer gain on several pairs with different domain distances.

Hypothesis — the stronger the source–target label correspondence, the larger the transfer gain.
The main experiment alone gives only 3 points (f_u/CL/t½), confounded with target size.
Here, pairs with different domain distances are built for **the same endpoint (CL)** and compared.

Pairs
- in vitro → human/rat/dog/monkey : source fixed, only the target changes
- animal (3 species) → human      : the closest pair

For each pair, ΔR² of PTFT, LCW and IVP over TargetOnly is measured and viewed alongside the 1-NN label correspondence.
Output: results/cross_domain/{runs.csv, correspondence.csv}
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_data as C
import proposed_methods as P

RDLogger.DisableLog('rdApp.*')
OUT_DIR = os.path.join(C.ROOT, 'results', 'cross_domain')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(3))
METHODS = {'TargetOnly': {}, 'PTFT': {}, 'PTFT-shuffled': {'shuffle': True},
           'LCW': {'lam': 0.3, 'tau': 1.0}, 'IVP': {}}
_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def nn_correspondence(src_smiles, src_y, tgt_smiles, tgt_y, thr=0.4):
    """Spearman correlation between the labels of each target molecule and its structurally nearest source molecule.

    Same approach as the main experiment (3.2). Shared molecules would be more exact,
    but the in vitro source and the animal data share almost no molecules, so 1-NN is used as an approximation.
    """
    S = [_fp.GetFingerprint(Chem.MolFromSmiles(s)) for s in src_smiles]
    rows = []
    for s, y in zip(tgt_smiles, tgt_y):
        sim = np.asarray(DataStructs.BulkTanimotoSimilarity(
            _fp.GetFingerprint(Chem.MolFromSmiles(s)), S))
        j = int(np.argmax(sim))
        rows.append((sim[j], y, src_y[j]))
    d = pd.DataFrame(rows, columns=['sim', 'y_t', 'y_s'])
    q = d[d.sim >= thr]
    rho = spearmanr(q.y_t, q.y_s)[0] if len(q) >= 10 else np.nan
    return rho, len(q), float(d.sim.median())


def main():
    P.set_representation('RDKit')          # the best representation in the QSAR comparison
    pairs, feats, animal = C.build_all()

    print('\n== 1) label correspondence (1-NN, Tanimoto ≥ 0.4)', flush=True)
    corr = []
    for name, folds in pairs.items():
        d = folds[0]
        rho, n, med = nn_correspondence(d['src_smiles'], d['y_src'],
                                        d['fit_smiles'], d['y_fit'])
        corr.append({'pair': name, 'nn_rho': rho, 'n_pairs': n, 'median_sim': med,
                     'n_source': len(d['X_src']), 'n_target_fit': len(d['X_fit'])})
        print(f'  {name:<22} ρ={rho:+.3f} (n={n}, median sim={med:.2f})', flush=True)
    pd.DataFrame(corr).to_csv(os.path.join(OUT_DIR, 'correspondence.csv'), index=False)

    print('\n== 2) transfer experiments', flush=True)
    rows, t0 = [], time.time()
    for name, folds in pairs.items():
        for m, hp in METHODS.items():
            for seed in SEEDS:
                for f in range(C.N_FOLDS):
                    # The in vitro pairs have practically the same source (differs by one after leakage removal).
                    # Share the pretrained model so the same training is not repeated four times.
                    sk = 'cd_invitro' if name.startswith('in vitro') else 'cd_animal'
                    r = P.run_method(m.replace('-shuffled', ''), folds[f],
                                     f'cd_{name}', f, seed, hp, src_key=sk)
                    rows.append({'pair': name, 'method': m, 'seed': seed, 'fold': f, **r})
            sub = pd.DataFrame(rows)
            sub = sub[(sub.pair == name) & (sub.method == m)]
            print(f'  {name:<22} {m:<14} R²={sub.r2.mean():+.4f} ({time.time()-t0:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'runs.csv'), index=False)

    runs = pd.DataFrame(rows)
    print('\n== 3) ΔR² over TargetOnly', flush=True)
    summ = []
    for name in pairs:
        base = runs[(runs.pair == name) & (runs.method == 'TargetOnly')].groupby('fold').r2.mean()
        row = {'pair': name, 'nn_rho': [c['nn_rho'] for c in corr if c['pair'] == name][0]}
        for m in METHODS:
            if m == 'TargetOnly':
                row['TargetOnly R²'] = base.mean()
                continue
            a = runs[(runs.pair == name) & (runs.method == m)].groupby('fold').r2.mean()
            row[f'Δ{m}'] = (a - base).mean()
        summ.append(row)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 200)
    print(s.round(4).to_string(index=False))
    print(f'\nTotal {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
