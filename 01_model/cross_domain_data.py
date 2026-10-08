"""Data construction for the cross-domain transfer experiments.

The main study covers a single domain gap: in vitro → human in vivo.
Adding animal PK data (PKSmart, Lombardo 2013) gives **several pairs with different gap sizes**,
so the hypothesis "source–target label correspondence determines transfer success" can be tested at several points.

Pairs built
- in vitro CLint → human CL   (existing, correspondence 0.19)
- in vitro CLint → rat/dog/monkey CL   (source fixed, only the target changes)
- rat+dog+monkey CL → human CL         (correspondence 0.54-0.83, the closest pair)

Same conventions as the main experiment: log10 transform, source z-scored over the whole set, target z-scored with fold-train statistics,
Murcko scaffold 5-fold, and each fold's test molecules are removed from the source to prevent leakage.
"""
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANIMAL = os.path.join(ROOT, 'data', 'animal_pk', 'Animal_PK_data.csv')
NORM = os.path.join(ROOT, 'data', 'normalized_data')
FEAT = os.path.join(ROOT, 'features', 'rdkit_descriptors_all.npz')
N_FOLDS = 5
_DESC = list(Descriptors.descList)


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


def _desc(smiles, names=None):
    # If names is given, compute only those descriptors in that order (to match the columns of the existing file)
    fns = [dict(_DESC)[n] for n in names] if names is not None else [fn for _, fn in _DESC]
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        v = []
        for fn in fns:
            try:
                v.append(fn(m))
            except Exception:
                v.append(np.nan)
        out.append(v)
    X = np.asarray(out, dtype='float64')
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')


def build_features(extra_smiles):
    """Add animal molecules to the existing CV-molecule descriptors to build a single representation table."""
    if os.path.exists(FEAT):
        d = np.load(FEAT, allow_pickle=True)
        X, keys = d['X'], d['keys']          # npz is lazy → extract once before the loop
        return {s: X[i] for i, s in enumerate(keys)}

    base = np.load(os.path.join(ROOT, 'features', 'rdkit_descriptors.npz'), allow_pickle=True)
    bk, bX, names = base['keys'].tolist(), base['X'], base['names'].tolist()
    new = sorted(set(extra_smiles) - set(bk))
    print(f'  computing extra descriptors: {len(new)} molecules × {len(names)} descriptors')
    keys = bk + new
    X = np.vstack([bX, _desc(new, names)]) if new else bX
    np.savez_compressed(FEAT, keys=np.array(keys), X=X, names=np.array(names))
    return {s: X[i] for i, s in enumerate(keys)}


def load_animal():
    a = pd.read_csv(ANIMAL)
    a['smiles'] = a['smiles_r'].map(canon)
    return a[a['smiles'].notna()].copy()


def _scaffold_folds(smiles, seed=0):
    """Assign Murcko scaffold groups to 5 folds in order of size (same scheme as the main experiment)."""
    sc = pd.Series([MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smiles])
    groups = sc.groupby(sc).indices
    keys = sorted(groups)
    np.random.RandomState(seed).shuffle(keys)
    fold = np.empty(len(smiles), dtype=int)
    sizes = np.zeros(N_FOLDS, dtype=int)
    for k in sorted(keys, key=lambda k: -len(groups[k])):
        f = int(np.argmin(sizes))
        fold[groups[k]] = f
        sizes[f] += len(groups[k])
    return fold


def make_pair(source, target, feats, name=''):
    """source/target DataFrame(smiles, value; original units) → per-fold training data bundle.

    value must be positive and is log10-transformed. The target is z-scored with fold-train statistics,
    the source with whole-set statistics. Each fold's test molecules are removed from the source.
    """
    src = source[source.value > 0].groupby('smiles', as_index=False).value.mean()
    tgt = target[target.value > 0].groupby('smiles', as_index=False).value.mean()
    src, tgt = (d[d.smiles.isin(feats)].reset_index(drop=True) for d in (src, tgt))
    src['log'], tgt['log'] = np.log10(src.value), np.log10(tgt.value)
    tgt['fold'] = _scaffold_folds(tgt.smiles.tolist())

    s_mu, s_sd = src['log'].mean(), src['log'].std()
    emb = lambda ss: np.stack([feats[s] for s in ss]).astype('float32')
    folds = {}
    for f in range(N_FOLDS):
        te, tr = tgt[tgt.fold == f], tgt[tgt.fold != f]
        mu, sd = tr['log'].mean(), tr['log'].std()
        rng = np.random.RandomState(1000 + f)
        va_n = max(int(round(len(tr) * 0.2)), 5)
        va_i = rng.choice(len(tr), va_n, replace=False)
        fit_i = np.setdiff1d(np.arange(len(tr)), va_i)
        s = src[~src.smiles.isin(set(te.smiles))]           # prevent leakage
        Xs, Xf = emb(s.smiles), emb(tr.smiles.iloc[fit_i])
        Xv, Xt = emb(tr.smiles.iloc[va_i]), emb(te.smiles)

        # Some RDKit descriptors such as Ipc reach 10^30, so using mean and std
        # overflows the sum into NaN. Quantile (1-99%) scaling is robust to extreme values.
        # Statistics are computed only on the fold's training data (source + target fit) to prevent test leakage.
        clean = lambda A: np.nan_to_num(A.astype('float64'), nan=0.0, posinf=0.0, neginf=0.0)
        ref = clean(np.vstack([Xs, Xf]))
        lo, hi = np.percentile(ref, [1, 99], axis=0)
        span = np.maximum(hi - lo, 1e-8)
        z = lambda A: np.clip((clean(A) - lo) / span, -5, 5).astype('float32')
        Xs, Xf, Xv, Xt = z(Xs), z(Xf), z(Xv), z(Xt)

        folds[f] = {
            'src_smiles': s.smiles.tolist(), 'X_src': Xs,
            'y_src': ((s['log'] - s_mu) / s_sd).to_numpy('float32'),
            'fit_smiles': tr.smiles.iloc[fit_i].tolist(), 'X_fit': Xf,
            'y_fit': ((tr['log'].iloc[fit_i] - mu) / sd).to_numpy('float32'),
            'X_val': Xv, 'y_val': ((tr['log'].iloc[va_i] - mu) / sd).to_numpy('float32'),
            'X_test': Xt, 'y_test': ((te['log'] - mu) / sd).to_numpy('float32'),
            'n_src_removed': len(src) - len(s),
        }
    if name:
        d0 = folds[0]
        print(f'  {name:<28} source {len(d0["X_src"])} (leakage removed {d0["n_src_removed"]}) | '
              f'target {len(tgt)} → fit {len(d0["X_fit"])} val {len(d0["X_val"])} test {len(d0["X_test"])}')
    return folds


def build_all():
    """Build and return every (source, target) pair compared in the paper."""
    a = load_animal()
    inv = pd.read_csv(os.path.join(NORM, 'clearance_source.csv'), low_memory=False)
    inv['smiles'] = inv['smiles'].astype(str)
    hum = pd.read_csv(os.path.join(NORM, 'clearance_target.csv'))
    hum['smiles'] = hum['smiles'].astype(str)

    feats = build_features(set(a.smiles))

    # value in normalized_data is already log10(+z) transformed, so undo it with 10** to get original units
    inv_raw = inv.assign(value=10 ** inv['value'])
    hum_raw = hum.assign(value=10 ** hum['value'])

    sp_cl = {sp: a[a[f'{sp}_CL_mL_min_kg'].notna()][['smiles', f'{sp}_CL_mL_min_kg']]
             .rename(columns={f'{sp}_CL_mL_min_kg': 'value'}) for sp in ('rat', 'dog', 'monkey')}
    animal_all = pd.concat(sp_cl.values(), ignore_index=True)

    pairs = {}
    print('Transfer pairs built:')
    pairs['in vitro → human'] = make_pair(inv_raw, hum_raw, feats, 'in vitro → human CL')
    for sp in ('rat', 'dog', 'monkey'):
        pairs[f'in vitro → {sp}'] = make_pair(inv_raw, sp_cl[sp], feats, f'in vitro → {sp} CL')
    pairs['animal → human'] = make_pair(animal_all, hum_raw, feats, 'animal (3 species) → human CL')
    return pairs, feats, a
