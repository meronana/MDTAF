"""Proposed methods: use the source by 'whether its labels actually transfer', not by 'where it lies'.

Diagnosis (results of 3.2 and 3.4):
- MMD/CORAL/DANN/CDAN only align P(x). But the P(x) gap is almost identical for the three endpoints
  (MMD 0.30-0.33, domain AUC 0.93-0.96), so this direction cannot create differences between endpoints.
- Only IW uses source labels, but its weights are the density ratio p_t(x)/p_s(x), i.e. defined in input space.
  The axis that actually separates the endpoints is label correspondence (1-NN ρ: f_u 0.70 / CL 0.19 / t½ 0.05).

Proposed methods
- LCW  : source weights = relevance × correspondence instead of the density ratio (direct replacement for IW)
- IVP  : inject the source model's in vitro prediction into the target head (learned IVIVE)
- PTFT : pretrain the encoder on source labels, then fine-tune on the target (frozen-embedding version of notebook 10)
- PTFT+LCW, PTFT+IVP : combinations

Shared conventions (same as 01-08)
- same scaffold 5-fold, same inner val split, val R² early stopping, test used only for final evaluation
- AdamW(lr 1e-3, wd 1e-4), Huber(δ=1), batch 32, grad clip 5, max 80 epochs, patience 15
- seed formula seed*100 + fold
- correspondence weights use target **train** labels only (no test leakage)
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from train_utils import fit_early_stopping, split_train_val, target_smiles

RDLogger.DisableLog('rdApp.*')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'cv_datasets')
FEAT_DIR = os.path.join(ROOT, 'features')
DEVICE = torch.device('cpu')          # MLP on frozen embeddings → CPU is enough (09/10 use the GPU)
PROPERTIES = ['fu', 'clearance', 'half_life']
N_FOLDS = 5
BATCH = 32

_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_emb = np.load(os.path.join(FEAT_DIR, 'graphormer_cv_embeddings.npz'), allow_pickle=True)
_emb_X, _emb_keys = _emb['X'], _emb['keys']        # npz is lazy → extract once before the loop
SMILES_TO_EMB = {s: _emb_X[i] for i, s in enumerate(_emb_keys)}

REPRESENTATION = 'Graphormer'
D_IN = 768
STANDARDIZE = False        # only enabled for representations with large scale differences such as RDKit descriptors


def set_representation(kind):
    """Switch the molecular representation: 'Graphormer' (768d) or 'RDKit' (descriptors).

    RDKit descriptors did better in the QSAR baseline (15), so to check whether the DA conclusions
    depend on the representation, the same code can swap only the representation.
    RDKit has large scale differences, so it is standardised with per-fold train statistics (handled in load_fold).
    """
    global SMILES_TO_EMB, D_IN, REPRESENTATION, STANDARDIZE
    if kind == 'Graphormer':
        SMILES_TO_EMB = {s: _emb_X[i] for i, s in enumerate(_emb_keys)}
        STANDARDIZE = False
    elif kind == 'RDKit':
        d = np.load(os.path.join(FEAT_DIR, 'rdkit_descriptors.npz'), allow_pickle=True)
        X, keys = d['X'], d['keys']
        SMILES_TO_EMB = {s: X[i] for i, s in enumerate(keys)}
        STANDARDIZE = True
    else:
        raise ValueError(kind)
    REPRESENTATION = kind
    D_IN = len(next(iter(SMILES_TO_EMB.values())))
    _fp_cache.clear()
    _src_cache.clear()
    _corr_cache.clear()
    _folds_cache.clear()
    return D_IN


# ---------------------------------------------------------------- data

_rd_keys = set(np.load(os.path.join(FEAT_DIR, 'rdkit_descriptors.npz'), allow_pickle=True)['keys'].tolist())     if os.path.exists(os.path.join(FEAT_DIR, 'rdkit_descriptors.npz')) else None
# Restrict to the intersection so every representation uses the same molecule set (fair comparison)
COMMON = set(SMILES_TO_EMB) & _rd_keys if _rd_keys else set(SMILES_TO_EMB)
_folds_cache = {}


def _load(path):
    """CV csv → (smiles, X, y). Only molecules present in both representations, file order preserved."""
    df = pd.read_csv(path, low_memory=False)
    keep = [(s, v) for s, v in zip(df['smiles'].astype(str), df['value']) if s in COMMON]
    smi = [s for s, _ in keep]
    X = np.stack([SMILES_TO_EMB[s] for s in smi]).astype('float32')
    return smi, X, np.array([v for _, v in keep], dtype='float32')


def load_fold(prop, fold, train_frac=1.0, sub_seed=0):
    """Source / target (train → inner train and val) / target test for one fold.

    If train_frac < 1, the target training data (inner train and val) is randomly subsampled to that fraction.
    **The test set is never touched** — a fixed evaluation set keeps the size comparison fair and reduces evaluation noise.
    Across endpoints, target size (f_u 353 / CL 1027 / t½ 1164) and label correspondence point in the same direction,
    so this separates whether a gain comes from 'correspondence' or from 'lack of data'.
    """
    s_smi, X_s, y_s = _load(os.path.join(DATA_DIR, f'fold_{fold}', f'{prop}_source.csv'))
    t_smi, X_t, y_t = _load(os.path.join(DATA_DIR, f'fold_{fold}', f'{prop}_target_train.csv'))
    _, X_te, y_te = _load(os.path.join(DATA_DIR, f'fold_{fold}', f'{prop}_target_test.csv'))

    # Pass the indices through to use the same inner split as the existing notebooks
    idx = np.arange(len(t_smi), dtype='float32')[:, None]
    tr_i, _, va_i, _ = split_train_val(idx, y_t, fold, smiles=target_smiles(DATA_DIR, prop, fold, 'train', SMILES_TO_EMB))
    tr_i, va_i = tr_i[:, 0].astype(int), va_i[:, 0].astype(int)

    if train_frac < 1.0:                     # subsample training data only (test unchanged)
        rng = np.random.RandomState(7000 + fold * 17 + sub_seed)
        tr_i = np.sort(rng.choice(tr_i, max(int(round(len(tr_i) * train_frac)), 20), replace=False))
        va_i = np.sort(rng.choice(va_i, max(int(round(len(va_i) * train_frac)), 10), replace=False))

    if STANDARDIZE:          # standardise only with statistics of the fold's training data (source + target fit)
        ref = np.vstack([X_s, X_t[tr_i]]).astype('float64')
        mu, sd = ref.mean(0), ref.std(0)
        sd[sd < 1e-8] = 1.0
        # Some RDKit descriptors such as Ipc reach 10^20, so extreme values remain even after standardisation.
        # Clip to ±10 to prevent exploding gradients and float32 overflow.
        X_s, X_t, X_te = (np.clip((a.astype('float64') - mu) / sd, -10, 10).astype('float32')
                          for a in (X_s, X_t, X_te))

    return {
        'src_smiles': s_smi, 'X_src': X_s, 'y_src': y_s,
        'fit_smiles': [t_smi[i] for i in tr_i], 'X_fit': X_t[tr_i], 'y_fit': y_t[tr_i],
        'X_val': X_t[va_i], 'y_val': y_t[va_i],
        'X_test': X_te, 'y_test': y_te,
    }


# ------------------------------------------------- correspondence weights

_fp_cache = {}


def _fps(smiles, key):
    if key not in _fp_cache:
        _fp_cache[key] = [_fpgen.GetFingerprint(Chem.MolFromSmiles(str(s))) for s in smiles]
    return _fp_cache[key]


_corr_cache = {}


def correspondence_weights(d, prop, fold, k=5, tau=1.0):
    """Weight of source sample i = relevance_i × correspondence_i.

    relevance_i     : mean Tanimoto to the k nearest target-train neighbours (the role IW's density ratio used to play)
    correspondence_i: Gaussian kernel of |y_src_i - (similarity-weighted mean of neighbouring target labels)|
                      Source and target are both z-scored, so their scales are comparable.

    IW only looks at relevance. Here we multiply by 'does that label actually predict the neighbouring target labels'.
    With a large tau the correspondence term flattens and the weighting converges to relevance only (similar to IW).
    Returns: (relevance, residual) — the final weight is built at the call site together with τ and λ.
    """
    ck = (prop, fold, k)
    if ck in _corr_cache:
        return _corr_cache[ck]
    S = _fps(d['src_smiles'], ('src', prop, fold))
    T = _fps(d['fit_smiles'], ('fit', prop, fold))
    y_t = d['y_fit']
    rel, resid = np.empty(len(S), 'float32'), np.empty(len(S), 'float32')
    for i, fp in enumerate(S):
        sim = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, T), dtype='float64')
        top = np.argpartition(-sim, min(k, len(sim) - 1))[:k]
        w = sim[top]
        rel[i] = w.mean()
        resid[i] = abs(d['y_src'][i] - (np.average(y_t[top], weights=w) if w.sum() > 0 else y_t.mean()))
    _corr_cache[ck] = (rel, resid)
    return rel, resid


def make_lcw_weights(rel, resid, tau, mode='both', clip=(0.0, 10.0)):
    """mode: both = relevance × correspondence (proposed), rel = relevance only (input-space criterion like IW),
    corr = correspondence only. Comparing the three decomposes which term contributes."""
    corr = np.exp(-(resid ** 2) / (2 * tau ** 2))
    w = {'both': rel * corr, 'rel': rel, 'corr': corr}[mode]
    w = np.clip(w, *clip)
    return (w / max(w.mean(), 1e-8)).astype('float32')


# ---------------------------------------------------------------- models

class Encoder(nn.Module):
    """Shared encoder identical to the existing DA models (768 → 256 → 128)."""

    def __init__(self, d_in=None, d_hidden=256, dropout=0.2):
        super().__init__()
        d_in = D_IN if d_in is None else d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.ReLU(), nn.Dropout(dropout))
        self.d_out = d_hidden // 2

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """Same as the existing TaskHead (128 → 128 → 1). If extra>0 it also receives e.g. in vitro predictions."""

    def __init__(self, d_in, d_hidden=128, extra=0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in + extra, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))

    def forward(self, h, extra=None):
        if extra is not None:
            h = torch.cat([h, extra], dim=1)
        return self.net(h).squeeze(-1)


class GRL(torch.autograd.Function):
    """Gradient reversal (same as DANN/CDAN)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.alpha, None


class DomainClassifier(nn.Module):
    def __init__(self, d_in, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(d_hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Model(nn.Module):
    """Target head + (optional) source head / domain classifier / in vitro prediction injection."""

    def __init__(self, d_in=None, source_head=False, extra=0, domain_clf=0):
        super().__init__()
        self.encoder = Encoder(d_in)
        self.head = Head(self.encoder.d_out, extra=extra)
        self.src_head = Head(self.encoder.d_out) if source_head else None
        self.domain_clf = DomainClassifier(domain_clf) if domain_clf else None

    def forward(self, x, extra=None):
        return self.head(self.encoder(x), extra)

    def forward_source(self, x):
        return self.src_head(self.encoder(x))


@torch.no_grad()
def evaluate(model, X, y, extra=None):
    model.eval()
    x = torch.tensor(X, device=DEVICE)
    e = None if extra is None else torch.tensor(extra, device=DEVICE).unsqueeze(1)
    p = model(x, e).cpu().numpy()
    p = np.nan_to_num(p, nan=0.0, posinf=1e6, neginf=-1e6)
    return {'r2': float(r2_score(y, p)), 'rho': float(spearmanr(y, p)[0]),
            'rmse': float(np.sqrt(np.mean((y - p) ** 2))), 'mae': float(np.mean(np.abs(y - p)))}, p


def _opt(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


def _batches(n):
    perm = np.random.permutation(n)
    for i in range(0, n, BATCH):
        yield perm[i:i + BATCH]


# ---------------------------------------------------------------- training loop

def _epoch_target_only(model, X, y, opt, crit, extra=None):
    model.train()
    for idx in _batches(len(X)):
        e = None if extra is None else torch.tensor(extra[idx], device=DEVICE).unsqueeze(1)
        opt.zero_grad()
        loss = crit(model(torch.tensor(X[idx], device=DEVICE), e), torch.tensor(y[idx], device=DEVICE))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()


def _epoch_weighted_source(model, d, w, opt, crit_t, crit_s, lam):
    """L_target + λ·mean(w · L_source). Epochs count target batches; source is randomly sampled each batch."""
    model.train()
    X_t, y_t, X_s, y_s = d['X_fit'], d['y_fit'], d['X_src'], d['y_src']
    for idx in _batches(len(X_t)):
        js = np.random.randint(0, len(X_s), len(idx))
        opt.zero_grad()
        loss = crit_t(model(torch.tensor(X_t[idx], device=DEVICE)), torch.tensor(y_t[idx], device=DEVICE))
        src = crit_s(model.forward_source(torch.tensor(X_s[js], device=DEVICE)),
                     torch.tensor(y_s[js], device=DEVICE))
        loss = loss + lam * (src * torch.tensor(w[js], device=DEVICE)).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()


_src_cache = {}


def fit_source_model(d, prop, fold, seed, shuffle_labels=False):
    """Source model (stage 1 of IVP and PTFT). Trained once per (prop, fold, seed) and cached.

    If shuffle_labels=True, source labels are shuffled → a control with the same molecules but no label information.
    """
    key = (prop, fold, seed, shuffle_labels)
    if key not in _src_cache:
        dd = d
        if shuffle_labels:
            dd = dict(d, y_src=np.random.RandomState(seed).permutation(d['y_src']))
        _src_cache[key] = _fit_source_model(dd, seed)
    return _src_cache[key]


# ------------------------------------- conditional alignment (proposals A and B)

def _gauss(a, b, sigma):
    return torch.exp(-torch.cdist(a, b) ** 2 / (2 * sigma ** 2))


def conditional_mmd(f_s, f_t, y_s, y_t, sigma_f=None, sigma_y=1.0):
    """CMMD: MMD measured with a joint kernel over features and labels.

        k((f,y),(f',y')) = k_f(f,f') · k_y(y,y')

    As σ_y → ∞, k_y ≡ 1 and it **becomes exactly the existing marginal MMD**.
    With a small σ_y, only source–target pairs with similar labels contribute to the alignment
    (= conditional alignment). In other words, the existing MMD is a special case of this method.

    σ_f defaults to the in-batch median heuristic. The fixed σ=1 of the existing implementation
    drops kernel values to ~0.04 in the 128-d feature space, which weakens the alignment signal.
    """
    if sigma_f is None:
        with torch.no_grad():
            z = torch.cat([f_s, f_t])
            d2 = torch.cdist(z, z) ** 2
            med = d2[~torch.eye(len(z), dtype=torch.bool, device=z.device)].median()
            sigma_f = torch.sqrt(torch.clamp(med / 2, min=1e-6))
    ys, yt = y_s.unsqueeze(1), y_t.unsqueeze(1)
    kss = _gauss(f_s, f_s, sigma_f) * _gauss(ys, ys, sigma_y)
    ktt = _gauss(f_t, f_t, sigma_f) * _gauss(yt, yt, sigma_y)
    kst = _gauss(f_s, f_t, sigma_f) * _gauss(ys, yt, sigma_y)
    mmd = kss.mean() + ktt.mean() - 2 * kst.mean()
    return torch.sqrt(torch.clamp(mmd, min=1e-8))


def _epoch_cmmd(model, d, opt, crit, lam, sigma_y):
    """L_target + λ · CMMD(source, target). Source labels are used only in the alignment kernel."""
    model.train()
    X_t, y_t, X_s, y_s = d['X_fit'], d['y_fit'], d['X_src'], d['y_src']
    for idx in _batches(len(X_t)):
        js = np.random.randint(0, len(X_s), len(idx))
        xt = torch.tensor(X_t[idx], device=DEVICE)
        xs = torch.tensor(X_s[js], device=DEVICE)
        yt = torch.tensor(y_t[idx], device=DEVICE)
        ys = torch.tensor(y_s[js], device=DEVICE)
        opt.zero_grad()
        ft, fs = model.encoder(xt), model.encoder(xs)
        loss = crit(model.head(ft), yt) + lam * conditional_mmd(fs, ft, ys, yt, sigma_y=sigma_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()


def _epoch_cdan_mm(model, d, opt, crit, lam):
    """CDAN ported to regression with the original paper's multilinear conditioning.

    domain classifier input = f ⊗ [ŷ, 1] = concat(f·ŷ, f)  (dim 2·d_out).
    The existing implementation used sigmoid(|ŷ|) as a loss weight (confidence weighting), which is not conditioning.
    """
    model.train()
    X_t, y_t, X_s, y_s = d['X_fit'], d['y_fit'], d['X_src'], d['y_src']
    bce = nn.BCEWithLogitsLoss()
    for idx in _batches(len(X_t)):
        js = np.random.randint(0, len(X_s), len(idx))
        xt = torch.tensor(X_t[idx], device=DEVICE)
        xs = torch.tensor(X_s[js], device=DEVICE)
        opt.zero_grad()
        ft, fs = model.encoder(xt), model.encoder(xs)
        pt, ps = model.head(ft), model.head(fs)
        task = crit(pt, torch.tensor(y_t[idx], device=DEVICE))
        mt = torch.cat([ft * pt.detach().unsqueeze(1), ft], dim=1)
        ms = torch.cat([fs * ps.detach().unsqueeze(1), fs], dim=1)
        dom = torch.cat([GRL.apply(ms, lam), GRL.apply(mt, lam)])
        lbl = torch.cat([torch.zeros(len(ms), device=DEVICE), torch.ones(len(mt), device=DEVICE)])
        loss = task + lam * bce(model.domain_clf(dom), lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()


def _fit_source_model(d, seed, max_epochs=40, patience=8):
    """Model trained on source labels only. Early stopping on a 90/10 split within the source."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(d['X_src'])
    rng = np.random.RandomState(seed)
    va = rng.choice(n, max(int(0.1 * n), 1), replace=False)
    tr = np.setdiff1d(np.arange(n), va)
    m = Model().to(DEVICE)
    opt, crit = _opt(m), nn.HuberLoss(delta=1.0)
    fit_early_stopping(m, lambda: _epoch_target_only(m, d['X_src'][tr], d['y_src'][tr], opt, crit),
                       lambda mm, X, y: evaluate(mm, X, y)[0], d['X_src'][va], d['y_src'][va],
                       max_epochs=max_epochs, patience=patience)
    return m


@torch.no_grad()
def _predict(model, X):
    model.eval()
    return model(torch.tensor(X, device=DEVICE)).cpu().numpy().astype('float32')


# ---------------------------------------------------------------- methods

USES_SOURCE_MODEL = ('IVP', 'PTFT', 'PTFT+IVP', 'PTFT+LCW')
USES_WEIGHTED_SOURCE = ('LCW', 'LCW-rel', 'LCW-corr', 'PTFT+LCW')
USES_CMMD = ('CMMD', 'CMMD+LCW')
USES_CDAN_MM = ('CDAN-mm',)


def run_method(method, d, prop, fold, seed, hp, src_key=None):
    """hp: LCW family {'lam','tau'}, otherwise {}. shuffle=True gives the shuffled-source-label control.

    If src_key is given, the source-model cache is shared under that key. When comparing several pairs
    with the same source but different targets, this avoids repeating the same pretraining.
    Correspondence weights must differ for every (source, target) pair, so prop is used as is.
    """
    src_key = prop if src_key is None else src_key
    shuffle = bool(hp.get('shuffle', False))
    torch.manual_seed(seed * 100 + fold)
    np.random.seed(seed * 100 + fold)
    extra_tr = extra_va = extra_te = None

    if method in ('IVP', 'PTFT+IVP'):
        src_model = fit_source_model(d, src_key, fold, seed * 100 + fold, shuffle)
        extra_tr, extra_va, extra_te = (_predict(src_model, d[k]) for k in ('X_fit', 'X_val', 'X_test'))

    model = Model(source_head=method in USES_WEIGHTED_SOURCE,
                  extra=1 if method in ('IVP', 'PTFT+IVP') else 0,
                  domain_clf=2 * 128 if method in USES_CDAN_MM else 0).to(DEVICE)

    if method.startswith('PTFT'):
        # Stage 1: pretrain the encoder on source labels → keep only the encoder, re-initialise the head
        pre = fit_source_model(d, src_key, fold, seed * 100 + fold, shuffle)
        model.encoder.load_state_dict(pre.encoder.state_dict())
        torch.manual_seed(seed * 100 + fold)   # fix the head-initialisation seed

    opt = _opt(model)
    crit = nn.HuberLoss(delta=1.0)
    if method in USES_WEIGHTED_SOURCE:
        rel, resid = correspondence_weights(d, prop, fold)
        mode = {'LCW-rel': 'rel', 'LCW-corr': 'corr'}.get(method, 'both')
        w = make_lcw_weights(rel, resid, hp['tau'], mode)
        crit_s = nn.HuberLoss(delta=1.0, reduction='none')
        step = lambda: _epoch_weighted_source(model, d, w, opt, crit, crit_s, hp['lam'])
    elif method in USES_CMMD:
        step = lambda: _epoch_cmmd(model, d, opt, crit, hp['lam'], hp['sigma_y'])
    elif method in USES_CDAN_MM:
        step = lambda: _epoch_cdan_mm(model, d, opt, crit, hp['lam'])
    else:
        step = lambda: _epoch_target_only(model, d['X_fit'], d['y_fit'], opt, crit, extra_tr)

    best_epoch, best_val = fit_early_stopping(
        model, step, lambda m, X, y: evaluate(m, X, y, extra_va)[0], d['X_val'], d['y_val'])
    m, _ = evaluate(model, d['X_test'], d['y_test'], extra_te)
    return {**m, 'val_r2': best_val, 'best_epoch': best_epoch}
