"""Training utilities shared by all model notebooks.

- split_train_val: split a target train fold into inner train/val by scaffold (for early stopping and λ selection).
  With the same fold_idx, the baseline and all DA methods use the identical split.
- fit_early_stopping: early stopping on a val metric; the best weights are kept with deepcopy and restored.
  The test fold is never seen during training and is used only for the final evaluation.
"""
import copy
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')

VAL_FRAC = 0.2
MAX_EPOCHS = 80
PATIENCE = 15


def _scaffold(smi):
    m = Chem.MolFromSmiles(str(smi))
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else str(smi)


def target_smiles(data_dir, prop, fold_idx, split, smiles_to_emb):
    """Return the list of target SMILES in the same order and with the same filter (molecules with embeddings) as the notebook loader."""
    df = pd.read_csv(os.path.join(data_dir, f'fold_{fold_idx}', f'{prop}_target_{split}.csv'))
    return [s for s in df['smiles'].astype(str) if s in smiles_to_emb]


def split_train_val(X, y, fold_idx, smiles, val_frac=VAL_FRAC):
    """Scaffold-based inner train/val split (same difficulty as the outer test fold).

    Molecules sharing a Murcko scaffold go to one side only. Scaffold groups are shuffled with the seed,
    then val is filled group by group until it reaches val_frac.
    """
    assert len(smiles) == len(X), (len(smiles), len(X))
    groups = {}
    for i, s in enumerate(smiles):
        groups.setdefault(_scaffold(s), []).append(i)
    keys = sorted(groups)
    rng = np.random.RandomState(1000 + fold_idx)
    rng.shuffle(keys)

    n_val_target = int(round(len(X) * val_frac))
    val_idx = []
    for k in keys:
        if len(val_idx) >= n_val_target:
            break
        val_idx.extend(groups[k])
    val_idx = np.array(sorted(val_idx))
    tr_idx = np.setdiff1d(np.arange(len(X)), val_idx)
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def fit_early_stopping(model, run_epoch, evaluate_fn, X_val, y_val,
                       max_epochs=MAX_EPOCHS, patience=PATIENCE, metric='r2'):
    """run_epoch(): train for one epoch. evaluate_fn(model, X, y) -> metric dict.

    Returns (best_epoch, best_val_score). On exit the model is restored to the best weights.
    """
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    wait = 0

    for epoch in range(max_epochs):
        run_epoch()
        score = evaluate_fn(model, X_val, y_val)[metric]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return best_epoch, best_score
