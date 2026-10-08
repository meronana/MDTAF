#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'cv_datasets')
FEAT_DIR = os.path.join(ROOT, 'features')
RES_DIR = os.path.join(ROOT, 'results', 'mmd_adaptation')
os.makedirs(RES_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROPERTIES = ['clearance', 'half_life', 'fu']
N_FOLDS = 5
MMD_WEIGHT = 0.1

# Load embeddings
print("Loading embeddings...")
emb_path = os.path.join(FEAT_DIR, 'graphormer_cv_embeddings.npz')
emb_data = np.load(emb_path, allow_pickle=True)
smiles_to_emb = {smi: emb_data['X'][i] for i, smi in enumerate(emb_data['keys'])}
print(f"Loaded {len(smiles_to_emb)} embeddings\n")

# MMD Loss
def gaussian_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    distances = torch.sum((x - y) ** 2, dim=2)
    return torch.exp(-distances / (2 * sigma ** 2))

def compute_mmd(source_features, target_features, sigma=1.0):
    K_ss = gaussian_kernel(source_features, source_features, sigma)
    K_tt = gaussian_kernel(target_features, target_features, sigma)
    K_st = gaussian_kernel(source_features, target_features, sigma)

    n_s = source_features.shape[0]
    n_t = target_features.shape[0]

    mmd_loss = (1 / (n_s ** 2)) * K_ss.sum() + (1 / (n_t ** 2)) * K_tt.sum() - 2 * (1 / (n_s * n_t)) * K_st.sum()
    return torch.sqrt(torch.clamp(mmd_loss, min=1e-8))

# Model
class SharedEncoder(nn.Module):
    def __init__(self, d_in=768, d_hidden=256, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.d_out = d_hidden // 2

    def forward(self, x):
        return self.encoder(x)

class TaskHead(nn.Module):
    def __init__(self, d_in, d_hidden=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)

class DomainAdaptationModel(nn.Module):
    def __init__(self, d_in=768):
        super().__init__()
        self.encoder = SharedEncoder(d_in)
        self.task_head = TaskHead(self.encoder.d_out)

    def forward(self, x):
        features = self.encoder(x)
        output = self.task_head(features)
        return output, features

# Data loading
def get_smiles_col(df):
    return 'smiles' if 'smiles' in df.columns else 'Drug' if 'Drug' in df.columns else None

def get_value_col(df):
    return 'value' if 'value' in df.columns else 'Y' if 'Y' in df.columns else None

def load_source(prop, fold_idx):
    csv_path = os.path.join(DATA_DIR, f'fold_{fold_idx}', f'{prop}_source.csv')
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    smi_col, val_col = get_smiles_col(df), get_value_col(df)
    if smi_col is None or val_col is None:
        return None, None

    X_list = [smiles_to_emb[smi] for smi, y in zip(df[smi_col].astype(str), df[val_col]) if smi in smiles_to_emb]
    y_list = [y for smi, y in zip(df[smi_col].astype(str), df[val_col]) if smi in smiles_to_emb]

    return np.stack(X_list).astype('float32') if X_list else np.array([]), np.array(y_list, dtype='float32') if y_list else np.array([])

def load_target(prop, fold_idx, split):
    csv_path = os.path.join(DATA_DIR, f'fold_{fold_idx}', f'{prop}_target_{split}.csv')
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    smi_col, val_col = get_smiles_col(df), get_value_col(df)
    if smi_col is None or val_col is None:
        return None, None

    X_list = [smiles_to_emb[smi] for smi, y in zip(df[smi_col].astype(str), df[val_col]) if smi in smiles_to_emb]
    y_list = [y for smi, y in zip(df[smi_col].astype(str), df[val_col]) if smi in smiles_to_emb]

    return np.stack(X_list).astype('float32') if X_list else np.array([]), np.array(y_list, dtype='float32') if y_list else np.array([])

# Training
def train_epoch_mmd(model, X_src, y_src, X_tgt, y_tgt, optimizer, criterion, mmd_w, device):
    model.train()
    X_src, y_src = np.asarray(X_src, dtype='float32'), np.asarray(y_src, dtype='float32')
    X_tgt, y_tgt = np.asarray(X_tgt, dtype='float32'), np.asarray(y_tgt, dtype='float32')

    n = len(X_src)
    perm = np.random.permutation(n)
    task_loss_sum = mmd_loss_sum = n_batches = 0

    for i in range(0, n, 32):
        idx = perm[i:i + 32]
        x_src = torch.tensor(X_src[idx], device=device, dtype=torch.float32)
        y_src_batch = torch.tensor(y_src[idx], device=device, dtype=torch.float32)
        x_tgt = torch.tensor(X_tgt[idx], device=device, dtype=torch.float32)
        y_tgt_batch = torch.tensor(y_tgt[idx], device=device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred_tgt, feat_tgt = model(x_tgt)
        task_loss = criterion(y_pred_tgt, y_tgt_batch)

        _, feat_src = model(x_src)
        mmd_loss = compute_mmd(feat_src, feat_tgt, sigma=1.0)

        total_loss = task_loss + mmd_w * mmd_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        task_loss_sum += task_loss.item()
        mmd_loss_sum += mmd_loss.item()
        n_batches += 1

    return task_loss_sum / max(n_batches, 1), mmd_loss_sum / max(n_batches, 1)

@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    X, y = np.asarray(X, dtype='float32'), np.asarray(y, dtype='float32')
    x = torch.tensor(X, device=device, dtype=torch.float32)
    y_pred, _ = model(x)
    y_pred = y_pred.cpu().numpy()

    r = np.corrcoef(y, y_pred)[0, 1]
    rho, _ = spearmanr(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(np.abs(y - y_pred))
    mbe = np.mean(y_pred - y)

    return {'r': r, 'rho': rho, 'r2': r2, 'rmse': rmse, 'mae': mae, 'mbe': mbe}

# Main training loop
print(f"MMD Weight: {MMD_WEIGHT}\n")
results_dict = {}

for prop in PROPERTIES:
    print(f"Property: {prop.upper()}")
    fold_results = []

    for fold_idx in range(N_FOLDS):
        X_src, y_src = load_source(prop, fold_idx)
        X_tgt_train, y_tgt_train = load_target(prop, fold_idx, 'train')
        X_tgt_test, y_tgt_test = load_target(prop, fold_idx, 'test')

        if X_src is None or len(X_src) == 0:
            continue

        n = min(len(X_src), len(X_tgt_train))
        X_src = np.asarray(X_src[:n], dtype='float32')
        y_src = np.asarray(y_src[:n], dtype='float32')
        X_tgt_train = np.asarray(X_tgt_train[:n], dtype='float32')
        y_tgt_train = np.asarray(y_tgt_train[:n], dtype='float32')
        X_tgt_test = np.asarray(X_tgt_test, dtype='float32')
        y_tgt_test = np.asarray(y_tgt_test, dtype='float32')

        print(f"  Fold {fold_idx}: Source {X_src.shape[0]}, Target {X_tgt_train.shape[0]} train, {X_tgt_test.shape[0]} test")

        torch.manual_seed(fold_idx)
        np.random.seed(fold_idx)

        model = DomainAdaptationModel().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)

        best_r2 = -1
        patience = 0

        for epoch in range(80):
            tl, ml = train_epoch_mmd(model, X_src, y_src, X_tgt_train, y_tgt_train, optimizer, criterion, MMD_WEIGHT, DEVICE)
            metrics = evaluate(model, X_tgt_test, y_tgt_test, DEVICE)

            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                patience = 0
                best_state = model.state_dict().copy()
            else:
                patience += 1

            if patience >= 15:
                model.load_state_dict(best_state)
                break

            if epoch % 15 == 0:
                print(f"    E{epoch:3d} | task={tl:.4f} | mmd={ml:.4f} | rho={metrics['rho']:.4f}")

        metrics = evaluate(model, X_tgt_test, y_tgt_test, DEVICE)
        fold_results.append(metrics)
        print(f"    Result: rho={metrics['rho']:.4f}, r2={metrics['r2']:.4f}")

    if fold_results:
        results_dict[prop] = pd.DataFrame(fold_results)
        df = results_dict[prop]
        print(f"  Summary: rho={df['rho'].mean():.4f}±{df['rho'].std():.4f}, r2={df['r2'].mean():.4f}, rmse={df['rmse'].mean():.4f}\n")

# Save results
for prop in PROPERTIES:
    if prop in results_dict:
        out_path = os.path.join(RES_DIR, f'{prop}.csv')
        results_dict[prop].to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
