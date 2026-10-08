# Curated datasets

Only the curated sets used in MDTAF are included here. Raw downloads, intermediate files and features are not committed; they can be regenerated with the notebooks in `0_data_process/`.

## `normalized_data/`

| File | Domain | Source | `value` |
|---|---|---|---|
| `{endpoint}_source.csv` | in vitro ADME | ChEMBL, TDC, Biogen | log10, then z-score over the whole source set |
| `{endpoint}_target.csv` | human in vivo PK | PKSmart | log10 (z-scored per CV fold, see below) |

Endpoints: `fu` (fraction unbound), `clearance`, `half_life`.

Curation: canonical RDKit SMILES; duplicate measurements from the same document and assay are merged by mean, while independent measurements are kept; only exact (`=`) ChEMBL values are used. See `0_data_process/` for details.

## `cv_datasets/`

Bemis–Murcko scaffold 5-fold split of each target set (`fold_0` … `fold_4`).

- `{endpoint}_target_train.csv`, `{endpoint}_target_test.csv`: z-scored with that fold's train statistics.
- `{endpoint}_source.csv`: the source set with that fold's test compounds removed (no leakage).
- `target_norm_stats.csv`: per-fold `log10_mean` and `log10_std` for converting predictions back: `log10(value) = z * log10_std + log10_mean`.
- `cv_summary.csv`: sizes of every split.

## Licence and attribution

These sets are derived from public sources. Please cite and respect the terms of each original source: ChEMBL (CC BY-SA 3.0), Therapeutics Data Commons, the Biogen ADME dataset (Fang et al., 2023) and PKSmart.
