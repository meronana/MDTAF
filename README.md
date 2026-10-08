# MDTAF

A systematic framework for assessing molecular domain transferability in pharmacokinetic prediction.

MDTAF asks, before any transfer model is trained, whether a source domain (in vitro ADME assays) can help predict a target domain (human in vivo PK). It profiles each source–target pair along input-space and label-space dimensions, then checks that profile against what domain adaptation actually delivers for fraction unbound (f<sub>u</sub>), clearance (CL) and half-life (t<sub>½</sub>).

## Repository layout

| Path | Contents |
|---|---|
| `0_data_process/` | Data collection (ChEMBL, TDC, Biogen, PKSmart), deduplication, normalisation, scaffold CV split, Graphormer embedding extraction |
| `benchmark/` | The nine-phase transferability assessment (`phase_01` … `phase_09`) |
| `01_model/` | Baselines, domain adaptation (MMD, CORAL, DANN, CDAN, importance weighting), significance tests, proposed label-aware methods, external validation and ablations |
| `src/` | Shared models, DA losses, Graphormer encoder and the `transferability_assessment` package |
| `data/` | Curated datasets and scaffold CV splits (see [`data/README.md`](data/README.md)) |
| `framwork/` | Project website (Vite) |

Notebooks are numbered in the order they were run. Paths are relative to the notebook's folder, so run Jupyter from inside `0_data_process/`, `01_model/` or `benchmark/`.

## Requirements

Python 3.10+ with PyTorch, RDKit, scikit-learn, XGBoost, pandas, NumPy, SciPy and matplotlib. Graphormer embedding extraction and fine-tuning need a CUDA GPU.

Large intermediate files (`features/`, `checkpoints/`, `results/`) are not committed. Rebuild them in order: `0_data_process/08_extract_graphormer_embeddings_v2.ipynb` for the embeddings, then the notebooks in `benchmark/` and `01_model/`.

## Website

```bash
cd framwork
npm install
npm run dev
```

`npm run data` refreshes the site's numbers from `results/`, and `python framwork/scripts/export_downloads.py` refreshes the data-page manifest after the curated data change.

## Data and licence

The curated datasets are derived from public sources. Please cite and respect the terms of each original source: ChEMBL (CC BY-SA 3.0), Therapeutics Data Commons, the Biogen ADME dataset (Fang et al., 2023) and PKSmart.
