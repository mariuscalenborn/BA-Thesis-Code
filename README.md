# Credit Scoring — Performance–Explainability Trade-off (Thesis Code)

Code repository for the bachelor thesis *Evaluating the Performance–Explainability
Trade-off in Credit Scoring Under Regulatory Constraints*. It contains the
complete pipeline that produces every reported figure, together with the lightweight
result artefacts the thesis reports. The methodology specification lives in
[`docs/protocol.md`](docs/protocol.md) and is kept in sync with the code.

Four models — Logistic Regression, Explainable Boosting Machine, XGBoost, CatBoost —
are evaluated on four datasets under one common protocol: a single fixed stratified
80/20 split per dataset, hyperparameter search inside the training partition only,
refit, isotonic calibration, frozen thresholds, and exactly one test evaluation.

## Layout

| Path | Purpose |
|---|---|
| `notebooks/01–04_preprocessing_*.ipynb` | Raw → **semi-raw** CSV per dataset (German Credit/Statlog, Taiwan, Home Credit, Lending Club); deterministic operations only |
| `notebooks/05_fixed_split_all.py` | **Primary protocol**: one fixed stratified 80/20 split per dataset, training-only five-fold search, refit, calibration, frozen thresholds, one test evaluation, explanations, paired bootstrap |
| `notebooks/06_lc_oot.py` | Lending Club single out-of-time split (separate robustness experiment; cutoff 2015) |
| `notebooks/07_lc_data_audit.py` | Lending Club outcome-maturity / feature-timing audit |
| `notebooks/08_learning_curves.py` | Learning curves (Home Credit, Lending Club; frozen configs, main protocol's split) |
| `notebooks/09_posthoc_from_scores.py` | Raw-vs.-calibrated Brier, EMP-ROI sensitivity, and fairness probe from persisted scores (no retraining) |
| `notebooks/10_ablation_protected_taiwan.py` | Protected-attribute ablation (Taiwan) reusing the selected configurations |
| `scripts/plot_pairwise_forest.py` | Deterministic forest plot from the persisted paired-bootstrap summaries; no fitting or resampling |
| `scripts/plot_emp_figures.py` | Deterministic EMP interval and ROI-sensitivity figures from persisted raw EMP fractions; no fitting or resampling |
| `src/` | Shared modules: `seeds.py`, `datasets.py`, `preprocessing.py`, `models.py`, `search.py`, `splits.py`, `thresholds.py`, `evaluation.py`, `bootstrap.py`, `test_access.py`, `semiraw.py`, `run_utils.py` |
| `run_all_sodalab04.sh` | Full production chain, six stages |
| `tests/` | Split integrity, preprocessing leakage, weighting, search, thresholds, bootstrap, test-access governance, explanations, reproducibility |
| `results/` | Metrics, selected hyperparameters, bootstrap intervals, explanation rankings and figures of the reported run |
| `docs/protocol.md` | Binding methodology specification |
| `thesis/` | LaTeX sources of the thesis itself: `thesis.tex`, the eight chapter files under `text/`, `bibliography.bib` and the figures in `img/` |

## Data

Only the two UCI-derived semi-raw exports ship with this repository. The raw archives
do not: they total roughly 6.5 GB, and the two Kaggle sources require accepting their
terms. Each dataset below names its landing page, a direct link where one exists, the
file to keep, and the exact path it belongs at. Every command runs from the
repository root and leaves you there.

### 1. German Credit (Statlog) — Hofmann (1994)

- Landing page: <https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data>
- Direct: <https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip>

```bash
mkdir -p data/raw/south_german
curl -L -o /tmp/statlog.zip https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip
unzip -j /tmp/statlog.zip german.data -d data/raw/south_german
rm /tmp/statlog.zip
```

Keep `german.data` → `data/raw/south_german/german.data` (78 KB).
Take the **original** Statlog file, not Grömping's corrected South German Credit
re-release — the two differ in several columns.

### 2. Taiwan Credit Card Default — Yeh & Lien (2009)

- Landing page: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- Direct: <https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip>

```bash
mkdir -p data/raw/taiwan
curl -L -o /tmp/taiwan.zip https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
unzip -j /tmp/taiwan.zip -d data/raw/taiwan
rm /tmp/taiwan.zip
```

Keep `default of credit card clients.xls` → `data/raw/taiwan/` (5.3 MB).

### 3. Home Credit Default Risk — Kaggle competition

- Landing page: <https://www.kaggle.com/c/home-credit-default-risk/data>
- Requires a Kaggle account and accepting the competition rules on that page first.

```bash
pip install kaggle          # already in requirements.txt
kaggle competitions download -c home-credit-default-risk -f application_train.csv \
  -p data/raw/home_credit
unzip -o data/raw/home_credit/application_train.csv.zip -d data/raw/home_credit
rm data/raw/home_credit/application_train.csv.zip
```

Keep `application_train.csv` → `data/raw/home_credit/` (158 MB). Only this table is
used; `bureau`, `previous_application` and the other auxiliary files are not — they
carry post-approval information about other loans.

### 4. Lending Club — Kaggle dataset

- Landing page: <https://www.kaggle.com/datasets/wordsforthewise/lending-club>

```bash
kaggle datasets download -d wordsforthewise/lending-club \
  -f accepted_2007_to_2018Q4.csv -p data/raw/lending_club/archive
```

Keep the accepted-loans file → `data/raw/lending_club/archive/`. Notebook 04 resolves
it in three shapes and reports which it used, so any of these works:

```
data/raw/lending_club/archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv
data/raw/lending_club/archive/accepted_2007_to_2018Q4.csv
data/raw/lending_club/archive/accepted_2007_to_2018Q4.csv.gz
```

pandas reads the gzipped form directly, so there is no need to decompress (374 MB
instead of 1.6 GB). The `rejected_*` file of the same dataset is not used.

### Verifying what you downloaded

SHA-256 of the files this study ran on:

| File | SHA-256 |
|---|---|
| `german.data` | `b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871` |
| `default of credit card clients.xls` | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| `application_train.csv` | `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f` |
| `accepted_2007_to_2018Q4.csv.gz` | `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d` |

```bash
shasum -a 256 data/raw/south_german/german.data
```

A mismatch on the Lending Club line most likely means you kept the uncompressed CSV
rather than the gzipped one, which is fine — the hash simply covers the gzipped form.

### Building the semi-raw exports

Run notebooks 01–04. They write to `data/processed/v4/` and apply only deterministic,
row-wise operations: categorical columns stay raw, missing values survive, no column
is dropped for missingness and nothing is clipped. Everything that depends on a sample
statistic — missingness filter, one-hot schema, imputation, winsorization, scaling —
is fitted at runtime inside the pipeline, per fit subset. The chain's preflight refuses
to start on a missing, encoded or clipped export.

```
data/processed/v4/south_german_credit.csv     ships with this repository
data/processed/v4/taiwan_credit.csv           ships with this repository
data/processed/v4/home_credit.csv             build with notebook 03
data/processed/v4/lending_club_full.csv       build with notebook 04
```

The two UCI-derived exports are small and freely licensed, so they are included here:
the test suite and the smoke run below work straight after cloning, with no download
and no Kaggle account. The two Kaggle-derived exports are not redistributed.

## Setup

```bash
conda env create -f environment.yml
conda activate credit-xai
```

or, without conda:

```bash
pip install -r requirements.txt
```

Python 3.11, `scikit-learn>=1.6`, `xgboost>=2.0`, `catboost==1.2.10`, `interpret>=0.6`.
Set `PYTHONHASHSEED=0` for any run whose output you intend to compare.

## Running

### 1. Tests — no data required

```bash
python -m pytest
```

Covers split integrity, preprocessing leakage, class weighting, calibration, search,
thresholds, bootstrap and the test-access governance.

### 2. Smoke run — CPU, a few minutes

The fastest way to see the whole primary runner end to end. Needs only the German
Credit export from notebook 01:

```bash
OUTBASE=results/smoke_v4 DATASETS=south_german N_MAX=400 USE_GPU=0 \
  LR_ITER=2 EBM_ITER=1 XGB_ITER=1 CAT_ITER=1 EXPLAIN_CASES=50 \
  EXPLAIN_BACKGROUND=50 N_BOOTSTRAP=10 python notebooks/05_fixed_split_all.py
```

This writes to `results/smoke_v4/` and leaves the reported results untouched.

### 3. Full chain — GPU recommended, several days

```bash
tmux new -s v4
bash run_all_sodalab04.sh
```

Six stages: primary fixed split (German Credit → Taiwan → Home Credit → Lending Club),
Lending Club out-of-time, Lending Club data audit, learning curves, post-hoc analyses,
output check. Runtime is dominated by Lending Club (1.35 M resolved loans), which is why
it runs last — the stages after the primary run resume from checkpoints, but the
hyperparameter searches do not: `GridSearchCV`/`RandomizedSearchCV` offer no candidate
checkpoint, so a crash restarts that model's search.

The script requires `tmux` (an SSH drop would kill a multi-day run) and a clean git
tree, and it calls `nvidia-smi` when running on GPU. Overrides:

```bash
ALLOW_NO_TMUX=1   # run outside tmux
USE_GPU=0         # CPU only; skips the nvidia-smi check
FORCE=1           # ignore existing checkpoints
```

Individual runners accept environment overrides (`DATASETS`, `N_MAX`, `LC_N`, `*_ITER`,
`EXPLAIN_CASES`, `N_BOOTSTRAP`, `OUTBASE`, `OUTDIR`).

### 4. Re-running over the shipped results

`results/` already holds the artefacts of the reported run, and most runners skip
work they find finished. To recompute rather than inspect, choose one of:

```bash
# a) write somewhere else, leaving the shipped results untouched (recommended)
OUTBASE=results/rerun python notebooks/05_fixed_split_all.py

# b) recompute in place
FORCE=1 AUDIT_FORCE=1 bash run_all_sodalab04.sh

# c) start from nothing
rm -rf results && bash run_all_sodalab04.sh
```

Stage checkpoints are deliberately not shipped: they carry the code fingerprint of
the run that wrote them, so any later run would refuse to resume from them.

### 5. Figures

Both read `results/` only, so they run without any raw data:

```bash
python scripts/plot_emp_figures.py \
  --fixed-split results/fixed_split_v4 \
  --posthoc-input results/posthoc_v4/emp_roi_grid.csv \
  --output-dir results/figures --png

python scripts/plot_pairwise_forest.py \
  --input results/fixed_split_v4 \
  --output results/figures/cross_group_differences.pdf
```

They perform no model fitting, prediction, EMP recomputation or bootstrap
resampling.

## Results

`results/` holds the artefacts of the reported run. Where to find which number:

| File | Contents |
|---|---|
| `results/fixed_split_v4/<dataset>/final_metrics.csv` | AUC-ROC, PR-AUC, EMP, Brier, balanced accuracy, F1 per model |
| `results/fixed_split_v4/<dataset>/auc_delta_bootstrap.csv` | Paired stratified bootstrap: ΔAUC point estimate and 95 % interval for all model pairs |
| `results/fixed_split_v4/<dataset>/bootstrap_ci.csv` | Per-model bootstrap intervals |
| `results/fixed_split_v4/<dataset>/<model>/selection.json` | Selected hyperparameters, search budget, early-stopping rounds |
| `results/fixed_split_v4/<dataset>/shap_importance.csv` | Attributions aggregated to source features |
| `results/fixed_split_v4/<dataset>/stability_summary.json` | Top-10 membership frequencies, rank intervals, cross-model Jaccard |
| `results/fixed_split_v4/<dataset>/cross_model_spearman.csv` | The six pairwise cross-model Spearman correlations |
| `results/fixed_split_v4/<dataset>/economic_config.json` | Frozen EMP parameters: training-partition priors, ROI, LGD |
| `results/oot_v4/lending_club/` | Out-of-time split (cutoff 2015) |
| `results/learning_curves_v4/` | AUC and Top-10 Jaccard by training size (Home Credit, Lending Club) |
| `results/posthoc_v4/` | EMP-ROI sensitivity grid, raw-vs.-calibrated Brier, fairness probe, Taiwan ablation |
| `results/data_audit_v4/` | Lending Club outcome-maturity audit |
| `results/figures/` | EMP figures and the cross-group forest plot |

Fitted pipelines (`*.joblib`) and the per-row score arrays are deliberately not
versioned — they are large and runner 05 regenerates them. `split_indices.npz` **is**
versioned, so the split can be verified without a rerun.

## Reproducibility

Each production run writes a `run_manifest.json` (git commit, package versions,
SHA-256 of inputs and code, feature list, explained sample IDs, the full seed registry)
and a `split_manifest.json` plus `split_indices.npz` that pin both partitions and all
three training-internal fold families — hyperparameter search (5), calibration (3) and
threshold OOF (5) — by hashed source-row IDs *before* the first model fit. Later runs
abort on any mismatch. All four models load the same indices. All seeds live in
`src/seeds.py` (42 / 4201–4230); set `PYTHONHASHSEED=0`.

Test-set access is governed: `results/fixed_split_v4/analysis_plan.json` lists every
permitted analysis and `test_access_log.jsonl` records each access.

Reproducibility comparisons distinguish deterministic artefacts (splits, sampled
candidates, selected hyperparameters, fitted preprocessing parameters, predictions,
thresholds, metrics, explanation rankings) from volatile execution metadata
(timestamps, durations, log ordering, temporary paths), which is excluded.

That determinism holds **within a fixed environment and device**. The reported run
used Python 3.11 on GPU (`USE_GPU=1`); every `run_manifest.json` records the exact
interpreter and package versions. Re-running under a different Python, a different
XGBoost or CatBoost release, or on CPU reproduces the split, the fold assignments and
the selected configurations exactly, and logistic regression bit-identically, but the
boosted ensembles and the EBM can land on slightly different test metrics — their
fits depend on library version and device. Use `environment.yml` for an exact
comparison.

EMP is stored internally as a decimal fraction of principal. In reported results, EMP
levels are displayed as percentages of principal and absolute EMP differences as
percentage points; these are not relative percentage changes.
