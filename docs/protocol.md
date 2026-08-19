# Protocol — Methodology Specification

What the pipeline does and which artefact holds which number. The reasoning, the
related work and the limitations are in the thesis.

## The protocol

Each dataset is split **once** into a stratified 80 % training partition and an
independent 20 % test partition. Everything that could adapt to data is decided
inside the training partition; the test partition is read exactly once per model.

```
Full dataset
├── Training partition 80 %
│   ├── 5-fold CV for hyperparameter selection (mean validation ROC-AUC)
│   ├── refit of the selected model on all training data
│   ├── 3-fold internal isotonic calibration
│   └── out-of-fold thresholds, training data only
└── Test partition 20 %
    └── one final evaluation per model
```

## Datasets

Read only from `data/processed/v4/*.csv`. The column contract is declared in
`src/datasets.py`; no runner derives column semantics elsewhere.

| Key | Display name | File | Rows | Categorical | Auxiliary | ROI |
|---|---|---|---|---|---|---|
| `south_german` | German Credit (Statlog) | `south_german_credit.csv` | 1,000 | 5 | – | 0.2644 |
| `taiwan` | Taiwan Credit Card Default | `taiwan_credit.csv` | 30,000 | 2 | – | 0.2644 |
| `home_credit` | Home Credit Default Risk | `home_credit.csv` | 307,511 | 16 | – | 0.2644 |
| `lending_club` | Lending Club | `lending_club_full.csv` | 1,345,350 | 7 | `int_rate`, `issue_year` | mean `int_rate` of the training partition / 100 |

**Semi-raw export contract.** Notebooks 01–04 apply only deterministic, row-wise
operations: target definition, identifier and leakage drops, documented sentinel
repairs (`DAYS_EMPLOYED == 365243`, `XNA`), externally justified ordinal maps
(`german.doc`, `term`, `emp_length`), the Lending Club status filter. No sample
statistic is permitted, so missing values, raw category levels and unclipped outliers
survive the export. Every file carries `source_row_id` in export order; it is the key
for all later joins, and positional joins are forbidden.
`src/semiraw.py::audit_semiraw_export` checks the contract against the written file.

## Seeds

`src/seeds.py` is the only source; the values are frozen.

| Purpose | Seed |
|---|---|
| global (`random`, `numpy`) | 42 |
| train/test split | 4201 |
| hyperparameter CV / randomized search | 4202 / 4203 |
| calibration CV / threshold OOF CV | 4204 / 4205 |
| LR / EBM / XGBoost / CatBoost | 4210 / 4211 / 4212 / 4213 |
| bootstrap / explanation sampling / learning curves | 4220 / 4221 / 4230 |

Also frozen: `np.nanquantile(method="linear")`, search tie epsilon 1e-6, 95 %
percentile bootstrap intervals, reproducibility tolerances rtol 1e-9 / atol 1e-12.
The chain sets `PYTHONHASHSEED=0`. `models.positive_class_column` verifies the
positive-class index in `classes_` rather than assuming it.

## Split artefacts

`src/splits.py` writes `split_manifest.json` and `split_indices.npz` before the first
fit. The manifest pins the file SHA-256, the column list, target and ID hashes, both
partitions (sizes, class counts, default rates, membership hashes) and three fold
families inside the training partition:

| Family | Splits | Seed | Purpose |
|---|---|---|---|
| `hyperparameter_cv` | 5 | 4202 | candidate comparison |
| `calibration_cv` | 3 | 4204 | isotonic calibration |
| `threshold_oof_cv` | 5 | 4205 | out-of-fold scores for the EMP cut-off |

Every re-run validates against the manifest and aborts with `SplitIdentityError`
before anything is fitted; `FORCE` does not bypass this. All four models load the same
indices.

## Preprocessing (`src/preprocessing.py`)

Fitted on the current fit subset, in this order:

1. `MissingnessFilter(0.40)`
2. `ColumnTransformer` — numeric: `Winsorizer` (upper P99) → median imputation;
   binary numeric: most-frequent imputation; categorical: `UnknownFiller` →
   `SafeOneHotEncoder`
3. `StandardScaler` **only** inside the LR pipeline

Validation and test data receive `transform` only. The encoder uses `drop=None`, and
`Unknown` is always part of the schema, so a missing value stays distinguishable from
an unseen category (all-zero row). Every `transform` writes per-feature diagnostics to
`ohe_diagnostics.json` and `preprocessing_report.json`, labelled by partition.

## Class imbalance

No over- or undersampling; weights are computed inside every `fit` from the `y` handed
to it and never reach the calibration stage.

| Model | Mechanism |
|---|---|
| LR | `class_weight="balanced"` |
| EBM | `BalancedEBMClassifier` → `compute_sample_weight` in the base fit |
| XGBoost, CatBoost | `FitLocalPosWeightClassifier` → `scale_pos_weight` in the fit |

## Hyperparameter selection

One search space for all datasets (`src/search.py::PARAM_GRIDS`), parameter names
pipeline-prefixed. EBM uses `outer_bags=8` throughout. The number of boosting rounds is
not a search parameter.

| Model | Grid | Budget | Coverage | Searcher |
|---|---|---|---|---|
| LR | 6 | 6 | 100 % | `GridSearchCV` |
| EBM | 96 | 20 | 21 % | `RandomizedSearchCV` |
| XGBoost | 144 | 30 | 21 % | `search.py::_run_es_search` |
| CatBoost | 27 | 20 | 74 % | `search.py::_run_es_search` |

`_run_es_search` replicates `RandomizedSearchCV` semantics — same sampler, seed 4203,
same folds, ROC-AUC on the validation fold — and additionally hands that fold to the
wrapper as the early-stopping evaluation set, which sklearn's searchers cannot route.

Settings: `scoring="roc_auc"`, `error_score="raise"`, `n_jobs=1`, `cv` = the five
materialised training folds, `refit=False` plus an explicit
`clone().set_params().fit()` in the runner.

`select_best` breaks ties in a fixed order: highest mean validation AUC → smaller
standard deviation → documented simpler configuration → lexicographically first
serialised parameter set; ties declared below |Δ| < 1e-6. The resolved path goes to
`selection.json`, all candidates to `cv_results.{json,csv}`. CV scores are a selection
diagnostic and are never reported as generalisation estimates.

**Early stopping is confined to the search**: patience 50, cap 2,000, metric AUC
(`src/seeds.py::ES_PATIENCE` / `ES_MAX_ROUNDS`). The median best iteration + 1 travels
inside the selected configuration, so refit, calibration, threshold cross-fitting,
ablation and learning curves are plain capped fits. `tests/test_seeds_and_contract.py`
enforces that no fit outside the search receives an evaluation set.

## Refit, calibration, thresholds

Refit on the complete training partition → `final_pipeline.joblib`, the uncalibrated
base model and the explanation object. Calibration is
`CalibratedClassifierCV(method="isotonic", cv=<three training folds>, ensemble=False,
n_jobs=1)` — no `sample_weight` reaches the calibrator. `p_raw` and `p_calibrated` are
persisted separately. A prespecified sigmoid-versus-isotonic comparison runs on German
Credit only; isotonic was fixed as the primary method before any test score.

Two thresholds, both written to `thresholds.json` before the test evaluation:

- **base-rate threshold** — the training default rate, applied to calibrated test
  probabilities as a descriptive operating point, not an optimal lending threshold;
- **economic cut-off** — the pipeline is cross-fitted over five training folds so every
  training row gets exactly one out-of-fold prediction (asserted); those raw scores
  yield η and the cut-off is the (1−η) quantile, then frozen and applied unchanged to
  the raw test scores.

`economic_config.json` is written **before the first test access**: training-partition
priors, ROI and its provenance, loss-given-default mixture (p0 = 0.55, p1 = 0.10), η
and cut-off per model, EMP implementation version. `emp_credit_scoring` requires the
prior as a mandatory argument; the observed test default rate is descriptive only and
is never substituted, including in the ROI sensitivity analysis, where ROI alone varies.

## Final evaluation

Exactly one evaluation per model and dataset, through `src/test_access.py`.

| Metric | Score version |
|---|---|
| ROC-AUC, PR-AUC, **EMP**, η | raw |
| Brier | calibrated |
| Precision / Recall / F1 at the EMP cut-off | raw |
| Balanced accuracy, F1, Precision, Recall at the base-rate threshold | calibrated |

EMP is stored as a dimensionless fraction of principal per applicant. Only the
presentation layer multiplies by 100: levels as percent of principal, absolute paired
differences as percentage points — neither is a relative percentage change.

**Governance.** `analysis_plan.json` is frozen before the first test access and lists
every permitted analysis. `test_access_log.jsonl` records each access: timestamp,
script, git commit, dataset, analysis ID, artefact, row count, `fit_occurred` flag.
The final-evaluation path checks via `assert_no_fit` that the model is already fitted.
No test result may change preprocessing, hyperparameters, calibration, thresholds,
metric definitions or model selection.

## Uncertainty

Paired stratified bootstrap on the fixed test partition: 1,000 replicates, seed 4220,
defaults and non-defaults drawn separately so every replicate keeps the observed class
counts. **One** index matrix per dataset serves all models; models, calibrators and
thresholds stay frozen and nothing is refitted inside a replicate. Reported: point
estimates, 95 % percentile intervals, and pairwise differences (notably ΔROC-AUC) with
intervals. No DeLong test.

## Explanations

The explanation object is always the **uncalibrated** final model on the complete
training partition.

- Rankings come from mean |local contribution|, never from raw coefficient magnitudes.
- One-hot levels are aggregated onto their source feature before any cross-model
  comparison; the absolute value is taken at column level, then summed
  (`aggregate_to_source_features(np.abs(values), ...)`).
- All models explain the same persisted test observations — 2,000 cases; German Credit:
  the whole test partition.
- Comparison happens on the raw score before calibration.
- EBM interaction terms are split equally between the participating source features.
- The case bootstrap measures uncertainty across applicants, not retraining stability.

Explainers: `LinearExplainer` (LR, interventional, background 500), `explain_local`
(EBM), `TreeExplainer` with `tree_path_dependent` and no background (XGBoost, CatBoost).
No KernelSHAP. **CatBoost trains on the same one-hot representation as every other
model and does not use its native ordered target statistics.**

Reported: per model, top-10 membership frequencies and rank intervals over 1,000 paired
case bootstraps; across models, Spearman correlation of the point vectors and top-10
Jaccard per replicate with 95 % intervals. Faithfulness tests are not implemented.

## Learning curves

`08_learning_curves.py` draws subsamples from the **training partition** and evaluates
on the **same** test partition; the split is loaded from `split_indices.npz`, not
recreated. Sizes 10K/50K/100K/250K, three repetitions (seed 4230 + rep), fixed
configurations. Only per-model slopes are interpreted, not levels.

## Runners and outputs

| Script | Role | Output |
|---|---|---|
| `notebooks/01–04_preprocessing_*.ipynb` | semi-raw exports | `data/processed/v4/` |
| `notebooks/05_fixed_split_all.py` | **main protocol**, 4 datasets × 4 models | `results/fixed_split_v4/` |
| `notebooks/06_lc_oot.py` | out-of-time robustness (cutoff 2015) | `results/oot_v4/` |
| `notebooks/07_lc_data_audit.py` | Lending Club maturity and feature-timing audit | `results/data_audit_v4/` |
| `notebooks/08_learning_curves.py` | learning curves | `results/learning_curves_v4/` |
| `notebooks/09_posthoc_from_scores.py` | Brier comparison, ROI grid, fairness probe | `results/posthoc_v4/` |
| `notebooks/10_ablation_protected_taiwan.py` | protected-feature ablation | `results/posthoc_v4/` |
| `scripts/plot_emp_figures.py` | EMP level and ROI-sensitivity figures | `results/figures/` |
| `scripts/plot_pairwise_forest.py` | cross-group forest plot | `results/figures/` |

Both plot scripts read persisted results only — no fit, no prediction, no EMP
recomputation, no resampling.

Per dataset under `results/fixed_split_v4/<dataset>/`: `split_manifest.json`,
`split_indices.npz`, `economic_config.json`, `final_metrics.csv`, `bootstrap_ci.csv`,
`auc_delta_bootstrap.csv`, `stability_summary.json`, `shap_importance.csv`,
`cross_model_spearman.csv`, `ohe_diagnostics.json`, `explanation_sample_ids.csv`,
`run_manifest.json`; per model `cv_results.{json,csv}`, `selection.json`,
`final_pipeline.joblib`, `calibrated_pipeline.joblib`, `thresholds.json`,
`oof_train_scores.npz`, `scores.npz`, `abs_attributions.npz`,
`preprocessing_report.json`. `analysis_plan.json` and `test_access_log.jsonl` sit one
level above.

Full chain: `bash run_all_sodalab04.sh` — requires tmux and a clean git tree, then runs
preflight, smoke, 05 → 06 → 07 → 08 → 09 → 10 and the output check. Stage checkpoints
carry a `code_fingerprint`, so a resume after a code change aborts instead of mixing
versions (resolve by deleting the output directory or setting `FORCE=1`). The searches
themselves cannot resume, which is why Lending Club runs last.

## Reproducibility and tests

Identical across repeated runs: splits and fold assignments, sampled candidates,
selected hyperparameters, fitted preprocessing parameters, predictions, thresholds,
metric values, explanation rankings. Excluded from byte equality: timestamps, runtimes,
access-log ordering, temporary paths.

`python -m pytest` (see `pytest.ini`), 125 cases: split integrity, preprocessing
leakage, weighting and calibration, search, early stopping, thresholds and economics,
bootstrap and test-access governance, explanations, seeds and data contract, EMP
presentation, reproducibility.
