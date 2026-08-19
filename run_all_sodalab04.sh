#!/usr/bin/env bash
# v4 production chain (see docs/protocol.md). Cheap before expensive; the stages
# after the primary run are resumable, so re-running this script continues the
# chain. The hyperparameter searches themselves are not resumable -- a crash
# inside one restarts that model's search, which is why Lending Club runs last.
set -Eeuo pipefail
cd "$(dirname "$0")"

if [[ -z "${TMUX:-}" && "${ALLOW_NO_TMUX:-0}" != "1" ]]; then
  echo "FATAL: run inside tmux (SSH drops would kill a multi-day run)."
  echo "       tmux new -s v4   # then rerun; ALLOW_NO_TMUX=1 overrides."
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "FATAL: uncommitted changes detected. Run only from the committed baseline."
  git status --short
  exit 1
fi

mkdir -p logs results

export USE_GPU="${USE_GPU:-1}"
export EBM_BAGS=8
export FORCE="${FORCE:-0}"
# Hash randomisation would make set and dict iteration order vary between runs.
export PYTHONHASHSEED=0

echo "### GPU + preflight ###"
if [[ "$USE_GPU" == "1" ]]; then
  nvidia-smi
else
  echo "USE_GPU=0 -- CPU run, skipping nvidia-smi."
fi
python - <<'EOF'
import sys
from pathlib import Path

sys.path.insert(0, ".")
import pandas as pd

from src.datasets import DATASETS, ID_COLUMN, validate_frame

for name, config in DATASETS.items():
    path = Path(config.semiraw_path)
    if not path.exists():
        raise SystemExit(f"FATAL: missing semi-raw CSV: {path} (run notebooks 01-04)")

# Cheap structural checks on the whole file, then a full contract check on the
# two small datasets. The large ones are validated inside the runner.
for name, config in DATASETS.items():
    header = pd.read_csv(config.semiraw_path, nrows=0).columns.tolist()
    if ID_COLUMN not in header:
        raise SystemExit(f"FATAL: {name} lacks {ID_COLUMN}; regenerate the export")
    missing_categoricals = [c for c in config.categorical_cols if c not in header]
    if missing_categoricals:
        raise SystemExit(f"FATAL: {name} lacks raw categorical columns {missing_categoricals}")

sample = pd.read_csv(
    DATASETS["home_credit"].semiraw_path,
    usecols=["AMT_INCOME_TOTAL", "CODE_GENDER"],
)
if sample["AMT_INCOME_TOTAL"].max() <= 1_000_000:
    raise SystemExit("FATAL: home_credit export appears globally clipped")
if pd.api.types.is_numeric_dtype(sample["CODE_GENDER"]):
    raise SystemExit("FATAL: home_credit CODE_GENDER is encoded; the export must keep it raw")

sample = pd.read_csv(
    DATASETS["lending_club"].semiraw_path, usecols=["annual_inc", "int_rate", "issue_year"]
)
if sample["annual_inc"].max() <= 250_000:
    raise SystemExit("FATAL: lending_club export appears globally clipped")

for name in ("south_german", "taiwan"):
    validate_frame(DATASETS[name], pd.read_csv(DATASETS[name].semiraw_path))

print("Preflight OK: semi-raw exports present, unencoded, unclipped")
EOF

echo "### Mini-smoke: one tiny end-to-end pass per runner ###"
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  OUTBASE=results/smoke_v4/fixed_split DATASETS=south_german N_MAX=400 \
    LR_ITER=2 EBM_ITER=1 XGB_ITER=1 CAT_ITER=1 \
    EXPLAIN_CASES=50 EXPLAIN_BACKGROUND=50 N_BOOTSTRAP=10 \
    python notebooks/05_fixed_split_all.py 2>&1 | tee logs/v4_smoke_fixed_split.log
  OUTDIR=results/smoke_v4/oot LC_N=20000 \
    LR_ITER=2 EBM_ITER=1 XGB_ITER=1 CAT_ITER=1 \
    EXPLAIN_CASES=50 EXPLAIN_BACKGROUND=50 \
    python notebooks/06_lc_oot.py 2>&1 | tee logs/v4_smoke_oot.log
fi

echo "### 1/6 Primary fixed split: german credit -> taiwan -> home credit -> lending club ###"
DATASETS=south_german,taiwan,home_credit,lending_club \
  python notebooks/05_fixed_split_all.py 2>&1 | tee logs/v4_fixed_split.log

echo "### 2/6 Lending Club out-of-time (cutoff 2015) ###"
python notebooks/06_lc_oot.py 2>&1 | tee logs/v4_lc_oot.log

echo "### 3/6 Lending Club data audit ###"
if [[ ! -s results/data_audit_v4/audit_summary.json || "${AUDIT_FORCE:-0}" == "1" ]]; then
  python notebooks/07_lc_data_audit.py 2>&1 | tee logs/v4_lc_data_audit.log
else
  echo "Audit checkpoint found: results/data_audit_v4/audit_summary.json"
fi

echo "### 4/6 Learning curves (home credit, lending club) ###"
python notebooks/08_learning_curves.py 2>&1 | tee logs/v4_learning_curves.log

echo "### 5/6 Post-hoc analyses from persisted scores ###"
python notebooks/09_posthoc_from_scores.py 2>&1 | tee logs/v4_posthoc.log
python notebooks/10_ablation_protected_taiwan.py 2>&1 | tee logs/v4_ablation.log

echo "### 6/6 Output check ###"
for path in \
  results/fixed_split_v4/analysis_plan.json \
  results/fixed_split_v4/test_access_log.jsonl \
  results/fixed_split_v4/south_german/run_manifest.json \
  results/fixed_split_v4/south_german/final_metrics.csv \
  results/fixed_split_v4/south_german/economic_config.json \
  results/fixed_split_v4/taiwan/run_manifest.json \
  results/fixed_split_v4/taiwan/final_metrics.csv \
  results/fixed_split_v4/home_credit/run_manifest.json \
  results/fixed_split_v4/home_credit/final_metrics.csv \
  results/fixed_split_v4/lending_club/run_manifest.json \
  results/fixed_split_v4/lending_club/final_metrics.csv \
  results/fixed_split_v4/lending_club/bootstrap_ci.csv \
  results/oot_v4/lending_club/run_manifest.json \
  results/learning_curves_v4/home_credit/run_manifest.json \
  results/learning_curves_v4/lending_club/run_manifest.json \
  results/posthoc_v4/emp_roi_grid.csv \
  results/posthoc_v4/ablation_protected_taiwan.csv; do
  [[ -s "$path" ]] || { echo "FATAL: required output missing: $path"; exit 1; }
done
echo "V4 CHAIN COMPLETE. All results are under results/."
