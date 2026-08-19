"""Post-hoc analyses computed from persisted test scores, without retraining."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.datasets import DATASETS, ID_COLUMN, get
from src.evaluation import emp_credit_scoring
from src.models import MODEL_NAMES
from src.run_utils import sha256_file, write_csv_atomic
from src.splits import split_artifact_paths


warnings.filterwarnings("ignore")

RESULTS_BASE = Path(os.environ.get("RESULTS_BASE", "results/fixed_split_v4"))
OUTDIR = Path(os.environ.get("OUTDIR", "results/posthoc_v4"))
ROI_GRID = [round(0.1 + 0.05 * index, 2) for index in range(9)]
SMOKE = os.environ.get("SMOKE", "0") == "1"

FAIRNESS_DATASETS = ("taiwan", "home_credit")


def _dataset_dir(dataset):
    return RESULTS_BASE / dataset


def _available(dataset):
    return (_dataset_dir(dataset) / "final_metrics.csv").exists()


def _load_scores(dataset, model):
    with np.load(_dataset_dir(dataset) / model / "scores.npz") as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def _economic_config(dataset):
    return json.loads((_dataset_dir(dataset) / "economic_config.json").read_text())


def brier_comparison():
    rows = []
    for dataset in DATASETS:
        if not _available(dataset):
            print(f"SKIP Brier comparison {dataset}: no v4 results")
            continue
        for model in MODEL_NAMES:
            scores = _load_scores(dataset, model)
            y = scores["y_test"].astype(int)
            brier_raw = float(brier_score_loss(y, scores["p_raw"]))
            brier_calibrated = float(brier_score_loss(y, scores["p_calibrated"]))
            rows.append({
                "dataset": dataset,
                "model": model,
                "brier_raw": brier_raw,
                "brier_calibrated": brier_calibrated,
                "brier_raw_minus_calibrated": brier_raw - brier_calibrated,
            })
        print(f"Brier comparison done: {dataset}")
    if rows:
        write_csv_atomic(OUTDIR / "brier_comparison.csv", pd.DataFrame(rows))


def emp_sensitivity():
    rows = []
    for dataset in DATASETS:
        if not _available(dataset):
            print(f"SKIP emp sensitivity {dataset}: no v4 results")
            continue
        config = _economic_config(dataset)
        prior_default = float(config["prior_default_train"])
        for model in MODEL_NAMES:
            scores = _load_scores(dataset, model)
            for roi in ROI_GRID:
                emp, eta = emp_credit_scoring(
                    scores["y_test"], scores["p_raw"], roi,
                    prior_default=prior_default,
                )
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "roi": roi,
                    "emp": emp,
                    "eta": eta,
                    "prior_default_frozen": prior_default,
                    "baseline_roi": float(config["roi"]),
                })
        print(f"emp sensitivity done: {dataset}")
    if rows:
        write_csv_atomic(OUTDIR / "emp_roi_grid.csv", pd.DataFrame(rows))


def _group_lookup(dataset):
    """Group labels keyed by ``source_row_id``, with a data-identity check."""
    config = get(dataset)
    manifest_path, _ = split_artifact_paths(_dataset_dir(dataset), smoke=SMOKE)
    manifest = json.loads(manifest_path.read_text())
    recorded = manifest["data_snapshot"]["sha256"]
    actual = sha256_file(config.semiraw_path)
    if recorded != actual:
        raise RuntimeError(
            f"{config.semiraw_path} no longer matches the snapshot the scores were "
            f"produced from ({actual} != {recorded}). Re-run script 05."
        )
    frame = pd.read_csv(config.semiraw_path, usecols=[ID_COLUMN, config.fairness_col])
    if frame[ID_COLUMN].duplicated().any():
        raise RuntimeError(f"{dataset}: {ID_COLUMN} is not unique")
    return frame.set_index(ID_COLUMN)[config.fairness_col]


def fairness_probe():
    for dataset in FAIRNESS_DATASETS:
        if not _available(dataset):
            print(f"SKIP fairness {dataset}: no v4 results")
            continue
        config = get(dataset)
        lookup = _group_lookup(dataset)
        base_rate = float(
            _economic_config(dataset)["thresholds"][MODEL_NAMES[0]]["base_rate_threshold"]
        )

        rows = []
        for model in MODEL_NAMES:
            scores = _load_scores(dataset, model)
            ids = scores["test_source_ids"]
            missing = np.setdiff1d(ids, lookup.index.to_numpy())
            if missing.size:
                raise RuntimeError(
                    f"{dataset}: {missing.size} test IDs have no source row"
                )
            group_values = lookup.reindex(ids).to_numpy()
            y = scores["y_test"].astype(int)
            p = scores["p_calibrated"]
            yhat = (p > base_rate).astype(int)
            for label in pd.unique(pd.Series(group_values).dropna()):
                mask = group_values == label
                if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                    continue
                positives = y[mask] == 1
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "group_column": config.fairness_col,
                    "group": str(label),
                    "n": int(mask.sum()),
                    "auc": float(roc_auc_score(y[mask], p[mask])),
                    "selection_rate": float(yhat[mask].mean()),
                    "tpr": float(yhat[mask][positives].mean()),
                    "base_rate_threshold": base_rate,
                })
        frame = pd.DataFrame(rows)
        write_csv_atomic(OUTDIR / f"fairness_{dataset}.csv", frame)

        summary_rows = []
        for model in MODEL_NAMES:
            model_rows = frame[frame["model"] == model].sort_values("n", ascending=False)
            if len(model_rows) < 2:
                continue
            first, second = model_rows.iloc[0], model_rows.iloc[1]
            summary_rows.append({
                "dataset": dataset,
                "model": model,
                "group_a": first["group"],
                "group_b": second["group"],
                "auc_gap": float(first["auc"] - second["auc"]),
                "demographic_parity_diff": float(
                    first["selection_rate"] - second["selection_rate"]
                ),
                "equal_opportunity_diff": float(first["tpr"] - second["tpr"]),
            })
        write_csv_atomic(
            OUTDIR / f"fairness_{dataset}_summary.csv", pd.DataFrame(summary_rows)
        )
        print(f"fairness probe done: {dataset}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    brier_comparison()
    emp_sensitivity()
    fairness_probe()
    print("\nPost-hoc analyses complete")


if __name__ == "__main__":
    main()
