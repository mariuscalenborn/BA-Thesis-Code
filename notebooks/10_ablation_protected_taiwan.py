"""Protected-attribute ablation on Taiwan at the selected configuration."""

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
from sklearn.metrics import roc_auc_score

from src.bootstrap import paired_stratified_indices
from src.datasets import TARGET_COLUMN, get, validate_frame
from src.models import MODEL_NAMES, build_pipeline, predict_positive_proba
from src.preprocessing import label_transforms
from src.run_utils import build_manifest, sha256_file, write_csv_atomic, write_json_atomic
from src.seeds import SEEDS, environment_determinism, seed_everything
from src.splits import (
    assert_manifest_covers_frame,
    assert_split_integrity,
    load_fixed_split,
    split_artifact_paths,
)
from src.test_access import ANALYSIS_PLAN_NAME, GuardedTestPartition, assert_no_fit


warnings.filterwarnings("ignore")

seed_everything()

USE_GPU = os.environ.get("USE_GPU", "0") == "1"
EBM_BAGS = int(os.environ.get("EBM_BAGS", "8"))
FORCE = os.environ.get("FORCE", "0") == "1"
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "1000"))

RESULTS_BASE = Path(os.environ.get("RESULTS_BASE", "results/fixed_split_v4"))
OUTDIR = Path(os.environ.get("OUTDIR", "results/posthoc_v4"))
DATASET = "taiwan"
SMOKE = os.environ.get("SMOKE", "0") == "1"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / "ablation_protected_taiwan.csv"
    if out_path.exists() and not FORCE:
        print(f"SKIP ablation: {out_path} exists")
        return

    dataset_dir = RESULTS_BASE / DATASET
    metrics_path = dataset_dir / "final_metrics.csv"
    if not metrics_path.exists():
        raise RuntimeError(f"Primary run output missing: {metrics_path}")
    if not (RESULTS_BASE / ANALYSIS_PLAN_NAME).exists():
        raise RuntimeError("The analysis plan must exist before this secondary analysis")
    primary = pd.read_csv(metrics_path)

    config = get(DATASET)
    frame = pd.read_csv(config.semiraw_path)
    validate_frame(config, frame)

    manifest_path, indices_path = split_artifact_paths(dataset_dir, smoke=SMOKE)
    manifest = json.loads(manifest_path.read_text())
    if manifest["data_snapshot"]["sha256"] != sha256_file(config.semiraw_path):
        raise RuntimeError(
            "The semi-raw file no longer matches the snapshot the primary run used"
        )

    assert_manifest_covers_frame(manifest, len(frame))
    split = load_fixed_split(indices_path)
    assert_split_integrity(split, len(frame))

    protected = list(config.protected_cols)
    missing = [column for column in protected if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Protected columns absent: {missing}")

    full_features = config.feature_columns(frame.columns)
    reduced_features = [c for c in full_features if c not in protected]
    X_reduced = frame[reduced_features]
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)

    X_train = X_reduced.iloc[split.train_index].reset_index(drop=True)
    y_train = y[split.train_index]
    y_test = y[split.test_index]

    access = GuardedTestPartition(
        RESULTS_BASE, repo_root=REPO_ROOT, script=Path(__file__).name
    )
    economic_config_path = dataset_dir / "economic_config.json"
    indices = paired_stratified_indices(
        y_test, n_bootstrap=N_BOOTSTRAP, random_state=SEEDS["bootstrap"]
    )

    rows = []
    for name in MODEL_NAMES:
        selection = json.loads((dataset_dir / name / "selection.json").read_text())
        params = selection["best_params"]

        pipeline = build_pipeline(
            name, config, use_gpu=USE_GPU, ebm_bags=EBM_BAGS
        ).set_params(**params)
        pipeline.fit(X_train, y_train)
        assert_no_fit(pipeline)
        label_transforms(pipeline.named_steps["prep"], "ablation_test")

        X_eval, y_eval = access.open(
            X_reduced, y, split.test_index,
            analysis="protected_feature_ablation",
            dataset=DATASET,
            model=name,
            economic_config_path=economic_config_path,
        )
        p_reduced = predict_positive_proba(pipeline, X_eval)

        with np.load(dataset_dir / name / "scores.npz") as arrays:
            p_full = arrays["p_raw"].copy()

        auc_full = float(primary.loc[primary["model"] == name, "auc"].iloc[0])
        auc_reduced = float(roc_auc_score(y_eval, p_reduced))

        deltas = np.empty(len(indices))
        for replicate, index in enumerate(indices):
            deltas[replicate] = (
                roc_auc_score(y_eval[index], p_reduced[index])
                - roc_auc_score(y_eval[index], p_full[index])
            )
        rows.append({
            "dataset": DATASET,
            "model": name,
            "protected_removed": ",".join(protected),
            "auc_full": auc_full,
            "auc_reduced": auc_reduced,
            "delta_auc": auc_reduced - auc_full,
            "delta_ci_low": float(np.quantile(deltas, 0.025)),
            "delta_ci_high": float(np.quantile(deltas, 0.975)),
            "share_delta_positive": float(np.mean(deltas > 0)),
            "n_features_full": len(full_features),
            "n_features_reduced": len(reduced_features),
            "hyperparameters": "frozen from the primary run",
        })
        print(f"  {name:<9} full={auc_full:.4f} reduced={auc_reduced:.4f} "
              f"delta={auc_reduced - auc_full:+.4f} "
              f"[{rows[-1]['delta_ci_low']:+.4f}, {rows[-1]['delta_ci_high']:+.4f}]")

    write_csv_atomic(out_path, pd.DataFrame(rows))
    write_json_atomic(OUTDIR / "ablation_protected_taiwan_manifest.json", build_manifest(
        repo_root=REPO_ROOT,
        run_name="ablation_protected_taiwan_v4",
        input_files=[config.semiraw_path],
        code_files=[Path(__file__).resolve()],
        feature_names=reduced_features,
        sample_ids=[],
        settings={
            "role": "prespecified secondary analysis",
            "protected_removed": protected,
            "hyperparameters": "frozen from the primary run's selection.json",
            "uncertainty": "paired bootstrap on the fixed test partition",
            "n_bootstrap": N_BOOTSTRAP,
            "determinism": environment_determinism(),
        },
    ))
    print(f"\nAblation complete: {out_path}")


if __name__ == "__main__":
    main()
