"""Learning curves on Home Credit and Lending Club at frozen configurations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.datasets import TARGET_COLUMN, get, validate_frame
from src.evaluation import (
    aggregate_to_source_features,
    compute_attributions,
    mean_abs_importance,
    topk_jaccard,
)
from src.models import (
    MODEL_NAMES,
    build_pipeline,
    inner_estimator,
    predict_positive_proba,
)
from src.preprocessing import (
    encoded_feature_names,
    label_transforms,
    source_feature_map,
)
from src.run_utils import (
    build_manifest,
    code_fingerprint,
    load_stage_checkpoint,
    write_csv_atomic,
    write_json_atomic,
    write_stage_checkpoint,
)
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
N_REPS = int(os.environ.get("LC_REPS", "3"))
FORCE = os.environ.get("FORCE", "0") == "1"
EXPLAIN_CASES = int(os.environ.get("EXPLAIN_CASES", "1000"))
EXPLAIN_BACKGROUND = int(os.environ.get("EXPLAIN_BACKGROUND", "500"))
SIZES = [int(size) for size in os.environ.get(
    "LC_SIZES", "10000,50000,100000,250000"
).split(",")]

RESULTS_BASE = Path(os.environ.get("RESULTS_BASE", "results/fixed_split_v4"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
OUTBASE = Path(os.environ.get("OUTBASE", "results/learning_curves_v4"))
CODE_FILES = [
    Path(__file__).resolve(),
    *(REPO_ROOT / f"src/{module}.py" for module in (
        "datasets", "evaluation", "models", "preprocessing", "run_utils",
        "seeds", "splits", "test_access",
    )),
]
CODE_FINGERPRINT = code_fingerprint(CODE_FILES, repo_root=REPO_ROOT)

FROZEN_PARAMS = {
    "LR": {"clf__lr__C": 1.0},
    "EBM": {},
    "XGBoost": {
        "clf__estimator__n_estimators": 300,
        "clf__estimator__max_depth": 6,
        "clf__estimator__learning_rate": 0.1,
    },
    "CatBoost": {
        "clf__estimator__iterations": 400,
        "clf__estimator__depth": 6,
        "clf__estimator__learning_rate": 0.1,
    },
}

CURVE_DATASETS = ("home_credit", "lending_club")


def run_dataset(dataset):
    output_dir = OUTBASE / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "run_manifest.json").exists() and not FORCE:
        print(f"SKIP {dataset}: learning curves already complete")
        return

    config = get(dataset)
    frame = pd.read_csv(config.semiraw_path)
    validate_frame(config, frame)
    X = frame[config.feature_columns(frame.columns)]
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)

    manifest_path, indices_path = split_artifact_paths(RESULTS_BASE / dataset, smoke=SMOKE)
    if not indices_path.exists():
        raise RuntimeError(
            f"{indices_path} is missing. The learning curves reuse the main "
            "protocol's split; run 05_fixed_split_all.py first."
        )
    assert_manifest_covers_frame(json.loads(manifest_path.read_text()), len(frame))
    split = load_fixed_split(indices_path)
    assert_split_integrity(split, len(frame))

    access = GuardedTestPartition(
        RESULTS_BASE, repo_root=REPO_ROOT, script=Path(__file__).name
    )
    economic_config_path = RESULTS_BASE / dataset / "economic_config.json"

    pool_index = np.asarray(split.train_index)
    y_pool = y[pool_index]
    sizes = [size for size in SIZES if size <= len(pool_index)]
    print(f"\n=== {config.display_name} ===")
    print(f"  training pool {len(pool_index)} | test {len(split.test_index)} "
          f"| sizes {sizes}")

    rng = np.random.default_rng(SEEDS["explanation_sampling"])
    n_test = len(split.test_index)
    explain_local = (
        np.arange(n_test) if n_test <= EXPLAIN_CASES
        else np.sort(rng.choice(n_test, EXPLAIN_CASES, replace=False))
    )
    explain_index = np.asarray(split.test_index)[explain_local]

    rows = []
    for size in sizes:
        for rep in range(N_REPS):
            subsample_index, _ = train_test_split(
                pool_index,
                train_size=size,
                stratify=y_pool,
                random_state=SEEDS["learning_curve"] + rep,
            )
            X_sub = X.iloc[subsample_index]
            y_sub = y[subsample_index]

            for name in MODEL_NAMES:
                checkpoint = output_dir / "checkpoints" / f"n{size}_rep{rep}_{name}.json"
                cached = load_stage_checkpoint(
                    checkpoint, fingerprint=CODE_FINGERPRINT, force=FORCE
                )
                if cached is not None:
                    rows.append(cached["row"])
                    continue

                started = time.time()
                pipeline = build_pipeline(
                    name, config, use_gpu=USE_GPU, ebm_bags=EBM_BAGS
                ).set_params(**FROZEN_PARAMS[name])
                pipeline.fit(X_sub, y_sub)

                assert_no_fit(pipeline)
                label_transforms(pipeline.named_steps["prep"], "learning_curve_test")
                X_eval, y_eval = access.open(
                    X, y, split.test_index,
                    analysis="learning_curves",
                    dataset=dataset,
                    model=name,
                    economic_config_path=economic_config_path,
                )
                auc = float(roc_auc_score(
                    y_eval, predict_positive_proba(pipeline, X_eval)
                ))

                preprocessor = pipeline.named_steps["prep"]
                feature_names = encoded_feature_names(preprocessor)
                mapping = source_feature_map(preprocessor, config)
                label_transforms(preprocessor, "learning_curve_explanation")
                values = compute_attributions(
                    name, inner_estimator(pipeline),
                    preprocessor.transform(X_sub),
                    preprocessor.transform(X.iloc[explain_index]),
                    feature_names,
                    background_size=EXPLAIN_BACKGROUND,
                    random_state=SEEDS["explanation_sampling"],
                )
                aggregated, sources = aggregate_to_source_features(
                    np.abs(values), feature_names, mapping
                )
                importance = mean_abs_importance(aggregated)

                row = {
                    "dataset": dataset,
                    "model": name,
                    "n_train": int(size),
                    "rep": int(rep),
                    "auc": auc,
                    "n_encoded_features": len(feature_names),
                    "top10": [sources[i] for i in np.argsort(-importance)[:10]],
                    "elapsed_seconds": time.time() - started,
                }
                write_stage_checkpoint(
                    checkpoint,
                    {"row": row, "importance": importance.tolist(),
                     "source_features": sources},
                    fingerprint=CODE_FINGERPRINT,
                )
                rows.append(row)
                print(f"  n={size:<7} rep={rep} {name:<9} AUC={auc:.4f} "
                      f"({row['elapsed_seconds']:.0f}s)")

    raw = pd.DataFrame(rows)
    write_csv_atomic(
        output_dir / "learning_curve_raw.csv",
        raw.assign(top10=raw["top10"].apply(lambda names: "|".join(names))),
    )

    curve_rows = []
    for (size, name), group in raw.groupby(["n_train", "model"]):
        rankings = [set(entry) for entry in group["top10"]]
        indexed = [
            np.asarray([1.0 if feature in ranking else 0.0
                        for feature in sorted(set().union(*rankings))])
            for ranking in rankings
        ]
        curve_rows.append({
            "dataset": dataset,
            "model": name,
            "n_train": int(size),
            "auc_mean": float(group["auc"].mean()),
            "auc_std": float(group["auc"].std(ddof=1)) if len(group) > 1 else float("nan"),
            "n_reps": int(len(group)),
            "top10_jaccard_across_reps": topk_jaccard(indexed, k=10)
            if len(indexed) > 1 else float("nan"),
        })
    write_csv_atomic(output_dir / "learning_curve.csv", pd.DataFrame(curve_rows))

    manifest = build_manifest(
        repo_root=REPO_ROOT,
        run_name=f"learning_curves_v4_{dataset}",
        input_files=[config.semiraw_path],
        code_files=CODE_FILES,
        feature_names=[],
        sample_ids=[int(value) for value in np.asarray(split.test_source_ids)[explain_local]],
        settings={
            "design": "subsamples of the main protocol's training partition",
            "test_partition": "identical to 05_fixed_split_all.py",
            "sizes": sizes,
            "n_reps": N_REPS,
            "subsample_seed_rule": "SEEDS['learning_curve'] + rep",
            "frozen_params": FROZEN_PARAMS,
            "interpretation": "per-model slopes only; levels are not comparable",
            "determinism": environment_determinism(),
            "code_fingerprint": CODE_FINGERPRINT,
        },
    )
    write_json_atomic(output_dir / "run_manifest.json", manifest)
    print(f"  written       {output_dir}")


def main():
    if not (RESULTS_BASE / ANALYSIS_PLAN_NAME).exists():
        raise RuntimeError(
            f"{RESULTS_BASE / ANALYSIS_PLAN_NAME} is missing; run "
            "05_fixed_split_all.py before any secondary test-partition analysis."
        )
    OUTBASE.mkdir(parents=True, exist_ok=True)
    requested = os.environ.get("DATASETS")
    names = requested.split(",") if requested else list(CURVE_DATASETS)
    for dataset in names:
        run_dataset(dataset.strip())


if __name__ == "__main__":
    main()
