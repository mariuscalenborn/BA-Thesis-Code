"""Lending Club out-of-time evaluation on a single temporal split."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.datasets import ID_COLUMN, TARGET_COLUMN, get, validate_frame
from src.evaluation import (
    METRIC_KEYS,
    aggregate_to_source_features,
    compute_attributions,
    mean_abs_importance,
    metrics_suite,
)
from src.models import (
    MODEL_NAMES,
    build_pipeline,
    fit_calibrated,
    inner_estimator,
    predict_positive_proba,
)
from src.preprocessing import (
    encoded_feature_names,
    label_transforms,
    preprocessing_report,
    source_feature_map,
)
from src.run_utils import (
    build_manifest,
    code_fingerprint,
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
)
from src.search import DEFAULT_BUDGETS, cv_results_records, run_search
from src.seeds import SEEDS, environment_determinism, seed_everything
from src.thresholds import (
    base_rate_threshold,
    build_economic_config,
    emp_cutoff_from_oof,
    oof_raw_scores,
    write_economic_config,
)


warnings.filterwarnings("ignore")

seed_everything()

CUTOFF = int(os.environ.get("OOT_CUTOFF", "2015"))
USE_GPU = os.environ.get("USE_GPU", "0") == "1"
EBM_BAGS = int(os.environ.get("EBM_BAGS", "8"))
LC_N = int(os.environ.get("LC_N", "0"))
FORCE = os.environ.get("FORCE", "0") == "1"
EXPLAIN_CASES = int(os.environ.get("EXPLAIN_CASES", "2000"))
EXPLAIN_BACKGROUND = int(os.environ.get("EXPLAIN_BACKGROUND", "500"))
SEARCH_JOBS = int(os.environ.get("SEARCH_JOBS", "1"))
N_THRESHOLD_FOLDS = 5

OUTDIR = Path(os.environ.get("OUTDIR", "results/oot_v4/lending_club"))
CODE_FILES = [
    Path(__file__).resolve(),
    *(REPO_ROOT / f"src/{module}.py" for module in (
        "datasets", "evaluation", "models", "preprocessing", "run_utils",
        "search", "seeds", "thresholds",
    )),
]
CODE_FINGERPRINT = code_fingerprint(CODE_FILES, repo_root=REPO_ROOT)
BUDGETS = {
    "LR": int(os.environ.get("LR_ITER", DEFAULT_BUDGETS["LR"])),
    "EBM": int(os.environ.get("EBM_ITER", DEFAULT_BUDGETS["EBM"])),
    "XGBoost": int(os.environ.get("XGB_ITER", DEFAULT_BUDGETS["XGBoost"])),
    "CatBoost": int(os.environ.get("CAT_ITER", DEFAULT_BUDGETS["CatBoost"])),
}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if (OUTDIR / "run_manifest.json").exists() and not FORCE:
        print(f"SKIP lc_oot: output already complete: {OUTDIR}")
        return

    config = get("lending_club")
    frame = pd.read_csv(config.semiraw_path)
    validate_frame(config, frame)
    if LC_N > 0 and len(frame) > LC_N:
        frame, _ = train_test_split(
            frame,
            train_size=LC_N,
            stratify=frame[TARGET_COLUMN],
            random_state=SEEDS["global"],
        )
        frame = frame.reset_index(drop=True)
        print(f"Smoke subsample: LC_N={LC_N}")

    feature_columns = config.feature_columns(frame.columns)
    X = frame[feature_columns]
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)
    years = frame["issue_year"].to_numpy(dtype=int)
    source_ids = frame[ID_COLUMN].to_numpy(dtype=np.int64)

    fit_mask = years <= CUTOFF - 1
    validation_mask = years == CUTOFF
    train_mask = years <= CUTOFF
    test_mask = years > CUTOFF
    if not all(mask.any() for mask in (fit_mask, validation_mask, train_mask, test_mask)):
        raise RuntimeError(f"Empty out-of-time partition for cutoff {CUTOFF}")

    train_positions = np.flatnonzero(train_mask)
    test_positions = np.flatnonzero(test_mask)
    X_train = X.iloc[train_positions].reset_index(drop=True)
    y_train = y[train_positions]
    X_test = X.iloc[test_positions].reset_index(drop=True)
    y_test = y[test_positions]
    test_years = years[test_positions]
    test_ids = source_ids[test_positions]

    train_years = years[train_positions]
    search_folds = [(
        np.flatnonzero(train_years <= CUTOFF - 1),
        np.flatnonzero(train_years == CUTOFF),
    )]

    roi = float(frame["int_rate"].to_numpy()[train_positions].mean()) / 100.0
    prior_default = float(np.mean(y_train == 1))
    fixed_threshold = base_rate_threshold(y_train)

    rng = np.random.default_rng(SEEDS["explanation_sampling"])
    if len(test_positions) <= EXPLAIN_CASES:
        explain_local = np.arange(len(test_positions))
    else:
        explain_local = np.sort(rng.choice(len(test_positions), EXPLAIN_CASES, replace=False))
    explain_ids = test_ids[explain_local]
    X_explain = X_test.iloc[explain_local]

    print(
        f"LC OOT: cutoff={CUTOFF} fit={int(fit_mask.sum())} "
        f"validation={int(validation_mask.sum())} train={len(y_train)} "
        f"test={len(y_test)} default_train={prior_default:.4f} "
        f"default_test={y_test.mean():.4f} ROI={roi:.4f}"
    )

    threshold_folds = list(
        StratifiedKFold(
            n_splits=N_THRESHOLD_FOLDS,
            shuffle=True,
            random_state=SEEDS["threshold_oof_cv"],
        ).split(np.zeros(len(y_train), dtype=np.uint8), y_train)
    )

    rows, year_rows, importance_rows = [], [], []
    importances_by_model, thresholds_by_model = {}, {}
    source_features = None

    for name in MODEL_NAMES:
        model_dir = OUTDIR / name
        model_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        template = build_pipeline(name, config, use_gpu=USE_GPU, ebm_bags=EBM_BAGS)

        search, selection = run_search(
            name, template, X_train, y_train, search_folds,
            budgets=BUDGETS, n_jobs=SEARCH_JOBS,
        )
        selection["code_fingerprint"] = CODE_FINGERPRINT
        selection["validation_design"] = f"temporal, issue_year == {CUTOFF}"
        write_json_atomic(model_dir / "selection.json", selection)
        write_csv_atomic(
            model_dir / "cv_results.csv",
            pd.DataFrame(cv_results_records(search.cv_results_, len(search_folds))),
        )

        final_pipeline = clone(template).set_params(**selection["best_params"])
        final_pipeline.fit(X_train, y_train)
        joblib.dump(final_pipeline, model_dir / "final_pipeline.joblib")

        calibrated = fit_calibrated(
            clone(template).set_params(**selection["best_params"]),
            X_train,
            y_train,
            cv=StratifiedKFold(
                n_splits=3, shuffle=True, random_state=SEEDS["calibration_cv"]
            ),
        )
        joblib.dump(calibrated, model_dir / "calibrated_pipeline.joblib")

        oof = oof_raw_scores(
            clone(template).set_params(**selection["best_params"]),
            X_train,
            y_train,
            threshold_folds,
        )
        emp_cutoff, eta_oof = emp_cutoff_from_oof(
            y_train, oof, roi, prior_default=prior_default
        )
        thresholds_by_model[name] = {
            "emp_cutoff": emp_cutoff,
            "eta_oof": eta_oof,
            "base_rate_threshold": fixed_threshold,
        }
        write_json_atomic(model_dir / "thresholds.json", {
            "model": name,
            **thresholds_by_model[name],
            "emp_cutoff_source": (
                "quantile 1-eta of five-fold cross-fitted raw scores inside the "
                f"training window (issue_year <= {CUTOFF})"
            ),
            "code_fingerprint": CODE_FINGERPRINT,
        })

        label_transforms(final_pipeline.named_steps["prep"], "oot_test")
        p_raw = predict_positive_proba(final_pipeline, X_test)
        p_calibrated = predict_positive_proba(calibrated, X_test)
        metrics = metrics_suite(
            y_test, p_raw, p_calibrated,
            emp_cutoff=emp_cutoff,
            base_rate_threshold=fixed_threshold,
            roi=roi,
            prior_default=prior_default,
        )
        write_npz_atomic(
            model_dir / "scores.npz",
            p_raw=p_raw.astype(np.float32),
            p_calibrated=p_calibrated.astype(np.float32),
            y_test=y_test.astype(np.int8),
            test_years=test_years.astype(np.int16),
            test_source_ids=test_ids,
        )

        row = {
            "model": name,
            "cutoff": CUTOFF,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "train_base_rate": fixed_threshold,
            "default_rate_test": float(y_test.mean()),
            "roi": roi,
            "emp_cutoff": emp_cutoff,
            "eta_oof": eta_oof,
            "validation_auc": selection["mean_validation_auc"],
            "code_fingerprint": CODE_FINGERPRINT,
            "elapsed_seconds": time.time() - started,
            **{key: float(metrics[key]) for key in METRIC_KEYS},
        }
        rows.append(row)
        emp_raw = row["emp"]
        emp_percent = 100.0 * emp_raw
        print(
            f"  {name:<8} AUC={row['auc']:.4f} PR={row['pr_auc']:.4f} "
            f"Brier={row['brier']:.4f} EMP={emp_percent:.2f}% of principal "
            f"valAUC={row['validation_auc']:.4f} | {row['elapsed_seconds']:.0f}s"
        )

        for year in sorted(np.unique(test_years)):
            mask = test_years == year
            year_metrics = metrics_suite(
                y_test[mask], p_raw[mask], p_calibrated[mask],
                emp_cutoff=emp_cutoff,
                base_rate_threshold=fixed_threshold,
                roi=roi,
                prior_default=prior_default,
            )
            year_rows.append({
                "model": name,
                "test_year": int(year),
                "n_test": int(mask.sum()),
                "default_rate_test": float(y_test[mask].mean()),
                **{key: float(year_metrics[key]) for key in METRIC_KEYS},
            })

        preprocessor = final_pipeline.named_steps["prep"]
        feature_names = encoded_feature_names(preprocessor)
        mapping = source_feature_map(preprocessor, config)
        label_transforms(preprocessor, "oot_explanation_background")
        background = preprocessor.transform(X_train)
        label_transforms(preprocessor, "oot_explanation_cases")
        values = compute_attributions(
            name, inner_estimator(final_pipeline), background,
            preprocessor.transform(X_explain), feature_names,
            background_size=EXPLAIN_BACKGROUND,
            random_state=SEEDS["explanation_sampling"],
        )
        aggregated, sources = aggregate_to_source_features(
            np.abs(values), feature_names, mapping
        )
        if source_features is None:
            source_features = sources
        write_npz_atomic(
            model_dir / "abs_attributions.npz",
            abs_attributions=aggregated.astype(np.float32),
            sample_ids=explain_ids,
            source_features=np.asarray(sources, dtype=object),
        )
        write_json_atomic(
            model_dir / "preprocessing_report.json", preprocessing_report(preprocessor)
        )

        importance = mean_abs_importance(aggregated)
        importances_by_model[name] = importance
        order = np.argsort(-importance)
        for rank, index in enumerate(order, start=1):
            importance_rows.append({
                "model": name,
                "feature": sources[index],
                "mean_abs_attribution": float(importance[index]),
                "rank": rank,
            })

    write_economic_config(OUTDIR / "economic_config.json", build_economic_config(
        dataset="lending_club_oot",
        roi=roi,
        prior_default=prior_default,
        thresholds_by_model=thresholds_by_model,
        roi_provenance=f"mean int_rate of the training window (issue_year <= {CUTOFF}) / 100",
    ))
    write_csv_atomic(OUTDIR / "final_metrics.csv", pd.DataFrame(rows))
    write_csv_atomic(OUTDIR / "per_year.csv", pd.DataFrame(year_rows))
    write_csv_atomic(OUTDIR / "shap_importance.csv", pd.DataFrame(importance_rows))
    write_csv_atomic(
        OUTDIR / "explanation_sample_ids.csv",
        pd.DataFrame({"source_row_id": [int(value) for value in explain_ids]}),
    )

    spearman_rows = []
    for left_index, left in enumerate(MODEL_NAMES):
        for right in MODEL_NAMES[left_index + 1:]:
            spearman_rows.append({
                "model_a": left,
                "model_b": right,
                "spearman": float(spearmanr(
                    importances_by_model[left], importances_by_model[right]
                ).statistic),
            })
    write_csv_atomic(OUTDIR / "cross_model_spearman.csv", pd.DataFrame(spearman_rows))

    manifest = build_manifest(
        repo_root=REPO_ROOT,
        run_name="oot_v4_lending_club",
        input_files=[config.semiraw_path],
        code_files=CODE_FILES,
        feature_names=source_features or [],
        sample_ids={"pooled_test": [int(value) for value in explain_ids]},
        settings={
            "design": "single_out_of_time_split",
            "role": "separate temporal robustness experiment, not the headline",
            "cutoff_year": CUTOFF,
            "fit_years": f"<= {CUTOFF - 1}",
            "validation_year": CUTOFF,
            "train_years": f"<= {CUTOFF}",
            "test_years": f"> {CUTOFF}",
            "search": {name: BUDGETS[name] for name in MODEL_NAMES},
            "boosting_rounds": "selected by the search like any other hyperparameter",
            "calibration": {
                "method": "isotonic", "cv": 3, "ensemble": False, "sample_weight": None,
            },
            "threshold_source": "five-fold cross-fitted scores inside the training window",
            "roi": roi,
            "prior_default_train": prior_default,
            "ebm_bags": EBM_BAGS,
            "explanation_cases": int(len(explain_ids)),
            "determinism": environment_determinism(),
            "code_fingerprint": CODE_FINGERPRINT,
            "interpretation": "exploratory_outcome_maturity_caveat",
        },
    )
    write_json_atomic(OUTDIR / "run_manifest.json", manifest)
    print(f"\nLC OOT v4 complete: {OUTDIR}")


if __name__ == "__main__":
    main()
