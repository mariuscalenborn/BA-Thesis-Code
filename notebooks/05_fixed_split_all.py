"""Primary protocol: one fixed 80/20 split per dataset, all four models."""

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

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss

from src.bootstrap import (
    bootstrap_model_metrics,
    bootstrap_pairwise_deltas,
    paired_stratified_indices,
)
from src.datasets import DATASETS, ID_COLUMN, TARGET_COLUMN, get, validate_frame
from src.evaluation import (
    METRIC_KEYS,
    aggregate_to_source_features,
    case_bootstrap_stability,
    compute_attributions,
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
    reset_unseen_diagnostics,
    source_feature_map,
    unseen_category_diagnostics,
)
from src.run_utils import (
    build_manifest,
    code_fingerprint,
    load_stage_checkpoint,
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
    write_stage_checkpoint,
)
from src.search import DEFAULT_BUDGETS, cv_results_records, run_search
from src.seeds import SEEDS, environment_determinism, seed_everything
from src.splits import (
    assert_split_integrity,
    split_artifact_paths,
    write_or_validate_fixed_split,
)
from src.test_access import (
    ANALYSIS_PLAN_NAME,
    GuardedTestPartition,
    assert_no_fit,
    write_analysis_plan,
)
from src.thresholds import (
    base_rate_threshold,
    build_economic_config,
    emp_cutoff_from_oof,
    oof_raw_scores,
    write_economic_config,
)


warnings.filterwarnings("ignore")

seed_everything()

USE_GPU = os.environ.get("USE_GPU", "0") == "1"
EBM_BAGS = int(os.environ.get("EBM_BAGS", "8"))
N_MAX = int(os.environ.get("N_MAX", "0"))
FORCE = os.environ.get("FORCE", "0") == "1"
EXPLAIN_CASES = int(os.environ.get("EXPLAIN_CASES", "2000"))
EXPLAIN_BACKGROUND = int(os.environ.get("EXPLAIN_BACKGROUND", "500"))
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "1000"))
SEARCH_JOBS = int(os.environ.get("SEARCH_JOBS", "1"))

OUTBASE = Path(os.environ.get("OUTBASE", "results/fixed_split_v4"))
BUDGETS = {
    "LR": int(os.environ.get("LR_ITER", DEFAULT_BUDGETS["LR"])),
    "EBM": int(os.environ.get("EBM_ITER", DEFAULT_BUDGETS["EBM"])),
    "XGBoost": int(os.environ.get("XGB_ITER", DEFAULT_BUDGETS["XGBoost"])),
    "CatBoost": int(os.environ.get("CAT_ITER", DEFAULT_BUDGETS["CatBoost"])),
}

CODE_FILES = [
    Path(__file__).resolve(),
    *(REPO_ROOT / f"src/{module}.py" for module in (
        "bootstrap", "datasets", "evaluation", "models", "preprocessing",
        "run_utils", "search", "seeds", "splits", "test_access", "thresholds",
    )),
]
CODE_FINGERPRINT = code_fingerprint(CODE_FILES, repo_root=REPO_ROOT)


def _model_dir(output_dir: Path, name: str) -> Path:
    path = output_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


MODEL_ARTIFACTS = (
    "selection.json", "cv_results.json", "final_pipeline.joblib",
    "calibrated_pipeline.joblib", "thresholds.json", "oof_train_scores.npz",
)


def _model_artifacts_complete(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in MODEL_ARTIFACTS)


def _load_frame(config):
    frame = pd.read_csv(config.semiraw_path)
    validate_frame(config, frame)
    if N_MAX and len(frame) > N_MAX:
        from sklearn.model_selection import train_test_split

        frame, _ = train_test_split(
            frame,
            train_size=N_MAX,
            stratify=frame[TARGET_COLUMN],
            random_state=SEEDS["global"],
        )
        frame = frame.reset_index(drop=True)
        print(f"  SMOKE: subsampled to {len(frame)} rows")
    return frame


def _resolve_roi(config, frame, train_index):
    """ROI from the training partition only; never from the full frame."""
    if config.roi == "int_rate_train_mean":
        value = float(frame["int_rate"].iloc[train_index].mean() / 100.0)
        return value, "mean int_rate of the training partition / 100"
    return float(config.roi), "Verbraken et al. (2014) baseline"


def run_dataset(dataset: str) -> None:
    config = get(dataset)
    output_dir = OUTBASE / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "run_manifest.json").exists() and not FORCE:
        print(f"SKIP {dataset}: output already complete")
        return

    print(f"\n=== {config.display_name} ===")
    frame = _load_frame(config)
    feature_columns = config.feature_columns(frame.columns)
    X = frame[feature_columns]
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)
    source_ids = frame[ID_COLUMN].to_numpy(dtype=np.int64)

    manifest_path, indices_path = split_artifact_paths(output_dir, smoke=bool(N_MAX))
    split, split_audit = write_or_validate_fixed_split(
        manifest_path,
        indices_path,
        dataset=dataset,
        data_path=config.semiraw_path,
        y=y,
        source_ids=source_ids,
        columns=list(frame.columns),
    )
    assert_split_integrity(split, len(frame))
    print(f"  split           {split_audit['status']}: "
          f"{len(split.train_index)} train / {len(split.test_index)} test")

    X_train = X.iloc[split.train_index].reset_index(drop=True)
    y_train = y[split.train_index]
    X_test = X.iloc[split.test_index].reset_index(drop=True)
    y_test = y[split.test_index]

    roi, roi_provenance = _resolve_roi(config, frame, split.train_index)
    prior_default = float(np.mean(y_train == 1))
    fixed_threshold = base_rate_threshold(y_train)
    print(f"  roi             {roi:.4f} ({roi_provenance})")
    print(f"  train prior     {prior_default:.4f} (frozen for EMP)")

    write_analysis_plan(
        OUTBASE / ANALYSIS_PLAN_NAME,
        datasets=list(DATASETS),
        models=list(MODEL_NAMES),
    )

    selections, thresholds_by_model, prep_reports = {}, {}, {}
    fitted, calibrated_models = {}, {}
    timings = {}

    for name in MODEL_NAMES:
        model_dir = _model_dir(output_dir, name)
        template = build_pipeline(name, config, use_gpu=USE_GPU, ebm_bags=EBM_BAGS)
        started = time.time()

        cached = load_stage_checkpoint(
            model_dir / "model_stage.json", fingerprint=CODE_FINGERPRINT, force=FORCE
        )
        if cached is not None and _model_artifacts_complete(model_dir):
            selections[name] = cached["selection"]
            thresholds_by_model[name] = cached["thresholds"]
            timings[name] = cached["timings"]
            fitted[name] = joblib.load(model_dir / "final_pipeline.joblib")
            calibrated_models[name] = joblib.load(model_dir / "calibrated_pipeline.joblib")
            reset_unseen_diagnostics(fitted[name].named_steps["prep"])
            prep_reports[name] = preprocessing_report(fitted[name].named_steps["prep"])
            print(f"  {name:<9} resumed from checkpoint "
                  f"(mean val AUC {cached['selection']['mean_validation_auc']:.4f})")
            continue

        search, selection = run_search(
            name,
            template,
            X_train,
            y_train,
            split.hyperparameter_folds,
            budgets=BUDGETS,
            n_jobs=SEARCH_JOBS,
        )
        selection["code_fingerprint"] = CODE_FINGERPRINT
        selection["search_seconds"] = time.time() - started
        records = cv_results_records(search.cv_results_, len(split.hyperparameter_folds))
        write_json_atomic(model_dir / "cv_results.json", records)
        write_csv_atomic(model_dir / "cv_results.csv", pd.DataFrame(records))
        write_json_atomic(model_dir / "selection.json", selection)
        print(f"  {name:<9} search  mean val AUC {selection['mean_validation_auc']:.4f} "
              f"(tie-break: {selection['tie_break']['resolved_by']})")

        final_pipeline = clone(template).set_params(**selection["best_params"])
        refit_started = time.time()
        final_pipeline.fit(X_train, y_train)
        refit_seconds = time.time() - refit_started
        joblib.dump(final_pipeline, model_dir / "final_pipeline.joblib")

        calibrated = fit_calibrated(
            clone(template).set_params(**selection["best_params"]),
            X_train,
            y_train,
            cv=split.calibration_folds,
        )
        joblib.dump(calibrated, model_dir / "calibrated_pipeline.joblib")

        oof = oof_raw_scores(
            clone(template).set_params(**selection["best_params"]),
            X_train,
            y_train,
            split.threshold_folds,
        )
        emp_cutoff, eta_oof = emp_cutoff_from_oof(
            y_train, oof, roi, prior_default=prior_default
        )
        thresholds = {
            "model": name,
            "base_rate_threshold": fixed_threshold,
            "base_rate_source": "training partition default rate",
            "emp_cutoff": emp_cutoff,
            "eta_oof": eta_oof,
            "emp_cutoff_source": (
                "quantile 1-eta of five-fold cross-fitted raw training scores"
            ),
            "n_oof_predictions": int(len(oof)),
            "code_fingerprint": CODE_FINGERPRINT,
        }
        write_json_atomic(model_dir / "thresholds.json", thresholds)
        write_npz_atomic(
            model_dir / "oof_train_scores.npz",
            p_oof_raw=oof.astype(np.float32),
            y_train=y_train.astype(np.int8),
            train_source_ids=split.train_source_ids,
        )

        report = preprocessing_report(final_pipeline.named_steps["prep"])
        write_json_atomic(model_dir / "preprocessing_report.json", report)

        selections[name] = selection
        thresholds_by_model[name] = thresholds
        prep_reports[name] = report
        fitted[name] = final_pipeline
        calibrated_models[name] = calibrated
        timings[name] = {
            "search_seconds": selection["search_seconds"],
            "refit_seconds": refit_seconds,
            "total_seconds": time.time() - started,
        }
        write_stage_checkpoint(
            model_dir / "model_stage.json",
            {"model": name, "selection": selection, "thresholds": thresholds,
             "timings": timings[name]},
            fingerprint=CODE_FINGERPRINT,
        )
        print(f"  {name:<9} frozen  eta={eta_oof:.4f} cutoff={emp_cutoff:.4f} "
              f"base rate={fixed_threshold:.4f}")

    economic_config = build_economic_config(
        dataset=dataset,
        roi=roi,
        prior_default=prior_default,
        thresholds_by_model={
            name: {
                "emp_cutoff": thresholds_by_model[name]["emp_cutoff"],
                "eta_oof": thresholds_by_model[name]["eta_oof"],
                "base_rate_threshold": thresholds_by_model[name]["base_rate_threshold"],
            }
            for name in MODEL_NAMES
        },
        roi_provenance=roi_provenance,
    )
    economic_config_path = output_dir / "economic_config.json"
    write_economic_config(economic_config_path, economic_config)

    access = GuardedTestPartition(OUTBASE, repo_root=REPO_ROOT, script=Path(__file__).name)
    scores_by_model, rows = {}, []

    for name in MODEL_NAMES:
        model_dir = _model_dir(output_dir, name)
        assert_no_fit(fitted[name])
        assert_no_fit(calibrated_models[name])
        label_transforms(fitted[name].named_steps["prep"], "test")
        X_eval, y_eval = access.open(
            X, y, split.test_index,
            analysis="primary_model_comparison",
            dataset=dataset,
            model=name,
            artifact=model_dir / "final_pipeline.joblib",
            economic_config_path=economic_config_path,
        )
        p_raw = predict_positive_proba(fitted[name], X_eval)
        p_calibrated = predict_positive_proba(calibrated_models[name], X_eval)

        metrics = metrics_suite(
            y_eval,
            p_raw,
            p_calibrated,
            emp_cutoff=thresholds_by_model[name]["emp_cutoff"],
            base_rate_threshold=thresholds_by_model[name]["base_rate_threshold"],
            roi=roi,
            prior_default=prior_default,
        )
        write_npz_atomic(
            model_dir / "scores.npz",
            p_raw=p_raw.astype(np.float32),
            p_calibrated=p_calibrated.astype(np.float32),
            y_test=y_eval.astype(np.int8),
            test_source_ids=split.test_source_ids,
        )
        scores_by_model[name] = (p_raw, p_calibrated)
        rows.append({
            "dataset": dataset,
            "model": name,
            **{key: float(metrics[key]) for key in METRIC_KEYS},
            "test_default_rate": metrics["test_default_rate"],
            "prior_default_frozen": metrics["prior_default_frozen"],
            "mean_validation_auc": selections[name]["mean_validation_auc"],
            "std_validation_auc": selections[name]["std_validation_auc"],
            "n_encoded_features": prep_reports[name]["n_encoded_features"],
            **timings[name],
            "code_fingerprint": CODE_FINGERPRINT,
        })
        emp_raw = metrics["emp"]
        emp_percent = 100.0 * emp_raw
        print(f"  {name:<9} test    AUC={metrics['auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
              f"Brier={metrics['brier']:.4f} EMP={emp_percent:.2f}% of principal")

    final_metrics = pd.DataFrame(rows)
    write_csv_atomic(output_dir / "final_metrics.csv", final_metrics)

    for name in MODEL_NAMES:
        preprocessor = fitted[name].named_steps["prep"]
        report = preprocessing_report(preprocessor)
        prep_reports[name] = report
        write_json_atomic(_model_dir(output_dir, name) / "preprocessing_report.json", report)

    if config.sigmoid_sensitivity:
        _calibration_sensitivity(
            dataset, config, output_dir, access, X, y, X_train, y_train, split,
            selections, scores_by_model, economic_config_path,
        )

    explain_ids, attributions_by_model, source_features = _explain(
        dataset, config, output_dir, access, X, X_train, split, fitted,
        economic_config_path,
    )
    write_json_atomic(output_dir / "ohe_diagnostics.json", {
        name: unseen_category_diagnostics(fitted[name].named_steps["prep"])
        for name in MODEL_NAMES
    })

    indices = paired_stratified_indices(
        y_test, n_bootstrap=N_BOOTSTRAP, random_state=SEEDS["bootstrap"]
    )
    summary, draws = bootstrap_model_metrics(
        y_test,
        scores_by_model,
        thresholds_by_model,
        roi=roi,
        prior_default=prior_default,
        indices=indices,
    )
    write_csv_atomic(output_dir / "bootstrap_ci.csv", pd.DataFrame(summary))
    deltas = bootstrap_pairwise_deltas(draws, summary)
    write_csv_atomic(output_dir / "auc_delta_bootstrap.csv", pd.DataFrame(deltas))

    stability, per_model, cross_model = case_bootstrap_stability(
        attributions_by_model,
        source_features,
        n_bootstrap=N_BOOTSTRAP,
        random_state=SEEDS["explanation_sampling"],
    )
    write_json_atomic(output_dir / "stability_summary.json", stability)
    write_csv_atomic(output_dir / "shap_importance.csv", pd.DataFrame(per_model))
    write_csv_atomic(output_dir / "cross_model_spearman.csv", pd.DataFrame(cross_model))

    manifest = build_manifest(
        repo_root=REPO_ROOT,
        run_name=f"fixed_split_v4_{dataset}",
        input_files=[config.semiraw_path],
        code_files=CODE_FILES,
        feature_names=encoded_feature_names(fitted["LR"].named_steps["prep"]),
        sample_ids=[int(value) for value in explain_ids],
        settings={
            "protocol": "v4 fixed 80/20 split, training-only model selection",
            "split_audit": split_audit,
            "n_train": int(len(split.train_index)),
            "n_test": int(len(split.test_index)),
            "search": {name: selections[name]["search"] for name in MODEL_NAMES},
            "budgets": BUDGETS,
            "selected_params": {
                name: selections[name]["best_params"] for name in MODEL_NAMES
            },
            "tie_breaks": {
                name: selections[name]["tie_break"] for name in MODEL_NAMES
            },
            "calibration": {
                "method": "isotonic",
                "cv": "three materialised training folds",
                "ensemble": False,
                "sample_weight": None,
            },
            "thresholds": thresholds_by_model,
            "economic_config": economic_config,
            "explanation": {
                "n_cases": int(len(explain_ids)),
                "full_test_partition": bool(config.explain_full_test),
                "source": "uncalibrated pipeline fitted on the full training partition",
                "aggregation": "encoded levels summed back onto source features",
                "ebm_interactions": "split equally between participating features",
                "catboost_categoricals": (
                    "one-hot encoded like every other model; CatBoost's native "
                    "ordered target statistics are deliberately not used"
                ),
            },
            "bootstrap": {
                "n_bootstrap": N_BOOTSTRAP,
                "seed": SEEDS["bootstrap"],
                "paired": True,
                "stratified": True,
                "delong": "not implemented in v4",
            },
            "determinism": environment_determinism(),
            "ebm_bags": EBM_BAGS,
            "use_gpu": USE_GPU,
            "smoke_n_max": N_MAX,
        },
    )
    write_json_atomic(output_dir / "run_manifest.json", manifest)
    print(f"  written         {output_dir}")


def _calibration_sensitivity(
    dataset, config, output_dir, access, X, y, X_train, y_train, split,
    selections, scores_by_model, economic_config_path,
):
    """Prespecified sigmoid-versus-isotonic comparison (German Credit only)."""
    records = []
    for name in MODEL_NAMES:
        template = build_pipeline(name, config, use_gpu=USE_GPU, ebm_bags=EBM_BAGS)
        sigmoid = fit_calibrated(
            clone(template).set_params(**selections[name]["best_params"]),
            X_train,
            y_train,
            cv=split.calibration_folds,
            method="sigmoid",
        )
        X_eval, y_eval = access.open(
            X, y, split.test_index,
            analysis="calibration_sensitivity",
            dataset=dataset,
            model=name,
            economic_config_path=economic_config_path,
        )
        p_sigmoid = predict_positive_proba(sigmoid, X_eval)
        p_raw, p_isotonic = scores_by_model[name]
        records.append({
            "dataset": dataset,
            "model": name,
            "brier_raw": float(brier_score_loss(y_eval, p_raw)),
            "brier_isotonic": float(brier_score_loss(y_eval, p_isotonic)),
            "brier_sigmoid": float(brier_score_loss(y_eval, p_sigmoid)),
            "primary_method": "isotonic",
        })
    write_csv_atomic(
        output_dir / "calibration_sensitivity.csv", pd.DataFrame(records)
    )


def _explain(
    dataset, config, output_dir, access, X, X_train, split, fitted,
    economic_config_path,
):
    """Attributions for a shared set of test applicants, all models identical."""
    n_test = len(split.test_index)
    if config.explain_full_test or n_test <= EXPLAIN_CASES:
        positions = np.arange(n_test)
    else:
        rng = np.random.default_rng(SEEDS["explanation_sampling"])
        positions = np.sort(rng.choice(n_test, EXPLAIN_CASES, replace=False))
    explain_index = np.asarray(split.test_index)[positions]
    explain_ids = np.asarray(split.test_source_ids)[positions]

    X_explain_raw, _ = access.open(
        X, np.zeros(len(X), dtype=int), explain_index,
        analysis="explanation_comparison",
        dataset=dataset,
        economic_config_path=economic_config_path,
    )

    attributions_by_model, source_features = {}, None
    for name in MODEL_NAMES:
        pipeline = fitted[name]
        preprocessor = pipeline.named_steps["prep"]
        feature_names = encoded_feature_names(preprocessor)
        mapping = source_feature_map(preprocessor, config)

        label_transforms(preprocessor, "explanation_background_train")
        background = preprocessor.transform(X_train)
        label_transforms(preprocessor, "explanation_cases_test")
        explain_matrix = preprocessor.transform(X_explain_raw)
        values = compute_attributions(
            name,
            inner_estimator(pipeline),
            background,
            explain_matrix,
            feature_names,
            background_size=EXPLAIN_BACKGROUND,
            random_state=SEEDS["explanation_sampling"],
        )
        aggregated, sources = aggregate_to_source_features(
            np.abs(values), feature_names, mapping
        )
        if source_features is None:
            source_features = sources
        elif sources != source_features:
            raise ValueError(
                f"{name}: source feature set differs from the other models"
            )
        attributions_by_model[name] = aggregated.astype(np.float32)
        write_npz_atomic(
            _model_dir(output_dir, name) / "abs_attributions.npz",
            abs_attributions=aggregated.astype(np.float32),
            sample_ids=explain_ids,
            source_features=np.asarray(sources, dtype=object),
        )
    write_csv_atomic(
        output_dir / "explanation_sample_ids.csv",
        pd.DataFrame({"source_row_id": explain_ids}),
    )
    return explain_ids, attributions_by_model, source_features


def main() -> None:
    requested = os.environ.get("DATASETS")
    names = requested.split(",") if requested else list(DATASETS)
    OUTBASE.mkdir(parents=True, exist_ok=True)
    write_analysis_plan(
        OUTBASE / ANALYSIS_PLAN_NAME,
        datasets=list(DATASETS),
        models=list(MODEL_NAMES),
    )
    for dataset in names:
        run_dataset(dataset.strip())


if __name__ == "__main__":
    main()
