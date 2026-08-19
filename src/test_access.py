"""Governance for the shared test partition."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from src.run_utils import write_json_atomic


ANALYSIS_PLAN_NAME = "analysis_plan.json"
ACCESS_LOG_NAME = "test_access_log.jsonl"

PERMITTED_ANALYSES = {
    "primary_model_comparison": "Four-model comparison on the common test partition",
    "calibration_evaluation": "Brier score and raw-vs-calibrated comparison",
    "paired_bootstrap": "Paired stratified bootstrap intervals and pairwise deltas",
    "explanation_comparison": "Attributions and cross-model explanation agreement",
    "learning_curves": "Dimension 3 subsample curves scored on the same test partition",
    "fairness_probe": "Group-wise performance at the frozen base-rate threshold",
    "protected_feature_ablation": "Prespecified secondary analysis without protected attributes",
    "roi_sensitivity": "Economic sensitivity across the prespecified ROI grid",
    "calibration_sensitivity": "Prespecified sigmoid-versus-isotonic comparison",
    "out_of_time_robustness": "Separate temporal experiment on Lending Club",
}


class ProtocolAccessError(RuntimeError):
    """Raised when the test partition is touched outside the frozen plan."""


def write_analysis_plan(path, *, datasets, models) -> dict:
    """Freeze the list of permitted test-partition analyses."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    plan = {
        "schema": "ba_project_analysis_plan",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": sorted(datasets),
        "models": list(models),
        "permitted_analyses": dict(sorted(PERMITTED_ANALYSES.items())),
        "primary_analysis": "primary_model_comparison",
        "rules": [
            "The test partition is read only by analyses listed here.",
            "No test-partition result may change preprocessing, hyperparameters, "
            "calibration, thresholds, metric definitions or model inclusion.",
            "Analyses other than the primary comparison are prespecified secondary "
            "analyses and are reported as such.",
            "Every access is appended to test_access_log.jsonl.",
        ],
    }
    write_json_atomic(path, plan)
    return plan


def load_analysis_plan(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ProtocolAccessError(
            f"{path} is missing. The analysis plan must be frozen before the test "
            "partition is opened."
        )
    return json.loads(path.read_text())


def _git_commit(repo_root) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()


def _append_jsonl(path, record) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        if path.exists():
            handle.write(path.read_text())
        handle.write(line)
        temporary = Path(handle.name)
    os.replace(temporary, path)


class GuardedTestPartition:
    """Guarded entry point to the test partition."""

    def __init__(self, root, *, repo_root, script):
        self.root = Path(root)
        self.repo_root = Path(repo_root)
        self.script = script
        self.plan = load_analysis_plan(self.root / ANALYSIS_PLAN_NAME)
        self.log_path = self.root / ACCESS_LOG_NAME

    def open(
        self,
        X,
        y,
        test_index,
        *,
        analysis,
        dataset,
        model=None,
        artifact=None,
        economic_config_path=None,
    ):
        """Return the test rows, after checking and logging the access."""
        if analysis not in self.plan["permitted_analyses"]:
            raise ProtocolAccessError(
                f"Analysis {analysis!r} is not in the frozen analysis plan "
                f"({sorted(self.plan['permitted_analyses'])})."
            )
        if dataset not in self.plan["datasets"]:
            raise ProtocolAccessError(
                f"Dataset {dataset!r} is not covered by the frozen analysis plan."
            )
        if economic_config_path is not None and not Path(economic_config_path).exists():
            raise ProtocolAccessError(
                f"{economic_config_path} is missing; the economic configuration must "
                "be frozen before the test partition is read."
            )
        _append_jsonl(self.log_path, {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": str(self.script),
            "git_commit": _git_commit(self.repo_root),
            "dataset": dataset,
            "analysis": analysis,
            "model": model,
            "artifact": str(artifact) if artifact is not None else None,
            "n_test_rows": int(len(test_index)),
            "fit_occurred": False,
        })
        return _take(X, test_index), _take(y, test_index)


def assert_no_fit(estimator) -> None:
    """Guard for the final-evaluation stage: the model must already be fitted."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(estimator)
    except NotFittedError as error:
        raise ProtocolAccessError(
            "Final evaluation received an unfitted estimator; it must never fit."
        ) from error


def _take(data, index):
    if hasattr(data, "iloc"):
        return data.iloc[index]
    import numpy as np

    return np.asarray(data)[index]
