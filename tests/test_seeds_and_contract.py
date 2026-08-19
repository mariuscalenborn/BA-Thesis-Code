"""Seed discipline and the semi-raw data contract."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest

from src.datasets import DATASETS, ID_COLUMN, TARGET_COLUMN, get, validate_frame
from src.seeds import SEEDS, model_seed
from src.semiraw import audit_semiraw_export


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SOURCES = sorted(
    [path for path in (REPO_ROOT / "src").glob("*.py") if path.name != "seeds.py"]
    + list((REPO_ROOT / "notebooks").glob("*.py"))
)

SEED_LITERAL = re.compile(r"(random_state|random_seed|seed)\s*=\s*(\d+)")


def test_the_registry_covers_every_stage_and_model():
    for key in (
        "global", "train_test_split", "hyperparameter_cv", "randomized_search",
        "calibration_cv", "threshold_oof_cv", "bootstrap", "explanation_sampling",
    ):
        assert key in SEEDS
    for name in ("LR", "EBM", "XGBoost", "CatBoost"):
        assert model_seed(name) in SEEDS.values()


def test_no_module_defines_its_own_seed_literal():
    """A stray literal would silently detach one operation from the registry."""
    offenders = []
    for path in PRODUCTION_SOURCES:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or "SEEDS[" in line or "model_seed(" in line:
                continue
            match = SEED_LITERAL.search(line)
            if match:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {stripped}")
    assert not offenders, "Seed literals outside src/seeds.py:\n" + "\n".join(offenders)


def test_early_stopping_lives_only_in_wrapper_and_search():
    """Early stopping is confined to the search path."""
    tokens = ("early_stopping", "eval_set", "best_iteration", "use_best_model")
    offenders = []
    for path in PRODUCTION_SOURCES:
        if path.name in ("models.py", "search.py"):
            continue
        text = path.read_text()
        for token in tokens:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert not offenders, "Early stopping outside wrapper/search:\n" + "\n".join(offenders)

    from src.search import PARAM_GRIDS

    assert "clf__estimator__n_estimators" not in PARAM_GRIDS["XGBoost"]
    assert "clf__estimator__iterations" not in PARAM_GRIDS["CatBoost"]


def test_no_production_source_still_speaks_of_outer_folds():
    forbidden = ("outer_fold", "outer fold", "outer training fold", "inner holdout",
                 "nested cross-validation", "64/16/20")
    offenders = []
    for path in PRODUCTION_SOURCES:
        text = path.read_text().lower()
        for token in forbidden:
            if token.lower() in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert not offenders, "Outer-CV terminology remnants:\n" + "\n".join(offenders)


def test_no_runner_reads_a_v3_results_directory():
    offenders = []
    for path in PRODUCTION_SOURCES:
        text = path.read_text()
        for token in ("holdout_v3", "oot_v3", "learning_curves_v3", "posthoc_v3",
                      "data_audit_v3"):
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert not offenders, "v3 result paths still referenced:\n" + "\n".join(offenders)


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_every_dataset_declares_a_usable_contract(dataset):
    config = get(dataset)
    assert config.display_name
    assert config.semiraw_path.as_posix().startswith("data/processed/v4/")
    assert isinstance(config.roi, (float, str))
    assert ID_COLUMN not in config.categorical_cols
    assert TARGET_COLUMN not in config.aux_cols


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_the_exported_file_matches_its_contract(dataset):
    config = get(dataset)
    if not config.semiraw_path.exists():
        pytest.skip(f"{config.semiraw_path} not generated in this checkout")
    expect_missing = dataset in ("home_credit", "lending_club")
    report = audit_semiraw_export(
        config.semiraw_path, dataset, expect_missing=expect_missing
    )
    assert report["n_features"] > 0
    assert 0.0 < report["default_rate"] < 1.0


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_the_export_carries_no_one_hot_columns(dataset):
    config = get(dataset)
    if not config.semiraw_path.exists():
        pytest.skip(f"{config.semiraw_path} not generated in this checkout")
    columns = pd.read_csv(config.semiraw_path, nrows=0).columns.tolist()
    for source in config.categorical_cols:
        assert source in columns
        assert not [c for c in columns if c != source and c.startswith(f"{source}_")]


def test_lending_club_keeps_its_auxiliary_columns_out_of_the_features():
    config = get("lending_club")
    if not config.semiraw_path.exists():
        pytest.skip("lending club export not generated in this checkout")
    columns = pd.read_csv(config.semiraw_path, nrows=0).columns.tolist()
    features = config.feature_columns(columns)
    for column in ("int_rate", "issue_year"):
        assert column in columns
        assert column not in features


def test_lending_club_dropped_the_endogenous_pricing_columns():
    from src.datasets import LENDING_CLUB_ENDOGENOUS_DROPS

    config = get("lending_club")
    if not config.semiraw_path.exists():
        pytest.skip("lending club export not generated in this checkout")
    columns = pd.read_csv(config.semiraw_path, nrows=0).columns.tolist()
    assert not [c for c in LENDING_CLUB_ENDOGENOUS_DROPS if c in columns]


def test_contract_validation_rejects_duplicate_identifiers(synthetic_config, synthetic_frame):
    broken = synthetic_frame.copy()
    broken.loc[1, ID_COLUMN] = broken.loc[0, ID_COLUMN]
    with pytest.raises(ValueError, match="not unique"):
        validate_frame(synthetic_config, broken)


def test_contract_validation_rejects_an_encoded_categorical(synthetic_config, synthetic_frame):
    broken = synthetic_frame.copy()
    broken["colour"] = 1
    with pytest.raises(ValueError, match="numeric in the semi-raw CSV"):
        validate_frame(
            synthetic_config.__class__(
                **{**synthetic_config.__dict__, "_expected_string_cols": ("colour",)}
            ),
            broken,
        )
