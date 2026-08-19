"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import DatasetConfig


N_ROWS = 400


@pytest.fixture
def synthetic_config(tmp_path):
    return DatasetConfig(
        name="synthetic",
        display_name="Synthetic",
        semiraw_path=tmp_path / "synthetic.csv",
        roi=0.2644,
        aux_cols=("aux_value",),
        categorical_cols=("colour", "region"),
        binary_numeric_cols=("has_phone",),
        winsor_cols=("income",),
        protected_cols=("age",),
        fairness_col="colour",
        missing_threshold=0.40,
    )


@pytest.fixture
def synthetic_frame():
    rng = np.random.default_rng(12345)
    income = rng.lognormal(mean=10.0, sigma=0.6, size=N_ROWS)
    income[:5] = 5_000_000.0

    age = rng.integers(20, 75, size=N_ROWS).astype(float)
    age[rng.choice(N_ROWS, 30, replace=False)] = np.nan

    has_phone = rng.integers(0, 2, size=N_ROWS).astype(float)
    has_phone[rng.choice(N_ROWS, 20, replace=False)] = np.nan

    colour = np.asarray(["red", "green", "blue"])[rng.integers(0, 3, size=N_ROWS)]
    colour = colour.astype(object)
    colour[rng.choice(N_ROWS, 25, replace=False)] = None
    colour[-3:] = "chartreuse"

    region = np.asarray(["north", "south"])[rng.integers(0, 2, size=N_ROWS)].astype(object)

    mostly_missing = np.full(N_ROWS, np.nan)
    mostly_missing[: int(0.35 * N_ROWS)] = rng.normal(size=int(0.35 * N_ROWS))

    signal = (
        0.4 * (np.log(income) - np.log(income).mean()) / np.log(income).std()
        + 0.6 * (colour == "red").astype(float)
    )
    probability = 1.0 / (1.0 + np.exp(-(signal - 0.8)))
    target = (rng.uniform(size=N_ROWS) < probability).astype(int)

    return pd.DataFrame({
        "source_row_id": np.arange(N_ROWS, dtype=np.int64),
        "income": income,
        "age": age,
        "has_phone": has_phone,
        "colour": colour,
        "region": region,
        "mostly_missing": mostly_missing,
        "aux_value": rng.normal(size=N_ROWS),
        "target": target,
    })


@pytest.fixture
def synthetic_dataset(synthetic_config, synthetic_frame):
    synthetic_frame.to_csv(synthetic_config.semiraw_path, index=False)
    return synthetic_config, synthetic_frame


@pytest.fixture
def synthetic_xy(synthetic_config, synthetic_frame):
    features = synthetic_config.feature_columns(synthetic_frame.columns)
    return (
        synthetic_frame[features],
        synthetic_frame["target"].to_numpy(dtype=int),
    )
