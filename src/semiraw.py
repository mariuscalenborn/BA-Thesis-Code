"""Audit of the semi-raw exports produced by notebooks 01-04."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets import DatasetConfig, ID_COLUMN, TARGET_COLUMN, get, validate_frame
from src.run_utils import sha256_file


def audit_semiraw_export(path, dataset: str, *, expect_missing: bool | None = None) -> dict:
    """Verify one exported CSV against its declared contract and print a report."""
    path = Path(path)
    config = get(dataset)
    frame = pd.read_csv(path)
    validate_frame(config, frame)

    features = config.feature_columns(frame.columns)
    missing_by_column = frame[features].isnull().mean()
    n_missing = int(frame[features].isnull().sum().sum())
    above_threshold = sorted(
        missing_by_column[missing_by_column > config.missing_threshold].index.tolist()
    )

    if expect_missing is True and n_missing == 0:
        raise AssertionError(
            f"{dataset}: the semi-raw export has no missing feature values. Median "
            "imputation must happen inside the fitted pipeline, not at export time."
        )
    if expect_missing is False and n_missing != 0:
        raise AssertionError(
            f"{dataset}: expected a complete export but found {n_missing} missing values."
        )

    onehot_suspects = [
        column
        for column in features
        if column not in config.categorical_cols
        and any(column.startswith(f"{source}_") for source in config.categorical_cols)
    ]
    if onehot_suspects:
        raise AssertionError(
            f"{dataset}: columns look one-hot encoded at export time: {onehot_suspects[:10]}. "
            "The encoder must run inside the pipeline."
        )

    report = {
        "dataset": dataset,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "n_features": len(features),
        "n_categorical": len(config.categorical_cols),
        "n_aux": len(config.aux_cols),
        "default_rate": float(frame[TARGET_COLUMN].mean()),
        "n_missing_feature_cells": n_missing,
        "columns_above_missing_threshold": above_threshold,
        "winsor_column_maxima": {
            column: float(np.nanmax(frame[column].to_numpy(dtype=float)))
            for column in config.winsor_cols
            if column in frame.columns
        },
    }

    print(f"Semi-raw audit — {config.display_name}")
    print(f"  file            {report['path']}")
    print(f"  sha256          {report['sha256']}")
    print(f"  rows x columns  {report['n_rows']} x {report['n_columns']}")
    print(f"  features        {report['n_features']} "
          f"({report['n_categorical']} categorical, {report['n_aux']} auxiliary)")
    print(f"  default rate    {report['default_rate']:.4f}")
    print(f"  {ID_COLUMN}   unique, complete")
    print(f"  missing cells   {report['n_missing_feature_cells']} (kept for the fitted imputer)")
    print(f"  >{config.missing_threshold:.0%} missing     {len(above_threshold)} columns retained; "
          "the runtime filter decides per fit subset")
    if report["winsor_column_maxima"]:
        print("  winsor columns  unclipped maxima "
              f"{ {k: round(v, 1) for k, v in report['winsor_column_maxima'].items()} }")
    return report


def assert_deterministic_rerun(path, expected_sha256: str) -> None:
    """Confirm a re-executed notebook reproduced the byte-identical export."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise AssertionError(
            f"{path}: export is not reproducible ({actual} != {expected_sha256}). "
            "Some operation depends on row order or an unseeded sample."
        )


def contract_summary(config: DatasetConfig) -> str:
    return (
        f"{config.display_name}: {len(config.categorical_cols)} raw categorical columns, "
        f"{len(config.aux_cols)} auxiliary columns, "
        f"winsorized at runtime: {list(config.winsor_cols) or 'none'}"
    )
