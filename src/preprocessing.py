"""Leakage-safe preprocessing pipeline, fitted on the current fit subset only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.datasets import DatasetConfig, WINSOR_QUANTILE
from src.seeds import QUANTILE_METHOD


UNKNOWN_LEVEL = "Unknown"


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Drop columns whose missing share in the fit subset exceeds ``threshold``."""

    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    def fit(self, X, y=None):
        frame = _as_frame(X)
        missing_share = frame.isnull().mean()
        self.columns_in_ = list(frame.columns)
        self.missing_share_ = {
            column: float(share) for column, share in missing_share.items()
        }
        self.columns_to_keep_ = [
            column for column in frame.columns if missing_share[column] <= self.threshold
        ]
        self.columns_dropped_ = [
            column for column in frame.columns if column not in self.columns_to_keep_
        ]
        return self

    def transform(self, X):
        frame = _as_frame(X)
        missing = [c for c in self.columns_to_keep_ if c not in frame.columns]
        if missing:
            raise ValueError(f"Transform input lacks fitted columns: {missing}")
        return frame[self.columns_to_keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_to_keep_, dtype=object)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip named columns at the fit subset's upper quantile."""

    def __init__(self, columns=(), quantile: float = WINSOR_QUANTILE):
        self.columns = columns
        self.quantile = quantile

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.columns_in_ = list(frame.columns)
        self.thresholds_ = {}
        for column in self.columns:
            if column not in frame.columns:
                continue
            values = frame[column].to_numpy(dtype=float)
            if np.all(np.isnan(values)):
                continue
            self.thresholds_[column] = float(
                np.nanquantile(values, self.quantile, method=QUANTILE_METHOD)
            )
        return self

    def transform(self, X):
        frame = _as_frame(X).copy()
        for column, threshold in self.thresholds_.items():
            frame[column] = frame[column].clip(upper=threshold)
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_in_, dtype=object)


class UnknownFiller(BaseEstimator, TransformerMixin):
    """Replace missing categorical values with the literal ``Unknown`` level."""

    def fit(self, X, y=None):
        self.columns_in_ = list(_as_frame(X).columns)
        return self

    def transform(self, X):
        frame = _as_frame(X).astype(object)
        return frame.where(frame.notna(), UNKNOWN_LEVEL)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_in_, dtype=object)


class SafeOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encoder with a training-only schema and unseen-level diagnostics."""

    def __init__(self, transform_label=None):
        self.transform_label = transform_label

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.columns_in_ = list(frame.columns)
        categories = []
        for column in frame.columns:
            observed = pd.unique(frame[column].dropna().astype(str))
            levels = sorted(set(observed.tolist()) | {UNKNOWN_LEVEL})
            categories.append(np.asarray(levels, dtype=object))
        self.categories_ = categories
        self.encoder_ = OneHotEncoder(
            categories=categories,
            drop=None,
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        ).fit(frame.astype(str))
        self.feature_names_out_ = list(
            self.encoder_.get_feature_names_out(self.columns_in_)
        )
        self.unseen_diagnostics_ = []
        return self

    def transform(self, X):
        frame = _as_frame(X)
        self._record_unseen(frame)
        return self.encoder_.transform(frame.astype(str))

    def _record_unseen(self, frame):
        report = {}
        affected = np.zeros(len(frame), dtype=bool)
        for position, column in enumerate(self.columns_in_):
            known = set(self.categories_[position].tolist())
            values = frame[column].astype(str)
            mask = ~values.isin(known)
            count = int(mask.sum())
            if count:
                affected |= mask.to_numpy()
                report[column] = {
                    "n_unseen_rows": count,
                    "share_unseen_rows": float(count / max(len(frame), 1)),
                    "unseen_levels": sorted(set(values[mask].tolist()))[:20],
                    "n_unseen_levels": int(values[mask].nunique()),
                }
        self.unseen_diagnostics_.append({
            "partition": self.transform_label or "unlabelled",
            "n_rows": int(len(frame)),
            "n_rows_with_unseen_level": int(affected.sum()),
            "share_rows_with_unseen_level": float(affected.sum() / max(len(frame), 1)),
            "by_feature": report,
        })

    def category_schema(self):
        return {
            column: [str(level) for level in self.categories_[position]]
            for position, column in enumerate(self.columns_in_)
        }

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


class RoleSelector:
    """Picklable column selector that intersects a declared role with what remains."""

    def __init__(self, role: str, config: DatasetConfig):
        self.role = role
        self.config = config

    def __call__(self, X):
        columns = list(_as_frame(X).columns)
        if self.role == "categorical":
            declared = list(self.config.categorical_cols)
        elif self.role == "binary":
            declared = list(self.config.binary_numeric_cols)
        elif self.role == "numeric":
            declared = self.config.numeric_columns(columns)
        else:
            raise ValueError(f"Unknown column role {self.role!r}")
        return [column for column in declared if column in columns]

    def __repr__(self):
        return f"RoleSelector({self.role!r}, {self.config.name!r})"

    def __eq__(self, other):
        return (
            isinstance(other, RoleSelector)
            and self.role == other.role
            and self.config.name == other.config.name
        )

    def __hash__(self):
        return hash((self.role, self.config.name))


def build_preprocessor(config: DatasetConfig) -> Pipeline:
    """Assemble the fit-subset-local preprocessing pipeline for one dataset."""
    numeric = Pipeline([
        ("winsor", Winsorizer(columns=tuple(config.winsor_cols))),
        ("impute", SimpleImputer(strategy="median")),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    categorical = Pipeline([
        ("unknown", UnknownFiller()),
        ("encode", SafeOneHotEncoder(transform_label="train")),
    ])
    transformer = ColumnTransformer(
        [
            ("num", numeric, RoleSelector("numeric", config)),
            ("bin", binary, RoleSelector("binary", config)),
            ("cat", categorical, RoleSelector("categorical", config)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("missing", MissingnessFilter(threshold=config.missing_threshold)),
        ("columns", transformer),
    ])


def label_transforms(preprocessor: Pipeline, label: str) -> None:
    """Tag every subsequent ``transform`` of this pipeline with a partition name."""
    encoder = _fitted_encoder(preprocessor)
    if encoder is not None:
        encoder.transform_label = label


def unseen_category_diagnostics(preprocessor: Pipeline) -> list[dict]:
    encoder = _fitted_encoder(preprocessor)
    return list(encoder.unseen_diagnostics_) if encoder is not None else []


def reset_unseen_diagnostics(preprocessor: Pipeline) -> None:
    """Clear accumulated transform diagnostics on a pipeline loaded from disk."""
    encoder = _fitted_encoder(preprocessor)
    if encoder is not None:
        encoder.unseen_diagnostics_ = []


def _fitted_encoder(preprocessor: Pipeline):
    transformer = preprocessor.named_steps["columns"]
    for name, fitted, columns in transformer.transformers_:
        if name == "cat" and len(columns):
            return fitted.named_steps["encode"]
    return None


def encoded_feature_names(preprocessor: Pipeline) -> list[str]:
    return [str(name) for name in preprocessor.get_feature_names_out()]


def source_feature_map(preprocessor: Pipeline, config: DatasetConfig) -> dict[str, str]:
    """Map every encoded column back to the source feature it came from."""
    transformer = preprocessor.named_steps["columns"]
    mapping: dict[str, str] = {}
    for name, fitted, columns in transformer.transformers_:
        if name == "cat" and len(columns):
            encoder = fitted.named_steps["encode"]
            for encoded in encoder.get_feature_names_out():
                encoded = str(encoded)
                origin = next(
                    (
                        source
                        for source in encoder.columns_in_
                        if encoded == source or encoded.startswith(f"{source}_")
                    ),
                    encoded,
                )
                mapping[encoded] = origin
        else:
            for column in columns if isinstance(columns, list) else []:
                mapping[str(column)] = str(column)
    for encoded in encoded_feature_names(preprocessor):
        mapping.setdefault(encoded, encoded)
    return mapping


def preprocessing_report(preprocessor: Pipeline) -> dict:
    """Fitted-parameter snapshot for the run manifest and leakage audits."""
    missing_step = preprocessor.named_steps["missing"]
    transformer = preprocessor.named_steps["columns"]
    report = {
        "missingness_filter": {
            "threshold": float(missing_step.threshold),
            "n_columns_in": len(missing_step.columns_in_),
            "n_columns_kept": len(missing_step.columns_to_keep_),
            "columns_dropped": list(missing_step.columns_dropped_),
        },
        "winsorization": {},
        "imputation_median": {},
        "imputation_most_frequent": {},
        "category_schema": {},
        "unseen_categories": [],
        "n_encoded_features": len(encoded_feature_names(preprocessor)),
    }
    for name, fitted, columns in transformer.transformers_:
        if name == "num":
            report["winsorization"] = dict(fitted.named_steps["winsor"].thresholds_)
            imputer = fitted.named_steps["impute"]
            report["imputation_median"] = {
                str(column): float(value)
                for column, value in zip(columns, imputer.statistics_)
            }
        elif name == "bin" and len(columns):
            imputer = fitted.named_steps["impute"]
            report["imputation_most_frequent"] = {
                str(column): float(value)
                for column, value in zip(columns, imputer.statistics_)
            }
        elif name == "cat" and len(columns):
            encoder = fitted.named_steps["encode"]
            report["category_schema"] = encoder.category_schema()
            report["unseen_categories"] = list(encoder.unseen_diagnostics_)
    return report


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError(
        "The v4 preprocessing pipeline operates on named columns; pass a DataFrame."
    )
