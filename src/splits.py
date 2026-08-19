"""Prospective split manifests: partitions and folds pinned before the first fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.run_utils import sha256_file, write_json_atomic, write_npz_atomic
from src.seeds import SEEDS


SCHEMA_NAME = "ba_project_fixed_split_manifest"
SCHEMA_VERSION = 1
ID_HASH_ENCODING = "sorted unique signed int64, little-endian, raw C-order bytes"
ORDERED_INT_HASH_ENCODING = "ordered signed int64, little-endian, raw C-order bytes"

TEST_SIZE = 0.20
N_HYPERPARAMETER_FOLDS = 5
N_CALIBRATION_FOLDS = 3
N_THRESHOLD_FOLDS = 5


class SplitIdentityError(RuntimeError):
    """Raised before model fitting when a split identity check fails."""


def split_artifact_paths(output_dir, *, smoke=False):
    """Manifest and index paths for one dataset directory."""
    output_dir = Path(output_dir)
    suffix = "_smoke" if smoke else ""
    return (
        output_dir / f"split_manifest{suffix}.json",
        output_dir / f"split_indices{suffix}.npz",
    )


def _int64_bytes(values, *, sort_values):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional integer array")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("Split hashes require integer values")
    normalized = np.asarray(array, dtype="<i8")
    if sort_values:
        normalized = np.sort(normalized)
    return np.ascontiguousarray(normalized).tobytes(order="C")


def hash_source_ids(values):
    """Hash partition membership independently of row order."""
    array = np.asarray(values)
    if len(np.unique(array)) != len(array):
        raise ValueError("Source row IDs must be unique")
    return hashlib.sha256(_int64_bytes(array, sort_values=True)).hexdigest()


def hash_ordered_ints(values):
    """Hash integer values while preserving row order."""
    return hashlib.sha256(_int64_bytes(values, sort_values=False)).hexdigest()


def hash_columns(columns):
    encoded = json.dumps(list(columns), ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _target_counts(y):
    labels, counts = np.unique(y, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _default_rate(y):
    y = np.asarray(y)
    return float(np.mean(y == 1)) if len(y) else float("nan")


def make_fixed_split(y, *, test_size=TEST_SIZE, random_state=None):
    """The one stratified train/test split of the whole dataset."""
    y = np.asarray(y)
    random_state = SEEDS["train_test_split"] if random_state is None else random_state
    train_index, test_index = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        stratify=y,
        shuffle=True,
        random_state=random_state,
    )
    return np.sort(train_index), np.sort(test_index)


def make_training_folds(y_train, *, n_splits, random_state):
    """Materialise fold indices local to the training partition."""
    y_train = np.asarray(y_train)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    placeholder = np.zeros(len(y_train), dtype=np.uint8)
    return [
        (np.asarray(train_idx, dtype=int), np.asarray(validation_idx, dtype=int))
        for train_idx, validation_idx in splitter.split(placeholder, y_train)
    ]


def build_fold_plan(y_train):
    """All three training-internal fold families, in one place."""
    return {
        "hyperparameter_cv": make_training_folds(
            y_train,
            n_splits=N_HYPERPARAMETER_FOLDS,
            random_state=SEEDS["hyperparameter_cv"],
        ),
        "calibration_cv": make_training_folds(
            y_train,
            n_splits=N_CALIBRATION_FOLDS,
            random_state=SEEDS["calibration_cv"],
        ),
        "threshold_oof_cv": make_training_folds(
            y_train,
            n_splits=N_THRESHOLD_FOLDS,
            random_state=SEEDS["threshold_oof_cv"],
        ),
    }


def _fold_records(folds, y_train, train_source_ids):
    records = []
    for fold, (train_idx, validation_idx) in enumerate(folds, start=1):
        records.append({
            "fold": fold,
            "n_fit": int(len(train_idx)),
            "n_validation": int(len(validation_idx)),
            "fit_source_ids_sha256": hash_source_ids(train_source_ids[train_idx]),
            "validation_source_ids_sha256": hash_source_ids(train_source_ids[validation_idx]),
            "fit_target_counts": _target_counts(y_train[train_idx]),
            "validation_target_counts": _target_counts(y_train[validation_idx]),
        })
    return records


def build_fixed_split_manifest(
    *,
    dataset,
    data_path,
    y,
    source_ids,
    columns,
    test_size=TEST_SIZE,
    random_state=None,
):
    """Build the deterministic manifest for one semi-raw data snapshot."""
    data_path = Path(data_path)
    y = np.asarray(y)
    source_ids = np.asarray(source_ids)
    if len(y) != len(source_ids):
        raise ValueError("Target and source row IDs have different lengths")
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError("Source row IDs must be unique")

    random_state = SEEDS["train_test_split"] if random_state is None else random_state
    train_index, test_index = make_fixed_split(
        y, test_size=test_size, random_state=random_state
    )
    y_train, y_test = y[train_index], y[test_index]
    train_source_ids = source_ids[train_index]
    folds = build_fold_plan(y_train)

    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "provenance": "written_prospectively_before_first_fit",
        "data_snapshot": {
            "path": data_path.as_posix(),
            "sha256": sha256_file(data_path),
            "n_rows": int(len(y)),
            "n_columns": int(len(columns)),
            "columns_sha256": hash_columns(columns),
            "source_row_ids_sha256": hash_ordered_ints(source_ids),
            "target_sha256": hash_ordered_ints(y),
            "target_counts": _target_counts(y),
            "default_rate": _default_rate(y),
        },
        "splitter": {
            "class": "sklearn.model_selection.train_test_split",
            "test_size": float(test_size),
            "stratify": "target",
            "shuffle": True,
            "random_state": int(random_state),
            "membership_hash_encoding": ID_HASH_ENCODING,
            "ordered_integer_hash_encoding": ORDERED_INT_HASH_ENCODING,
        },
        "partitions": {
            "train": {
                "n": int(len(train_index)),
                "source_ids_sha256": hash_source_ids(source_ids[train_index]),
                "target_counts": _target_counts(y_train),
                "default_rate": _default_rate(y_train),
            },
            "test": {
                "n": int(len(test_index)),
                "source_ids_sha256": hash_source_ids(source_ids[test_index]),
                "target_counts": _target_counts(y_test),
                "default_rate": _default_rate(y_test),
            },
        },
        "training_folds": {
            "hyperparameter_cv": {
                "class": "sklearn.model_selection.StratifiedKFold",
                "n_splits": N_HYPERPARAMETER_FOLDS,
                "shuffle": True,
                "random_state": SEEDS["hyperparameter_cv"],
                "folds": _fold_records(folds["hyperparameter_cv"], y_train, train_source_ids),
            },
            "calibration_cv": {
                "class": "sklearn.model_selection.StratifiedKFold",
                "n_splits": N_CALIBRATION_FOLDS,
                "shuffle": True,
                "random_state": SEEDS["calibration_cv"],
                "folds": _fold_records(folds["calibration_cv"], y_train, train_source_ids),
            },
            "threshold_oof_cv": {
                "class": "sklearn.model_selection.StratifiedKFold",
                "n_splits": N_THRESHOLD_FOLDS,
                "shuffle": True,
                "random_state": SEEDS["threshold_oof_cv"],
                "folds": _fold_records(folds["threshold_oof_cv"], y_train, train_source_ids),
            },
        },
        "seeds": dict(SEEDS),
    }


def _indices_payload(train_index, test_index, source_ids, folds):
    payload = {
        "train_index": np.asarray(train_index, dtype=np.int64),
        "test_index": np.asarray(test_index, dtype=np.int64),
        "train_source_ids": np.asarray(source_ids[train_index], dtype=np.int64),
        "test_source_ids": np.asarray(source_ids[test_index], dtype=np.int64),
    }
    for family, fold_list in folds.items():
        for fold, (fit_idx, validation_idx) in enumerate(fold_list, start=1):
            payload[f"{family}_fold{fold}_fit"] = np.asarray(fit_idx, dtype=np.int64)
            payload[f"{family}_fold{fold}_validation"] = np.asarray(
                validation_idx, dtype=np.int64
            )
    return payload


class FixedSplit:
    """Loaded partitions and fold definitions for one dataset."""

    def __init__(self, train_index, test_index, train_source_ids, test_source_ids, folds):
        self.train_index = train_index
        self.test_index = test_index
        self.train_source_ids = train_source_ids
        self.test_source_ids = test_source_ids
        self.folds = folds

    @property
    def hyperparameter_folds(self):
        return self.folds["hyperparameter_cv"]

    @property
    def calibration_folds(self):
        return self.folds["calibration_cv"]

    @property
    def threshold_folds(self):
        return self.folds["threshold_oof_cv"]


def write_or_validate_fixed_split(
    manifest_path,
    indices_path,
    *,
    dataset,
    data_path,
    y,
    source_ids,
    columns,
    test_size=TEST_SIZE,
    random_state=None,
):
    """Write the split on the first run; validate against it on every re-run."""
    manifest_path = Path(manifest_path)
    indices_path = Path(indices_path)
    y = np.asarray(y)
    source_ids = np.asarray(source_ids)

    actual = build_fixed_split_manifest(
        dataset=dataset,
        data_path=data_path,
        y=y,
        source_ids=source_ids,
        columns=columns,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, test_index = make_fixed_split(
        y,
        test_size=test_size,
        random_state=SEEDS["train_test_split"] if random_state is None else random_state,
    )
    folds = build_fold_plan(y[train_index])
    split = FixedSplit(
        train_index=train_index,
        test_index=test_index,
        train_source_ids=source_ids[train_index],
        test_source_ids=source_ids[test_index],
        folds=folds,
    )

    if not manifest_path.exists():
        write_npz_atomic(indices_path, **_indices_payload(train_index, test_index, source_ids, folds))
        actual["indices_artifact"] = {
            "path": indices_path.as_posix(),
            "sha256": sha256_file(indices_path),
        }
        write_json_atomic(manifest_path, actual)
        audit = {
            "status": "written_prospectively",
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "indices_sha256": actual["indices_artifact"]["sha256"],
            "data_sha256": actual["data_snapshot"]["sha256"],
            "membership_hashes_verified": True,
        }
        return split, audit

    try:
        frozen = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise SplitIdentityError(
            f"Cannot read split manifest {manifest_path}: {error}"
        ) from error

    checked = ("schema", "schema_version", "dataset", "data_snapshot", "splitter",
               "partitions", "training_folds", "seeds")
    mismatches = [key for key in checked if frozen.get(key) != actual.get(key)]
    if mismatches:
        raise SplitIdentityError(
            f"Split identity check failed for {dataset}: {', '.join(mismatches)} differ. "
            "No model was fitted. FORCE does not bypass this check; inspect the "
            "semi-raw data, the seed registry, and the manifest."
        )

    if not indices_path.exists():
        raise SplitIdentityError(
            f"Split manifest {manifest_path} exists but {indices_path} is missing. "
            "The index artefact is the object all four models load; refusing to "
            "regenerate it silently."
        )
    recorded = frozen.get("indices_artifact", {}).get("sha256")
    if recorded != sha256_file(indices_path):
        raise SplitIdentityError(
            f"{indices_path} does not match the hash recorded in {manifest_path}."
        )
    _verify_loaded_indices(indices_path, split)

    audit = {
        "status": "matched_existing_manifest",
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "indices_sha256": recorded,
        "data_sha256": actual["data_snapshot"]["sha256"],
        "membership_hashes_verified": True,
    }
    return split, audit


def _verify_loaded_indices(indices_path, split: FixedSplit):
    with np.load(indices_path) as arrays:
        for key, expected in (
            ("train_index", split.train_index),
            ("test_index", split.test_index),
            ("train_source_ids", split.train_source_ids),
            ("test_source_ids", split.test_source_ids),
        ):
            if key not in arrays or not np.array_equal(arrays[key], expected):
                raise SplitIdentityError(
                    f"{indices_path}: {key} differs from the recomputed split."
                )
        for family, fold_list in split.folds.items():
            for fold, (fit_idx, validation_idx) in enumerate(fold_list, start=1):
                for suffix, expected in (("fit", fit_idx), ("validation", validation_idx)):
                    key = f"{family}_fold{fold}_{suffix}"
                    if key not in arrays or not np.array_equal(arrays[key], expected):
                        raise SplitIdentityError(
                            f"{indices_path}: {key} differs from the recomputed folds."
                        )


def load_fixed_split(indices_path) -> FixedSplit:
    """Load the persisted split without recomputing it (consumers 08/09/10)."""
    indices_path = Path(indices_path)
    with np.load(indices_path) as arrays:
        keys = set(arrays.files)
        folds = {}
        for family in ("hyperparameter_cv", "calibration_cv", "threshold_oof_cv"):
            fold_list = []
            fold = 1
            while f"{family}_fold{fold}_fit" in keys:
                fold_list.append((
                    arrays[f"{family}_fold{fold}_fit"].copy(),
                    arrays[f"{family}_fold{fold}_validation"].copy(),
                ))
                fold += 1
            folds[family] = fold_list
        return FixedSplit(
            train_index=arrays["train_index"].copy(),
            test_index=arrays["test_index"].copy(),
            train_source_ids=arrays["train_source_ids"].copy(),
            test_source_ids=arrays["test_source_ids"].copy(),
            folds=folds,
        )


def assert_manifest_covers_frame(manifest: dict, n_rows: int) -> None:
    """Check that a manifest describes the frame a consumer just loaded."""
    recorded = int(manifest["data_snapshot"]["n_rows"])
    if recorded != n_rows:
        raise SplitIdentityError(
            f"The split manifest describes {recorded} rows but the loaded file has "
            f"{n_rows}. Secondary analyses need a full primary run; a smoke run "
            "(N_MAX) covers only part of the data."
        )


def assert_split_integrity(split: FixedSplit, n_rows: int) -> None:
    """Invariants every consumer may rely on."""
    train, test = np.asarray(split.train_index), np.asarray(split.test_index)
    if np.intersect1d(train, test).size:
        raise SplitIdentityError("Training and test partitions overlap")
    if len(train) + len(test) != n_rows:
        raise SplitIdentityError("Partitions do not cover the dataset exactly once")
    if not np.array_equal(np.union1d(train, test), np.arange(n_rows)):
        raise SplitIdentityError("Partition union is not the full row index")
    n_train = len(train)
    for family, fold_list in split.folds.items():
        covered = np.concatenate([validation for _, validation in fold_list])
        if not np.array_equal(np.sort(covered), np.arange(n_train)):
            raise SplitIdentityError(
                f"{family}: validation folds do not partition the training partition"
            )
        for fit_idx, validation_idx in fold_list:
            if np.intersect1d(fit_idx, validation_idx).size:
                raise SplitIdentityError(f"{family}: a fold overlaps its own validation part")
            if fit_idx.max(initial=-1) >= n_train or validation_idx.max(initial=-1) >= n_train:
                raise SplitIdentityError(f"{family}: fold indices escape the training partition")
