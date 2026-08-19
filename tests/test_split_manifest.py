"""Split integrity: one fixed partition, written before anything is fitted."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.seeds import SEEDS
from src.splits import (
    SplitIdentityError,
    assert_split_integrity,
    load_fixed_split,
    make_fixed_split,
    write_or_validate_fixed_split,
)


def _write(tmp_path, frame, dataset="synthetic", **overrides):
    csv_path = tmp_path / f"{dataset}.csv"
    if not csv_path.exists():
        frame.to_csv(csv_path, index=False)
    arguments = {
        "dataset": dataset,
        "data_path": csv_path,
        "y": frame["target"].to_numpy(dtype=int),
        "source_ids": frame["source_row_id"].to_numpy(dtype=np.int64),
        "columns": list(frame.columns),
    }
    arguments.update(overrides)
    return write_or_validate_fixed_split(
        tmp_path / "split_manifest.json",
        tmp_path / "split_indices.npz",
        **arguments,
    )


def test_first_run_writes_prospectively(tmp_path, synthetic_frame):
    split, audit = _write(tmp_path, synthetic_frame)
    assert audit["status"] == "written_prospectively"
    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    assert manifest["provenance"] == "written_prospectively_before_first_fit"
    assert (tmp_path / "split_indices.npz").exists()


def test_second_run_validates_against_the_manifest(tmp_path, synthetic_frame):
    _write(tmp_path, synthetic_frame)
    _, audit = _write(tmp_path, synthetic_frame)
    assert audit["status"] == "matched_existing_manifest"


def test_rejects_a_changed_split_seed(tmp_path, synthetic_frame):
    _write(tmp_path, synthetic_frame)
    with pytest.raises(SplitIdentityError):
        _write(tmp_path, synthetic_frame, random_state=SEEDS["train_test_split"] + 1)


def test_rejects_changed_source_ids(tmp_path, synthetic_frame):
    _write(tmp_path, synthetic_frame)
    shifted = synthetic_frame["source_row_id"].to_numpy(dtype=np.int64) + 1
    with pytest.raises(SplitIdentityError):
        _write(tmp_path, synthetic_frame, source_ids=shifted)


def test_rejects_changed_data_bytes(tmp_path, synthetic_frame):
    _write(tmp_path, synthetic_frame)
    modified = synthetic_frame.copy()
    modified.loc[0, "income"] = modified.loc[0, "income"] + 1.0
    modified.to_csv(tmp_path / "synthetic.csv", index=False)
    with pytest.raises(SplitIdentityError):
        _write(tmp_path, synthetic_frame)


def test_rejects_a_tampered_index_artifact(tmp_path, synthetic_frame):
    split, _ = _write(tmp_path, synthetic_frame)
    payload = dict(np.load(tmp_path / "split_indices.npz"))
    payload["train_index"] = payload["train_index"][:-1]
    np.savez_compressed(tmp_path / "split_indices.npz", **payload)
    with pytest.raises(SplitIdentityError, match="does not match the hash"):
        _write(tmp_path, synthetic_frame)


def test_manifest_records_membership_hashes_and_default_rates(tmp_path, synthetic_frame):
    _write(tmp_path, synthetic_frame)
    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    for side in ("train", "test"):
        partition = manifest["partitions"][side]
        assert len(partition["source_ids_sha256"]) == 64
        assert 0.0 < partition["default_rate"] < 1.0
    for family in ("hyperparameter_cv", "calibration_cv", "threshold_oof_cv"):
        folds = manifest["training_folds"][family]["folds"]
        assert len(folds) == manifest["training_folds"][family]["n_splits"]
        assert all(len(fold["fit_source_ids_sha256"]) == 64 for fold in folds)


def test_partitions_are_disjoint_and_exhaustive(tmp_path, synthetic_frame):
    split, _ = _write(tmp_path, synthetic_frame)
    assert_split_integrity(split, len(synthetic_frame))
    assert np.intersect1d(split.train_index, split.test_index).size == 0
    assert len(split.train_index) + len(split.test_index) == len(synthetic_frame)


def test_split_is_stratified(synthetic_frame):
    y = synthetic_frame["target"].to_numpy(dtype=int)
    train_index, test_index = make_fixed_split(y)
    assert abs(y[train_index].mean() - y[test_index].mean()) < 0.02


def test_split_is_reproducible(synthetic_frame):
    y = synthetic_frame["target"].to_numpy(dtype=int)
    first = make_fixed_split(y)
    second = make_fixed_split(y)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_every_fold_family_partitions_the_training_side(tmp_path, synthetic_frame):
    split, _ = _write(tmp_path, synthetic_frame)
    n_train = len(split.train_index)
    for family, folds in split.folds.items():
        covered = np.concatenate([validation for _, validation in folds])
        assert np.array_equal(np.sort(covered), np.arange(n_train)), family
        for fit_index, validation_index in folds:
            assert np.intersect1d(fit_index, validation_index).size == 0


def test_all_models_load_the_identical_split(tmp_path, synthetic_frame):
    """Loading the artefact is what makes 'the same split' checkable."""
    split, _ = _write(tmp_path, synthetic_frame)
    loaded = load_fixed_split(tmp_path / "split_indices.npz")
    assert np.array_equal(loaded.train_index, split.train_index)
    assert np.array_equal(loaded.test_index, split.test_index)
    for family in split.folds:
        for (fit_a, val_a), (fit_b, val_b) in zip(split.folds[family], loaded.folds[family]):
            assert np.array_equal(fit_a, fit_b)
            assert np.array_equal(val_a, val_b)


def test_integrity_check_rejects_folds_escaping_the_training_partition(
    tmp_path, synthetic_frame
):
    split, _ = _write(tmp_path, synthetic_frame)
    escaped = np.append(split.folds["hyperparameter_cv"][0][0], len(split.train_index))
    split.folds["hyperparameter_cv"][0] = (escaped, split.folds["hyperparameter_cv"][0][1])
    with pytest.raises(SplitIdentityError):
        assert_split_integrity(split, len(synthetic_frame))
