"""Atomic checkpoints and run manifests for long experiment runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_value)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv_atomic(path, dataframe):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        dataframe.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def code_fingerprint(paths, *, repo_root=None):
    """SHA-256 over the contents of every file that defines the computation."""
    repo_root = Path(repo_root or Path.cwd()).resolve()
    entries = []
    for path in paths:
        resolved = Path(path).resolve()
        try:
            key = resolved.relative_to(repo_root).as_posix()
        except ValueError:
            key = resolved.name
        entries.append((key, sha256_file(resolved)))
    digest = hashlib.sha256()
    for key, file_hash in sorted(entries):
        digest.update(key.encode())
        digest.update(file_hash.encode())
    return digest.hexdigest()


def load_stage_checkpoint(path, *, fingerprint, force=False):
    """Return a completed stage result, or ``None`` if the stage must run again."""
    path = Path(path)
    if force or not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("code_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Stale checkpoint from a different code version: {path}. "
            "Delete the output directory or rerun with FORCE=1."
        )
    return payload


def write_stage_checkpoint(path, payload, *, fingerprint, elapsed_seconds=None):
    record = {
        "code_fingerprint": fingerprint,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = float(elapsed_seconds)
    write_json_atomic(path, record)
    return record


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repo_root):
    def run(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def build_manifest(
    *,
    repo_root,
    run_name,
    input_files,
    code_files,
    feature_names,
    sample_ids,
    settings,
):
    repo_root = Path(repo_root).resolve()

    def hashed_files(paths):
        records = {}
        for path in paths:
            resolved = Path(path).resolve()
            try:
                key = str(resolved.relative_to(repo_root))
            except ValueError:
                key = str(resolved)
            records[key] = sha256_file(resolved)
        return records

    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "xgboost", "catboost", "interpret", "shap"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "run_name": run_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "command": [sys.executable, *sys.argv],
        "git": _git_metadata(repo_root),
        "packages": packages,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "USE_GPU", "EBM_BAGS", "LR_ITER", "EBM_ITER", "XGB_ITER", "CAT_ITER",
                "N_MAX", "LC_N", "MAX_TEST_YEAR", "TEST_YEARS", "FORCE", "DATASETS",
                "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            )
        },
        "runtime": {
            "cpu_count": os.cpu_count(),
            "use_gpu": os.environ.get("USE_GPU") == "1",
            "float_dtype": "float64 in pipelines, float32 in persisted score arrays",
        },
        "settings": settings,
        "inputs": hashed_files(input_files),
        "code": hashed_files(code_files),
        "feature_names": list(feature_names),
        "shap_sample_ids": sample_ids,
    }
