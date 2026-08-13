"""Artifact and submission I/O helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def save_submission(
    sample: pd.DataFrame,
    prediction: np.ndarray,
    path: Path,
) -> str:
    """Validate and save an id/label submission, returning its SHA256."""
    if "id" not in sample.columns:
        raise ValueError("sample submission must contain id")
    values = np.asarray(prediction, dtype=float)
    if len(values) != len(sample):
        raise ValueError(f"prediction length {len(values)} != sample length {len(sample)}")
    if not np.isfinite(values).all():
        raise ValueError("prediction contains non-finite values")
    submission = sample.loc[:, ["id"]].copy()
    submission["label"] = np.clip(values, 0.001, 0.999)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return sha256_file(path)


def append_experiment(path: Path, payload: dict[str, object]) -> None:
    """Append one compact immutable experiment record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
