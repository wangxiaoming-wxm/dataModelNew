"""Shared helpers for the fp_v8 audit and the vz21 rebuild.

Everything here is deliberately written from scratch (fast numpy target
encoding) so the audit does not inherit any assumption from the code under
review.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA = Path("/workspace/data")
ART = Path("/workspace/vz20/artifacts/vz20")
TE_SMOOTH = 20.0


def load_data():
    train = pd.read_csv(DATA / "train.csv", dtype={"id": str})
    test = pd.read_csv(DATA / "test.csv", dtype={"id": str})
    sample = pd.read_csv(DATA / "submit_sample.csv", dtype={"id": str})
    return train, test, sample


def id_bytes(ids: pd.Series) -> np.ndarray:
    """(n, 8) uint8 matrix of the 8 bytes behind each 16-hex-char id."""
    raw = ids.astype(str).str.lower().to_numpy()
    buf = np.frombuffer("".join(raw).encode("ascii"), dtype=np.uint8).reshape(len(raw), 16)
    # ascii hex -> nibble
    nib = np.where(buf >= 97, buf - 87, buf - 48).astype(np.uint8)
    return (nib[:, 0::2] << 4 | nib[:, 1::2]).astype(np.uint8)


def id_bits(byte_mat: np.ndarray) -> np.ndarray:
    """(n, 64) uint8 bit matrix, bit j of byte b at column b*8+j."""
    n = byte_mat.shape[0]
    out = np.empty((n, 64), dtype=np.uint8)
    for b in range(8):
        for j in range(8):
            out[:, b * 8 + j] = (byte_mat[:, b] >> j) & 1
    return out


def factorize(tr_key: np.ndarray, te_key: np.ndarray):
    """Joint factorization of a train/test key column into dense int codes."""
    both = np.concatenate([tr_key, te_key])
    uniq, codes = np.unique(both, return_inverse=True)
    return codes[: len(tr_key)], codes[len(tr_key) :], len(uniq)


def te_fit(codes: np.ndarray, y: np.ndarray, n_levels: int, prior: float, smooth: float = TE_SMOOTH):
    """Smoothed target-encoding table from the given rows only."""
    s = np.bincount(codes, weights=y, minlength=n_levels)
    c = np.bincount(codes, minlength=n_levels).astype(float)
    return (s + smooth * prior) / (c + smooth)


def te_oof(codes: np.ndarray, y: np.ndarray, n_levels: int, seed: int, n_splits: int = 5):
    """Fold-internal OOF target encoding. No label of a row ever reaches it."""
    oof = np.empty(len(y), dtype=float)
    prior = float(y.mean())
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tri, vai in skf.split(np.zeros(len(y)), y):
        table = te_fit(codes[tri], y[tri], n_levels, float(y[tri].mean()))
        seen = np.bincount(codes[tri], minlength=n_levels) > 0
        v = table[codes[vai]]
        v[~seen[codes[vai]]] = prior
        oof[vai] = v
    return oof


def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (ties handled), ~10x faster than sklearn in tight loops."""
    from scipy.stats import rankdata

    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def rank01(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(x) / len(x)
