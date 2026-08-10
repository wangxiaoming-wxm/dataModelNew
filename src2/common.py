"""Shared data loading, feature engineering and CV utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "data"

RAW_NOISE_COLS = [f"x{i}" for i in range(18)] + ["cc", "max_g", "V", "x19"]
BIN_COLS = ["t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2"]
GRADE_MAP = {"s": 1, "ss": 2, "sss": 3}


def load_raw():
    tr = pd.read_csv(f"{DATA_DIR}/train.csv")
    te = pd.read_csv(f"{DATA_DIR}/test.csv")
    return tr, te


def add_parsed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t3n"] = df["t3"].str.extract(r"^([0-9.]+)")[0].astype(float)
    df["t3s"] = df["t3"].str.extract(r"([A-Za-z]+)$")[0]
    df["car"] = df["source"].str.extract(r"CAR_(\d+)")[0].astype(int)
    df["eng"] = df["source"].str.extract(r"ENG_(\d+)")[0].astype(int)
    df["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    df["ver_num"] = df["version"].str.extract(r"v(\d+)")[0].astype(int)
    df["mon_num"] = df["month"].str.extract(r"M(\d+)")[0].astype(int)
    return df


def make_splits(y: np.ndarray, n_splits: int = 5, n_repeats: int = 4, seed0: int = 20260808):
    """Deterministic list of (repeat, fold, tr_idx, va_idx) shared across experiments."""
    out = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed0 + r)
        for f, (ti, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
            out.append((r, f, ti, vi))
    return out


def cv_report(y: np.ndarray, oof_by_repeat: dict[int, np.ndarray]) -> dict:
    aucs = [roc_auc_score(y, o) for o in oof_by_repeat.values()]
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "aucs": [float(a) for a in aucs],
    }
