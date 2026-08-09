#!/usr/bin/env python3
"""Embedding MLP arm screen (fold-local vocab; no TE)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from insurance_claim.model import TARGET

CAT_COLS = ["region", "source", "version", "grades", "month", "t3", "code"]
NUM_COLS_BASE = [
    "days",
    "condition",
    "V",
    "cc",
    "max_g",
    "age_range",
    "livability",
    "w1",
    "w2",
    "t1",
    "t2",
    "r1",
    "r2",
    "c1",
    "c2",
    "x19",
    "x20",
]


class EmbMLP(nn.Module):
    def __init__(self, card: dict[str, int], n_num: int, dim: int = 8, hidden: int = 128):
        super().__init__()
        self.embs = nn.ModuleDict(
            {c: nn.Embedding(card[c] + 1, dim, padding_idx=0) for c in card}
        )
        in_dim = dim * len(card) + n_num
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_cat, x_num):
        parts = [self.embs[c](x_cat[c]) for c in self.embs]
        h = torch.cat(parts + [x_num], dim=1)
        return self.net(h).squeeze(1)


def parse_extra(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    m = out["t3"].astype(str).str.extract(r"([+-]?\d+(?:\.\d+)?)([A-Za-z]+)?")
    out["t3_num"] = pd.to_numeric(m[0], errors="coerce")
    out["t3_sfx"] = m[1].fillna("na")
    out["car"] = pd.to_numeric(out["source"].astype(str).str.extract(r"CAR_(\d+)")[0], errors="coerce")
    out["years"] = out["days"] / 365.25
    out["ratio"] = out["condition"] / (out["days"].abs() + 1.0)
    out["log_days"] = np.log1p(out["days"].clip(lower=0))
    return out


def encode_fold(train_df, *others, cat_cols, num_cols):
    vocabs = {}
    x_cats_tr = {}
    for c in cat_cols:
        vals = train_df[c].astype(str).fillna("__NA__")
        uniq = sorted(vals.unique().tolist())
        vocab = {v: i + 1 for i, v in enumerate(uniq)}  # 0 = unk
        vocabs[c] = vocab
        x_cats_tr[c] = torch.tensor(vals.map(vocab).fillna(0).astype(int).to_numpy(), dtype=torch.long)

    def map_cats(df):
        out = {}
        for c in cat_cols:
            vals = df[c].astype(str).fillna("__NA__")
            out[c] = torch.tensor(vals.map(vocabs[c]).fillna(0).astype(int).to_numpy(), dtype=torch.long)
        return out

    scaler = StandardScaler()
    tr_num = scaler.fit_transform(train_df[num_cols].to_numpy(dtype=float))
    others_num = [scaler.transform(df[num_cols].to_numpy(dtype=float)) for df in others]
    x_num_tr = torch.tensor(np.nan_to_num(tr_num, nan=0.0), dtype=torch.float32)
    others_t = [
        (
            map_cats(df),
            torch.tensor(np.nan_to_num(num, nan=0.0), dtype=torch.float32),
        )
        for df, num in zip(others, others_num)
    ]
    card = {c: len(v) for c, v in vocabs.items()}
    return (x_cats_tr, x_num_tr), others_t, card


def train_one(x_cat, x_num, y, x_cat_va, x_num_va, y_va, card, seed=0, epochs=25, bs=512):
    torch.manual_seed(seed)
    model = EmbMLP(card, n_num=x_num.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    y_t = torch.tensor(y, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.float32)
    n = len(y)
    best_state = None
    best_auc = -1.0
    for ep in range(epochs):
        model.train()
        idx = np.random.RandomState(seed * 1000 + ep).permutation(n)
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            xc = {c: x_cat[c][b] for c in x_cat}
            logits = model(xc, x_num[b])
            loss = loss_fn(logits, y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(x_cat_va, x_num_va)
            prob = torch.sigmoid(logits).numpy()
            auc = roc_auc_score(y_va, prob)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(model(x_cat_va, x_num_va)).numpy()
    return model, p_va, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b8_nn"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026])
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train[TARGET].astype(int).to_numpy()
    tr = parse_extra(train.drop(columns=[TARGET]))
    te = parse_extra(test)
    cat_cols = CAT_COLS + ["t3_sfx"]
    num_cols = NUM_COLS_BASE + ["t3_num", "car", "years", "ratio", "log_days"] + [
        f"x{i}" for i in range(19) if f"x{i}" in tr.columns
    ]
    # fillna numerics
    for c in num_cols:
        med = tr[c].median()
        tr[c] = tr[c].fillna(med)
        te[c] = te[c].fillna(med)

    oof = np.zeros(len(tr))
    pte_acc = np.zeros(len(te))
    fold_rows = []
    for seed in args.seeds:
        oof_s = np.zeros(len(tr))
        pte = np.zeros(len(te))
        for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y)):
            (xc, xn), others, card = encode_fold(
                tr.iloc[a].reset_index(drop=True),
                tr.iloc[b].reset_index(drop=True),
                te,
                cat_cols=cat_cols,
                num_cols=num_cols,
            )
            (xc_va, xn_va), (xc_te, xn_te) = others
            model, p_va, best_auc = train_one(
                xc, xn, y[a], xc_va, xn_va, y[b], card, seed=seed + fold
            )
            oof_s[b] = p_va
            with torch.no_grad():
                pte += torch.sigmoid(model(xc_te, xn_te)).numpy() / 5.0
            fold_rows.append({"seed": seed, "fold": fold, "auc": float(best_auc)})
            print(f"nn seed={seed} fold={fold} auc={best_auc:.5f}", flush=True)
        print(f"nn seed={seed} OOF={roc_auc_score(y, oof_s):.6f}", flush=True)
        oof += oof_s
        pte_acc += pte
    oof /= len(args.seeds)
    pte_acc /= len(args.seeds)

    # compare vs B7
    b6 = np.load("artifacts/b6_frozen/predictions.npz")
    plus = np.load("reference/v10/oof_plus_h2_10.npz")["oof"]
    max3 = np.maximum(np.maximum(b6["oof_gap"], b6["oof_gap_bag"]), plus)
    max4 = np.maximum(max3, oof)
    metrics = {
        "solo_auc": float(roc_auc_score(y, oof)),
        "max4_auc": float(roc_auc_score(y, max4)),
        "corr_max3": float(np.corrcoef(oof, max3)[0, 1]),
        "folds": fold_rows,
        "delta_vs_b7": float(roc_auc_score(y, max4) - 0.7027049552615718),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", oof=oof, test=pte_acc, y=y)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
