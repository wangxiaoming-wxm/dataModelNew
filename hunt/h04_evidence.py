"""Is 0.70 a data-size problem or a signal problem?

Three cheap, independent measurements:

  C. Learning curve.  If AUC were still climbing with sample size, the ceiling
     argument would be premature.
  D. Neighbourhood concordance.  If the label were a (even very complicated)
     function of the features, rows that are near-identical in every
     informative column would share a label far more often than chance.
  E. Irreducible noise.  Regenerate labels from the calibrated risk function and
     measure the spread of achievable AUC.  This says how much of any apparent
     improvement is real and how much is coin-flipping.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

RNG = np.random.default_rng(20260809)


def signal_frame(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Only the columns proved to carry signal, on comparable scales."""
    scale = df.groupby("source")["condition"].median()
    cond_r = (df["condition"] / df["source"].map(scale)).fillna(1.0)
    ratio = df["days"] / cond_r.clip(lower=1e-9)
    cols = {
        "days": df["days"],
        "cond_r": cond_r,
        "log_ratio": np.log(ratio.clip(lower=1e-9)),
        "age_range": df["age_range"].astype(float),
        "region": df["region"].astype("category").cat.codes.astype(float),
        "source": df["source"].astype("category").cat.codes.astype(float),
        "grade": df["grades"].map({"s": 1, "ss": 2, "sss": 3}).astype(float),
    }
    for b in ("t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2"):
        cols[b] = df[b].astype(float)
    F = pd.DataFrame(cols)
    return F.to_numpy(dtype=float), list(F.columns)


def learning_curve(X: np.ndarray, y: np.ndarray) -> list[dict]:
    """OOF AUC as a function of how much training data each model gets."""
    out = []
    for frac in (0.15, 0.3, 0.5, 0.7, 0.85, 1.0):
        aucs = []
        for rep in range(3):
            idx = np.sort(RNG.choice(len(y), int(len(y) * frac), replace=False))
            Xs, ys = X[idx], y[idx]
            oof = np.zeros(len(ys))
            for ti, vi in StratifiedKFold(5, shuffle=True, random_state=rep).split(Xs, ys):
                m = lgb.LGBMClassifier(objective="binary", n_estimators=400,
                                       learning_rate=0.03, num_leaves=15,
                                       min_child_samples=40, feature_fraction=0.8,
                                       lambda_l2=10.0, verbose=-1, n_jobs=2,
                                       random_state=rep)
                m.fit(Xs[ti], ys[ti])
                oof[vi] = m.predict_proba(Xs[vi])[:, 1]
            aucs.append(roc_auc_score(ys, oof))
        out.append({"frac": frac, "n": int(len(y) * frac),
                    "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1))})
        print(f"    n={out[-1]['n']:6d}  AUC={out[-1]['auc_mean']:.5f} "
              f"+-{out[-1]['auc_sd']:.5f}", flush=True)
    return out


def neighbour_concordance(X: np.ndarray, y: np.ndarray) -> list[dict]:
    """Do near-identical rows share a label?"""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=11).fit(Z)
    dist, ind = nn.kneighbors(Z)
    dist, ind = dist[:, 1:], ind[:, 1:]          # drop self

    rows = []
    d1 = dist[:, 0]
    for lo, hi in ((0, 5), (5, 25), (25, 50), (50, 75), (75, 100)):
        m = (d1 >= np.percentile(d1, lo)) & (d1 <= np.percentile(d1, hi))
        same = (y[ind[m, 0]] == y[m]).mean()
        # under independence, P(same) = p^2 + (1-p)^2 within this subset
        p = y[m].mean()
        chance = p ** 2 + (1 - p) ** 2
        rows.append({"dist_pct": f"{lo}-{hi}", "n": int(m.sum()),
                     "p_same_label": float(same), "chance": float(chance),
                     "lift": float(same - chance)})
        print(f"    1-NN distance pct {lo:3d}-{hi:3d}: P(same label)={same:.4f} "
              f"chance={chance:.4f} lift={same - chance:+.4f}", flush=True)
    # neighbourhood label rate as a predictor
    nb_rate = y[ind].mean(axis=1)
    print(f"    AUC of the 10-NN label rate (in-sample, so optimistic): "
          f"{roc_auc_score(y, nb_rate):.4f}")
    return rows


def noise_floor(y: np.ndarray) -> dict:
    """If the labels were re-drawn from the fitted risk, how much would AUC move?"""
    from sklearn.isotonic import IsotonicRegression

    oofs = [np.load(f"artifacts/v2/arm_{a}.npz")["oof"] for a in ("cat_d5", "cat_d6", "cat_alt")]
    score = np.max(np.stack([pd.Series(v).rank(pct=True).to_numpy() for v in oofs]), axis=0)
    p = np.zeros(len(y))
    for ti, vi in StratifiedKFold(10, shuffle=True, random_state=7).split(score, y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score[ti], y[ti])
        p[vi] = iso.predict(score[vi])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    sims = [roc_auc_score(RNG.binomial(1, p), score) for _ in range(300)]
    res = {"mean": float(np.mean(sims)), "sd": float(np.std(sims, ddof=1)),
           "p2.5": float(np.percentile(sims, 2.5)), "p97.5": float(np.percentile(sims, 97.5))}
    print(f"    AUC of the SAME score against labels re-drawn from its own fitted risk: "
          f"{res['mean']:.5f} +-{res['sd']:.5f}  (95% {res['p2.5']:.5f}..{res['p97.5']:.5f})")
    return res


def main() -> None:
    tr = pd.read_csv("data/train.csv")
    y = tr["label"].to_numpy()
    X, names = signal_frame(tr)
    print(f"signal frame {X.shape}: {names}")

    print("\n[C] learning curve (LightGBM on the signal columns)")
    lc = learning_curve(X, y)
    d = lc[-1]["auc_mean"] - lc[-3]["auc_mean"]
    print(f"    slope over the last 2x of data: {d:+.5f} AUC")

    print("\n[D] neighbourhood concordance in the signal space")
    nc = neighbour_concordance(X, y)

    print("\n[E] irreducible noise")
    nf = noise_floor(y)

    json.dump({"learning_curve": lc, "neighbour_concordance": nc, "noise_floor": nf},
              open("hunt/out_h04.json", "w"), indent=2)


if __name__ == "__main__":
    main()
