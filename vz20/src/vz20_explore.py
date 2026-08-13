"""vz20 换轴探索: 系统性检验与 ratio/rate 解耦的强信号 / 泄漏 / 结构.

一次跑完所有"换轴"假设, 输出 artifacts/vz20/explore.json. 全部诚实 OOF/held-out.
结论(见 docs/CHANGE_AXIS_FINDINGS.md): 提供的 CSV 内不存在可利用的解耦结构; 冠军 0.749 的
+0.03~0.05 鸿沟不在这些文件里(指向外部数据/平台级泄漏).
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import spearmanr, rankdata

VZ20 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 90210


def main():
    tr = pd.read_csv(os.path.join(VZ20, "..", "data", "train.csv"), dtype={"id": str})
    te = pd.read_csv(os.path.join(VZ20, "..", "data", "test.csv"), dtype={"id": str})
    y = tr["label"].values.astype(int)
    n = len(y)
    R = {}

    # 1. 单变量 AUC 上限
    num = [c for c in tr.columns if c not in ("id", "label") and pd.api.types.is_numeric_dtype(tr[c])]
    uni = {}
    for c in num:
        v = np.nan_to_num(tr[c].values.astype(float), nan=np.nanmedian(tr[c].values.astype(float)))
        if v.std() == 0:
            continue
        uni[c] = float(roc_auc_score(y, v))
    R["univariate_auc_top"] = dict(sorted(uni.items(), key=lambda kv: -abs(kv[1] - 0.5))[:8])

    # 2. x 列: 是否为真实特征的噪声投影 + 残差是否含信号
    xcols = [f"x{i}" for i in range(19)]
    X = np.nan_to_num(tr[xcols].values.astype(float))
    realnum = ["days", "condition", "cc", "V", "max_g", "age_range"]
    Rr = np.nan_to_num(tr[realnum].values.astype(float))
    Rs = (Rr - Rr.mean(0)) / (Rr.std(0) + 1e-9)
    Cx = np.corrcoef(X.T)
    R["x_max_offdiag_corr"] = float(np.abs(Cx - np.eye(19)).max())
    resid = X - LinearRegression().fit(Rs, X).predict(Rs)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def oof_hgb(M):
        oof = np.zeros(n)
        for tri, vi in skf.split(M, y):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=4,
                                                l2_regularization=1.0, random_state=0)
            m.fit(M[tri], y[tri])
            oof[vi] = m.predict_proba(M[vi])[:, 1]
        return float(roc_auc_score(y, oof))
    R["hgb_rawx_oof"] = oof_hgb(X)
    R["hgb_xresidual_oof"] = oof_hgb(resid)

    # 3. x-空间 kNN (流形假设)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    oof = np.zeros(n)
    for tri, vi in skf.split(Xs, y):
        knn = KNeighborsClassifier(n_neighbors=50, weights="distance").fit(Xs[tri], y[tri])
        oof[vi] = knn.predict_proba(Xs[vi])[:, 1]
    R["knn_xspace_oof"] = float(roc_auc_score(y, oof))

    # 4. id 结构 (非 TE)
    tr_id = tr["id"].apply(lambda s: int(s, 16)).values
    R["id_overlap_train_test"] = len(set(tr["id"]) & set(te["id"]))
    R["spearman_idint_label"] = float(spearmanr(tr_id, y)[0])
    order = np.argsort(tr_id)
    roll = np.convolve(y[order], np.ones(1000) / 1000, mode="valid")
    R["id_sorted_rolling_label_range"] = [float(roll.min()), float(roll.max())]

    # 5. 行序 / 自相关
    R["rowindex_label_auc"] = float(roc_auc_score(y, np.arange(n)))
    yc = y - y.mean()
    R["label_autocorr_lag1"] = float(np.corrcoef(yc[:-1], yc[1:])[0, 1])

    # 6. 重复 / 分组
    feat = [c for c in tr.columns if c not in ("id", "label")]
    R["exact_dup_feature_rows_train"] = int(tr.duplicated(subset=feat).sum())
    R["dup_across_train_test"] = int(pd.concat([tr[feat], te[feat]]).duplicated().sum())
    cats = ["month", "region", "t3", "code", "x19", "x20", "age_range", "livability",
            "source", "grades", "version", "t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2"]
    ktr = tr[cats].astype(str).agg("|".join, axis=1)
    kte = te[cats].astype(str).agg("|".join, axis=1)
    R["catkey_unique_frac_train"] = float((ktr.value_counts() == 1).mean())
    R["catkey_test_coverage_in_train"] = float(kte.isin(set(ktr)).mean())

    # 7. 对抗验证 train vs test
    from catboost import CatBoostClassifier, Pool
    A = pd.concat([tr[feat], te[feat]], ignore_index=True)
    catf = [c for c in feat if A[c].dtype == object]
    for c in catf:
        A[c] = A[c].fillna("NA").astype(str)
    for c in feat:
        if c not in catf:
            A[c] = pd.to_numeric(A[c], errors="coerce")
    z = np.r_[np.zeros(n), np.ones(len(te))]
    oof = np.zeros(len(z))
    for tri, vi in StratifiedKFold(3, shuffle=True, random_state=0).split(A, z):
        m = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, verbose=0, allow_writing_files=False)
        m.fit(Pool(A.iloc[tri], z[tri], cat_features=catf), verbose=False)
        oof[vi] = m.predict_proba(A.iloc[vi])[:, 1]
    R["adversarial_train_test_auc"] = float(roc_auc_score(z, oof))

    os.makedirs(os.path.join(VZ20, "artifacts", "vz20"), exist_ok=True)
    with open(os.path.join(VZ20, "artifacts", "vz20", "explore.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
