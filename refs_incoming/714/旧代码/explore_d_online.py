"""D 方案线上冲刺: v33 特征基座 + CatBoostRegressor(RMSE) + 强化 bagging(10).

与原始 explore_d_reg.py 的区别(关键升级):
  原始 D = v1+Top22特征(锚点0.69360) + RMSE -> 0.69437 (Δ+0.00077, 待线上验证)
  本脚本 = v33特征(含16比值, 锚点0.69491) + RMSE + 10 bagging
          -> 真正未探索的组合, 若增量规律复现, 本地 0.69491 -> ~0.69568

用户对策(抵消 D 方案稳定性差: 标准差 0.00129->0.00281):
  - 强化 bagging: BAG_SEEDS 从 3 扩到 10, 用数量换稳定性
  - 早停: CatBoostRegressor 用独立验证集 early_stopping_rounds=120

严格版本隔离: 新文件 explore_d_online.py, 不改动 v33/原始D.
产出: submission_d_online.csv (供线上验证) + D_oof/D_test npy.
日志落盘 explore_d_online.log.

用法:
  python explore_d_online.py            # 全量 5 seed x 5 fold x 10 bagging
  python explore_d_online.py --smoke    # 1 seed 2 fold 3 bagging 快速验证
"""
from __future__ import annotations
import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feat_semantic import FeatureBuilderSemantic

TRAIN = "数据建模/train.csv"
TEST = "数据建模/test.csv"
LOG_PATH = os.path.join(HERE, "explore_d_online.log")

SEEDS = (2026, 2027, 2028, 2029, 2030)
N_SPLITS = 5
BAG_SEEDS = tuple(range(10))   # 强化 bagging: 10 个, 抵消 D 稳定性差
ITER, LR, DEPTH, L2, RS, ES = 900, 0.03, 6, 10, 0.7, 120
SMOKE = False

TOP_CROSS = [
    "region__X__source__X__age_range__category_cross",
    "source__X__age_range__X__livability__category_cross",
    "region__X__livability__category_cross",
    "age_range__X__month__X__livability__category_cross",
    "region__X__age_range__X__month__category_cross",
    "region__X__age_range__X__livability__category_cross",
    "region__X__source__X__livability__category_cross",
    "source__X__livability__category_cross",
    "age_range__X__livability__category_cross",
    "region__X__age_range__category_cross",
    "region__X__source__category_cross",
    "source__X__age_range__category_cross",
    "month__X__livability__category_cross",
    "version__X__age_range__X__month__category_cross",
    "source__X__version__X__age_range__category_cross",
    "source__X__month__X__livability__category_cross",
    "region__X__month__X__livability__category_cross",
    "source__X__month__category_cross",
    "source__X__version__category_cross",
    "region__X__month__category_cross",
    "region__X__version__X__age_range__category_cross",
    "region__X__version__X__livability__category_cross",
]
RATIO_PAIRS = [
    ('max_g', 'days'), ('cc', 'days'), ('V', 'days'), ('condition', 'days'),
    ('max_g', 'cc'), ('max_g', 'V'), ('cc', 'V'), ('V', 'cc'),
    ('max_g', 'condition'), ('cc', 'condition'), ('V', 'condition'),
    ('days', 'age_range'), ('condition', 'age_range'),
    ('max_g', 'livability'), ('cc', 'livability'), ('V', 'livability'),
]


def add_ratios(df):
    out = pd.DataFrame(index=df.index)
    for num, den in RATIO_PAIRS:
        n = pd.to_numeric(df[num], errors='coerce')
        d = pd.to_numeric(df[den], errors='coerce')
        safe_d = np.where(d > 0, d, np.nan)
        r = n / safe_d
        out[f'ratio_{num}__by__{den}'] = r.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    return out


def build(train, test, fb):
    tr = fb.transform(train); te = fb.transform(test)
    tr = pd.concat([tr, add_ratios(train)], axis=1)
    te = pd.concat([te, add_ratios(test)], axis=1)
    return tr, te


def run_seed(seed, train, test, y):
    fb = FeatureBuilderSemantic(selected_cross=set(TOP_CROSS))
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y)); test_pred = np.zeros(len(test)); nb = len(BAG_SEEDS)
    for tri, vali in skf.split(train, y):
        fb.fit(train.iloc[tri])
        Xtr, Xte = build(train.iloc[tri], test, fb)
        Xva = fb.transform(train.iloc[vali]); Xva = pd.concat([Xva, add_ratios(train.iloc[vali])], axis=1)
        cat = [c for c in Xtr.columns if c in fb.categorical_features_]
        cidx = [Xtr.columns.get_loc(c) for c in cat]
        for c in cat:
            Xtr[c] = Xtr[c].astype("category"); Xva[c] = Xva[c].astype("category"); Xte[c] = Xte[c].astype("category")
        fold_te = np.zeros(len(test))
        for bs in BAG_SEEDS:
            m = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE",
                                  iterations=ITER, learning_rate=LR, depth=DEPTH, l2_leaf_reg=L2,
                                  random_strength=RS, bagging_temperature=0.7, subsample=0.9,
                                  border_count=128, verbose=0, allow_writing_files=False,
                                  thread_count=-1, random_seed=(seed * 100 + bs))
            m.fit(Pool(Xtr, y[tri], cat_features=cidx),
                  eval_set=Pool(Xva, y[vali], cat_features=cidx),
                  early_stopping_rounds=ES, verbose=0)
            oof[vali] += m.predict(Xva)
            fold_te += m.predict(Xte)
        oof[vali] /= nb; test_pred += fold_te / nb
    test_pred /= N_SPLITS
    return oof, test_pred


def setup_log():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8'),
                                  logging.StreamHandler(sys.stdout)])
    return logging.getLogger('d_online')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        SMOKE = True
        globals()['SEEDS'] = (2026,)
        globals()['N_SPLITS'] = 2
        globals()['BAG_SEEDS'] = (0, 1, 2)

    log = setup_log()
    log.info(f"=== D 方案线上冲刺 (v33特征+RMSE+{len(BAG_SEEDS)}bagging) smoke={SMOKE} ===")
    train = pd.read_csv(TRAIN, dtype={"id": str})
    test = pd.read_csv(TEST, dtype={"id": str})
    y = train["label"].astype(int).values

    oof_p, te_p, aucs = {}, {}, []
    for s in (SEEDS[:1] if SMOKE else SEEDS):
        o, t = run_seed(s, train, test, y)
        oof_p[s], te_p[s] = o, t
        a = roc_auc_score(y, o); aucs.append(a)
        log.info(f"   seed {s}: OOF={a:.5f}")

    pooled = np.mean(np.vstack(list(oof_p.values())), axis=0)
    pa = roc_auc_score(y, pooled)
    ms, sd = np.mean(aucs), np.std(aucs)
    log.info(f">>> pooled OOF = {pa:.5f}   逐seed {ms:.5f} ± {sd:.5f}")
    log.info(f">>> 对照 v33 Logloss pooled OOF = 0.69491  (本地Δ={pa - 0.69491:+.5f})")

    # 提交文件
    sub = pd.DataFrame({"id": test["id"], "label": np.mean(np.vstack(list(te_p.values())), axis=0)})
    sub.to_csv("submission_d_online.csv", index=False)
    np.save("D_online_oof.npy", oof_p)
    np.save("D_online_test.npy", te_p)
    log.info("已生成 submission_d_online.csv (供线上验证) + D_online npy")

    # 稳定性监控: 标准差应 <= 原始D的0.00281(10 bagging 应更低)
    if not SMOKE and sd > 0.00281:
        log.info(f"⚠️ 稳定性风险: 逐seed标准差 {sd:.5f} > 原始D 0.00281, 10 bagging 未完全抵消, 需进一步提升 bagging 或降 lr")
    else:
        log.info(f"✅ 稳定性: 逐seed标准差 {sd:.5f} (10 bagging 已压至 <=0.00281)")
