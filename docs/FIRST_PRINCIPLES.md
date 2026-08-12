# 第一性原理复查（证据，非猜测）

## 问题

既有模型是否遗漏了：(1) 原始列；(2) 更丰富的 id 表示？

## 证据 A：原始列用量

`build_main` / `build_alt`（`src_super/train_super714.py`）实际读取的原始字段：

| 使用方式 | 列 |
|---|---|
| 核心数值/分箱 | days, condition, age_range, grades, region, source, month, version |
| 二值进 bin_pat | t1, t2, r1, r2, c1, c2, w1, w2 |
| 类别串 | x19, x20, livability, t3, code |
| 仅 main 数值 | cc, max_g, V |
| **完全未进入特征** | **x0..x18**（以及 id 本身，直至后续 TE） |

逐列 fold-local TE / 数值 AUC 扫描：`artifacts/first_principles/raw_column_sweep.csv`

被忽略列中较强单体信号（仍弱于 days）：

| 列 | TE AUC | vs AM40 Spearman |
|---|---:|---:|
| x5 | 0.533 | +0.05 |
| x16 | 0.515 | ≈0 |
| x17 | 0.517 | +0.07 |
| x1 (num) | 0.528 | — |

**选择无关**地把 x0..x18 全部做成 qbin20 TE 再 rank-mean：池 AUC≈0.549，与 champion Spearman≈0.08。

## 证据 B：更丰富的 id 表示

id = 16 位 hex（8 bytes），各 hex 位熵≈4 bit（均匀）。

| 表示 | 池 AUC | vs v3 champion Spearman |
|---|---:|---:|
| 既有 byte/nibble 池（v3） | ~0.55 | （已融入） |
| **全部 64 个 bit 平面 TE 池** | **0.591** | **≈0.03** |
| 16 nibble 全池 | 0.542 | — |
| 8 byte 全池 | 0.545 | — |

嵌套（折内选 bit）仍相对 v3 **+≈0.002**；无挑选全 64-bit 池混入后提升更大。  
结论：bit 平面是比「整字节 TE」更细、且与现有融合近乎正交的合法信号。

## 晋级配方（fp_v4）

```text
v3_dual = 0.55*AM40 + 0.45*(0.7*V7 + 0.3*V2)     # 既有
bits    = mean_rank(TE_6seed(all 64 id bits))     # 新
xs      = mean_rank(TE_6seed(qbin20(x0..x18)))    # 新（回收忽略列）
score   = 0.50*v3_dual + 0.35*bits + 0.15*xs
```

| 方案 | OOF | nested |
|---|---:|---:|
| AM40 | 0.70181 | — |
| v3 dual | 0.70717 | 0.70737 |
| **fp_v4（主交）** | **0.71640** | **≈0.71660** |

```bash
bash run_fp_v4.sh
bash run_fp_v4.sh --verify
```

主文件：`submissions/submission_champion.csv`

## 未晋级 / 弱信号

- source 解析 CAR/ENG、t3 后缀：被既有类别特征吸收，混入 champion 无增益  
- id 的 xor/popcount/sorted-bytes：弱或无效  
- 在折外挑选「最强 bit/x」再交：有挑选偏置；v4 刻意用**全量池**避免该偏置  
