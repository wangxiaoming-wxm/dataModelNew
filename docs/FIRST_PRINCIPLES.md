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

| 表示 | 池 AUC | vs 表格臂 Spearman |
|---|---:|---:|
| 既有 byte/nibble 池（v3） | ~0.55 | （已融入） |
| **全部 64 个 bit 平面 TE 池** | **0.591** | **≈0.03** |
| within-byte AND 二阶 | 0.681（and_all） | ≈0.14 |
| within-byte OR / XOR / TRI | 0.650 / 0.651 / 0.640 | ≈0.05–0.24 vs and |
| **cross-byte OR / XOR** | **0.657 / 0.675** | ≈0.05–0.10 vs heavy_xor |

嵌套（折内选 bit）仍相对 v3 **+≈0.002**；无挑选全量池混入后提升更大。  
结论：bit 平面及其布尔组合是与表格臂近乎正交的合法信号。

## 晋级配方

### fp_v4 → fp_v5 → fp_v6（已吸收）

```text
v3_dual = 0.55*AM40 + 0.45*(0.7*V7 + 0.3*V2)
bits    = mean_rank(TE_6seed(all 64 id bits))
xs      = mean_rank(TE_6seed(qbin20(x0..x18)))
and_all = 0.5*within-byte AND + 0.5*cross-byte same-bit AND
or/xor/tri = within-byte 对应运算池
```

### fp_v7（已吸收进 v8）

```text
cmean = 0.5*pool(cross-byte OR) + 0.5*pool(cross-byte XOR)
score = 0.15*v3 + 0.10*bits + 0.05*xs + 0.22*and_all
      + 0.06*tri + 0.06*or + 0.06*xor + 0.30*cmean   # cross30
```

### fp_v8（当前主交）

整字节对 XOR/AND（C(8,2)=28）相对 bit 组合是更粗、近正交的键空间。

```text
bytepair_mean = 0.5*pool(byte_i XOR byte_j) + 0.5*pool(byte_i AND byte_j)
score = 0.80*v7_cross30 + 0.20*bytepair_mean   # byte20
```

| 方案 | OOF | nested fold-mean |
|---|---:|---:|
| AM40 | 0.70181 | — |
| v3 dual | 0.70717 | — |
| fp_v4 | 0.71640 | 0.71660 |
| fp_v5 and_heavy | 0.72880 | 0.72900 |
| fp_v6 heavy_xor | 0.74342 | 0.74350 |
| fp_v7 cross30 | 0.75838 | 0.75851 |
| **fp_v8 byte20（主交）** | **0.77016** | **0.77033** |

```bash
bash run_fp_v8.sh
bash run_fp_v8.sh --verify
# 上一档备份
bash run_fp_v7.sh --verify
```

主文件：`submissions/submission_champion.csv`  
备份：`submission_fp_v8_tempered.csv`（= fp_v7）、`submission_fp_v8_aggressive.csv`（0.70*v7+0.30*bytepair）

## 未晋级 / 弱信号

- source 解析 CAR/ENG、t3 后缀：被既有类别特征吸收，混入 champion 无增益  
- id 的 popcount/sorted-bytes：弱或无效  
- 在折外挑选「最强 bit/x」再交：有挑选偏置；刻意用**全量池**避免该偏置  
- Jitter / Lossguide / 特征剪枝 / group-stats：见对应 docs，勿再烧提交  
