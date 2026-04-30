# Revision Experiments — BIB-26-0525

投稿期刊：*Briefings in Bioinformatics*  
论文：K-attention（面向 omics 序列分析的生物约束注意力算子）  
当前阶段：**大修（Major Revision）**

---

## 审稿人要求 & 实验状态

| 编号 | 审稿人 | 要求 | 状态 | 对应脚本 |
|------|--------|------|------|----------|
| **A** | R1-1 / R2-3 | 超参数敏感性扫描（kernel_size × num_kernels）| ✅ **Done** (3 seeds, 2026-04-24) | `run_expA.sh` / `run_expA_seeds12.sh` |
| **B** | R1-3 / R1-5 / R2-1 | 完整学习曲线（6模型 × 5数据量 × 3 seeds）| ✅ **Done** (5 seeds, 2026-04-24): RC 1k–100k × 5 seeds, Markov 2k–100k × 5 seeds | `run_expB_parallel.sh` / `submit_expB_ext.sh` |
| **C** | R1-4 | 约束消融（KNET_rc vs KNET_uncons_rc）| ✅ **Done** (2026-04-24): 小样本差距显著（KNET_rc n=10k→0.991 vs KNET_uncons→0.539）| `run_expC.sh` / `run_expC_ablation.sh` |
| **D（new）** | — | Simu16 学习曲线对比（KNET_rc vs CNN-TF，两台机器）| ✅ **Done** (2026-04-24) | `run_expD_192.sh` + cluster |
| **H** | R1-1/R2-3 | RC 多难度学习曲线（random_fix2/fix1/rand，3 configs × 6 models × 5 seeds）| ✅ **Done** (random_fix1/fix2 complete) | `submit_rc_tasks.sh` |
| **I** | R1-1/R2-3 | Markov 多熵值学习曲线（markov_0_75/1_0/1_25，3 configs × 6 models × 5 seeds）| 🔄 **Running**（KNET 缺失 0_75/1_25）| `submit_markov_tasks.sh` |
| **E** | R1-2 | 缩小论断范围（"omics pattern recognition"）| ⏳ Pending（文字修改）| — |
| **F** | R1-6 | 语言润色（口语表达）| ⏳ Pending（文字修改）| — |
| **G** | R2-2 | 回复 Swin/ARConv（2D视觉架构非 omics 标准基线）| ⏳ Pending（文字回复）| — |

---

## Exp A：超参数扫描（✅ 完成，3 seeds，2026-04-24）

**目标**：验证 KNET 对 kernel_size 和 num_kernels 的不敏感性。

| 参数 | 取值范围 |
|------|----------|
| kernel_size | 6, 8, 10, 12, 15 |
| num_kernels | 16, 32, 64, 128 |
| 数据集 | abs-ran_fix2（RC任务）+ markov_1_0_50000（Markov任务）|
| 总运行数 | 40 runs × 3 seeds = 120 runs（全部完成）|

**结果（k=12, nk=64 为论文默认值）：**

RC (abs-ran_fix2, KNET_rc)：

| num_kernels→ | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| k=6 | 0.875 | 0.914 | 0.929 | 0.944 |
| k=8 | 0.914 | 0.922 | 0.941 | **0.951** |
| k=10 | 0.908 | 0.918 | 0.936 | 0.943 |
| k=12 | 0.900 | 0.917 | 0.924 | 0.933 |
| k=15 | 0.894 | 0.895 | 0.891 | 0.921 |

Markov (markov_1_0_50000, KNET)：

| num_kernels→ | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| k=6 | 0.834 | 0.824 | 0.828 | 0.833 |
| k=8 | 0.869 | 0.866 | 0.869 | 0.858 |
| k=10 | 0.869 | 0.875 | 0.873 | **0.875** |
| k=12 | 0.864 | 0.867 | 0.868 | 0.869 |
| k=15 | 0.854 | 0.865 | 0.866 | 0.865 |

**结论**：num_kernels 增大整体有益；kernel_size 在 8–12 内差异小，默认 k=12 合理。

---

## Exp B：学习曲线（✅ 完成，5 seeds，2026-04-24）

**目标（最终）**：6 模型 × 7 RC数据量 + 6 Markov数据量 × 5 seeds（RC: 1k/2k/5k/10k/20k/50k/100k；Markov: 2k/5k/10k/20k/50k/100k）。全部完成，607 runs in exp_results_merged.csv。

**RC 任务（abs-ran_fix2，Simu7），5-seed mean AUROC：**

| 模型 | n=1k | n=2k | n=5k | n=10k | n=20k | n=50k | n=100k |
|------|------|------|------|-------|-------|-------|--------|
| KNET_rc | 0.656 | 0.730 | 0.916* | **0.999** | **0.999** | **0.999** | **0.999** |
| cnn_transformer_pm | 0.631 | 0.846 | 0.942 | 0.976 | 0.985 | 0.987 | 0.991 |
| transformer_cls | 0.800 | 0.825 | 0.819 | 0.829 | 0.988 | 0.993 | 0.993 |
| cnn | 0.715 | 0.799 | 0.852 | 0.898 | 0.887 | 0.893 | 0.909 |
| transformer_cls_kmer | 0.595 | 0.517 | 0.602 | 0.621 | 0.690 | 0.782 | 0.965 |
| mha | 0.639 | 0.714 | 0.718 | 0.743 | 0.743 | 0.765 | 0.795 |

*KNET_rc n=5k: seed=1 异常值 0.595（未收敛），其余 4 seeds ~0.995；报告时使用 median 或注明。

**Markov 任务，5-seed mean AUROC：**

| 模型 | n=2k | n=5k | n=10k | n=20k | n=50k | n=100k |
|------|------|------|-------|-------|-------|--------|
| KNET | 0.677 | 0.822 | **0.888** | **0.835** | 0.859 | 0.878 |
| cnn_transformer_pm | 0.553 | 0.728 | 0.825 | 0.783 | 0.840 | 0.852 |
| transformer_cls | 0.574 | 0.564 | 0.542 | 0.659 | 0.830 | 0.826 |
| cnn | 0.548 | 0.640 | 0.780 | 0.789 | 0.854 | 0.868 |
| transformer_cls_kmer | 0.538 | 0.707 | 0.809 | 0.772 | 0.813 | 0.826 |
| mha | 0.590 | 0.549 | 0.537 | 0.535 | 0.609 | 0.625 |

---

## Exp C：约束消融（✅ 完成，2026-04-24 扩展版）

**目标（修订版）**：对比有/无 point-to-point 约束在**小样本**下的性能差异（k=12, nk=64，数据集=random_rand 最难任务）。

### 满载数据消融（原版 run_expC.sh）

| 数据集 | KNET_rc mean | KNET_uncons_rc mean | Δ |
|--------|-------------|---------------------|---|
| abs-ran_fix2 (Simu7) | **0.9995** | 0.9992 | +0.0003 |
| random_rand (Simu16) | **0.9939** | 0.9938 | +0.0001 |

Markov (n=50k): KNET **0.8695** vs KNET_uncons 0.8605 (+0.009)

### 小样本消融（run_expC_ablation.sh，3 seeds，random_rand）

**RC 任务（random_rand，Simu16，最难）：**

| 模型 | n=2k | n=5k | n=10k | n=full |
|------|------|------|-------|--------|
| **KNET_rc (constrained, nk=64)** | 0.689 | 0.827 | **0.991** | **0.994** |
| KNET_uncons_rc (nk=64, 12×params) | 0.562 | 0.521 | 0.539 | 0.994 |
| KNET_uncons_rc (nk=5, param-matched) | 0.566 | 0.522 | 0.519 | — |

**关键发现**：无约束模型在 n=2k–10k 完全无法学习（AUROC ~0.52，接近随机），而约束模型 n=10k 已达 0.991。两者仅在满载数据时收敛。nk=5 参数匹配版与 nk=64 无约束版表现相似，说明参数量不是关键，约束本身是核心。

**Markov 任务（markov_1_0，3 seeds）：**

| 模型 | n=2k | n=5k | n=10k |
|------|------|------|-------|
| **KNET (constrained, nk=64)** | 0.677 | 0.822 | **0.888** |
| KNET_uncons (nk=64) | 0.638 | 0.779 | 0.847 |
| KNET_uncons (nk=5) | 0.614 | 0.737 | 0.865 |

约束带来一致 ~0.04 提升（小样本段）。

---

## Exp D（new）：Simu16 KNET_rc vs CNN-TF 学习曲线（✅ 完成，2026-04-24）

**目标**：在最难配置（Simu16，random_rand，0% PWM）上对比 KNET_rc vs 两种 CNN-TF 模型。

**任务分配**：
- KNET_rc（15 runs）→ 192.168.3.17（RTX 4090，lr=1e-2，batch=512）
- cnn_transformer_pm / cnn_transformer（30 runs）→ luminary 集群（RTX 3090，lr=1e-5，batch=128）

**结果（mean AUROC，KNET_rc n=5000 seed=0 异常值已排除）：**

| n | KNET_rc | cnn_transformer_pm | cnn_transformer |
|---|---------|---------------------|-----------------|
| 5,000 | **0.985** | 0.505 | 0.522 |
| 10,000 | **0.991** | 0.541 | 0.510 |
| 20,000 | **0.992** | 0.530 | 0.511 |
| 50,000 | **0.993** | 0.568 | 0.553 |
| 100,000 | **0.994** | 0.683 | 0.564 |

**关键结论**：CNN-Transformer 在 Simu16 全程接近随机（0.50–0.68），无论数据量多大都无法有效学习 RC 模式；KNET_rc 从最小样本起即达 ~0.985，差距约 0.45–0.49。这是 KNET 的 RC 感知 inductive bias 相对通用序列模型的决定性优势。

---

## 合并结果文件

```bash
# 两台机器结果合并后统一使用：
results/exp_results_merged.csv   # 278 条记录（192.168.3.17 + luminary）

# 分析命令：
cd src/simulation
python3 analyze_results.py --exp A --merged
python3 analyze_results.py --exp B --merged
python3 analyze_results.py --exp C --merged
python3 analyze_results.py --exp D --merged
```

---

## Exp H：RC Multi-Difficulty Learning Curves（✅ Done for fix2/fix1, ⏳ rand partial）

**目标**：在 3 个 RC 难度级别上验证 KNET_rc 的数据效率优势（k=12, nk=64, 6 models × 6 data sizes × 5 seeds = 180 runs each）。

### RC Task 1 — random_fix2（40% PWM, 177/180 runs, 5-seed mean AUROC）

| 模型 | n=1k | n=2k | n=5k | n=20k | n=50k | n=100k |
|------|------|------|------|-------|-------|--------|
| **KNET_rc** | **0.898** | **0.982** | **0.991** | **0.994** | **0.998** | **0.998** |
| cnn_transformer_pm | 0.619 | 0.720 | 0.818 | 0.933 | 0.979 | 0.988 |
| transformer_cls | 0.568 | 0.540 | 0.553 | 0.896 | 0.988 | 0.991 |
| cnn | 0.569 | 0.514 | 0.588 | 0.688 | 0.707 | 0.715 |
| transformer_cls_kmer | 0.602 | 0.571 | 0.538 | 0.625 | 0.662 | 0.687 |
| mha | 0.578 | 0.559 | 0.550 | 0.521 | 0.553 | 0.553 |

**关键发现**：KNET_rc 在 n=1k 即达 0.898，远超所有 baseline（最高 0.619）。CNN-TF/transformer 需 n=50k–100k 才逼近 KNET。

### RC Task 2 — random_fix1（20% PWM, 180/180 runs, 5-seed mean AUROC）

| 模型 | n=1k | n=2k | n=5k | n=20k | n=50k | n=100k |
|------|------|------|------|-------|-------|--------|
| **KNET_rc** | **0.544** | **0.698** | **0.990** | **0.994** | **0.994** | **0.995** |
| cnn_transformer_pm | 0.589 | 0.559 | 0.560 | 0.652 | 0.954 | 0.989 |
| transformer_cls | 0.534 | 0.516 | 0.593 | 0.567 | 0.583 | **0.993** |
| cnn | 0.504 | 0.484 | 0.574 | 0.535 | 0.562 | 0.567 |
| mha | 0.506 | 0.540 | 0.575 | 0.544 | 0.553 | 0.552 |
| transformer_cls_kmer | 0.538 | 0.526 | 0.514 | 0.533 | 0.550 | 0.529 |

**关键发现**：KNET_rc n=1k→0.544（与 random 相近），但 n=5k 即跳至 0.990！CNN-TF/transformer 需 50k–100k 才追上。CNN/MHA/kmer 所有数据量均失败（<0.57）。KNET_rc 仅用 5k 样本就超越所有模型在 100k 的性能（除 TF 的 0.993 外）。

### RC Task 3 — random_rand（0% PWM, ⏳ incomplete: 108/148+ runs）

| 模型 | n=1k | n=2k | n=5k | n=20k | n=50k | n=100k |
|------|------|------|------|-------|-------|--------|
| **KNET_rc** | 0.599 | — | **0.982** | **0.993** | **0.993** | **0.994** |
| cnn_transformer_pm | 0.579 | 0.567 | 0.506 | 0.528 | 0.667 | 0.806 |
| transformer_cls | 0.630 | 0.566 | 0.541 | 0.534 | 0.558 | 0.567 |
| transformer_cls_kmer | 0.519 | 0.536 | 0.526 | 0.545 | 0.522 | — |

**关键发现**：在最难任务上，KNET_rc n=5k→0.982、n=20k→0.993。所有其他模型完全失败（cnn_transformer_pm 最高仅 0.806 at n=100k）。**但数据不完整**：KNET_rc 缺失 n=2k/10k 及部分 seeds；CNN/MHA 全部缺失。

---

## Exp I：Markov Multi-Entropy Learning Curves（🔄 Running，仅 3/6 模型有 0_75/1_25 数据）

**目标**：在 3 个 Markov 熵值级别上验证 KNET 的数据效率（6 models × 6 data sizes × 5 seeds）。

### Markov Task 1 — markov_0_75（H=0.75, easiest, ⏳ KNET/CNN-TF/TF 数据缺失）

| 模型 | n=2k | n=5k | n=10k | n=20k | n=50k | n=100k |
|------|------|------|-------|-------|-------|--------|
| **cnn** | 0.723 | 0.858 | 0.897 | 0.925 | **0.943** | 0.940 |
| transformer_cls_kmer | 0.769 | 0.843 | 0.883 | 0.906 | 0.923 | 0.902 |
| mha | 0.565 | 0.566 | 0.624 | 0.605 | 0.619 | 0.552 |
| KNET | — | — | — | — | — | — |

### Markov Task 2 — markov_1_0（H=1.0, medium, ✅ 完成 from Exp B）

| 模型 | n=2k | n=5k | n=10k | n=20k | n=50k | n=100k |
|------|------|------|-------|-------|-------|--------|
| **KNET** | 0.677 | 0.800 | **0.890** | 0.834 | 0.869 | 0.879 |
| cnn | 0.548 | 0.636 | 0.795 | 0.788 | 0.854 | 0.868 |
| cnn_transformer_pm | 0.553 | 0.728 | 0.816 | 0.788 | 0.845 | 0.855 |
| transformer_cls_kmer | 0.538 | 0.730 | 0.823 | 0.768 | 0.815 | 0.826 |
| transformer_cls | 0.574 | 0.558 | 0.537 | 0.711 | 0.831 | 0.828 |
| mha | 0.590 | 0.547 | 0.546 | 0.544 | 0.607 | 0.613 |

### Markov Task 3 — markov_1_25（H=1.25, hardest, ⏳ KNET/CNN-TF/TF 数据缺失）

| 模型 | n=2k | n=5k | n=10k | n=20k | n=50k | n=100k |
|------|------|------|-------|-------|-------|--------|
| **cnn** | 0.522 | 0.566 | 0.628 | 0.629 | 0.755 | **0.759** |
| transformer_cls_kmer | 0.548 | 0.599 | 0.593 | 0.660 | 0.730 | 0.731 |
| mha | 0.496 | 0.530 | 0.593 | 0.566 | 0.557 | 0.539 |
| KNET | — | — | — | — | — | — |

**分析**：Markov H=0.75/1.25 的 KNET、cnn_transformer_pm、transformer_cls 数据正在集群运行中（`squeue -u liut` 查看）。当前可用的 3 个模型（cnn/mha/transformer_cls_kmer）显示 H=1.25 对所有模型都极具挑战性（最高 cnn 仅 0.759 at n=100k），而 H=0.75 相对容易（cnn 达 0.943）。

---

## 下一步

1. **等待 Markov Tasks 完成**：H=0.75/1.25 缺 KNET/cnn_transformer_pm/transformer_cls 数据
2. **等待 RC Task 3 补齐**：random_rand 缺 KNET_rc 剩余 seeds + CNN/MHA 数据
3. **绘图**：基于完整 `exp_results_merged.csv` 生成所有实验论文图
4. **文字修改**（E/F/G）：缩小论断范围、语言润色、回复 Swin/ARConv
5. **论文 Revision 提交**

---

## 关键修改说明（相对原始代码）

### run_bmk.py 修改点
1. **新增 model_type**：`KNET_rc`、`KNET_uncons_rc`、`KNET`、`KNET_uncons`、`cnn_transformer`、`cnn_transformer_pm`
2. **AUROCThresholdStop callback**：val_auroc ≥ 0.99 时提前终止（--auroc-threshold）
3. **新增参数**：`--batch-size`（默认512）、`--auroc-threshold`（默认0.99）、`--sample-size`

### 模型分工
- RC 任务 → `KNET_rc`（`KattentionModel`，reverse=True，无 mask，lr=1e-2）
- Markov 任务 → `KNET`（`KattentionModel_mask`，±2对角线 band mask，lr=1e-2）
- CNN-TF baseline → lr=1e-4（Simu7）/ lr=1e-5（Simu16）

### 新增基线（kattn/cnns.py）
- `cnn_transformer`：CNN(v→32→64→128) + TF(128h, 4head, 2层)，~334k 参数
- `cnn_transformer_pm`：CNN(v→32→64) + TF(64h, 4head, 2层)，~78k 参数（参数量匹配 KNET）

---

## 所有实验脚本

脚本均在 `src/simulation/`，需从该目录运行：

```bash
source env_setup.sh                  # 192.168.3.17 本机必须先执行

bash run_expA.sh                     # 超参数扫描（已完成）
bash run_expB_parallel.sh            # 学习曲线 Simu7（已完成）
bash run_expC.sh                     # 约束消融（已完成）
bash run_expD_192.sh                 # Simu16 KNET_rc（已完成，192.168.3.17）
bash submit_expD_cluster.sh          # Simu16 CNN-TF（已完成，集群）

python3 merge_results.py --inputs /path/a.csv /path/b.csv --output results/exp_results_merged.csv
python3 analyze_results.py --exp [A|B|C|D] --merged
```
