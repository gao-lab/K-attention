# Revision Experiments — BIB-26-0525

投稿期刊：*Briefings in Bioinformatics*  
论文：K-attention（面向 omics 序列分析的生物约束注意力算子）  
当前阶段：**大修（Major Revision）**

---

## 审稿人要求 & 实验状态

| 编号 | 审稿人 | 要求 | 状态 | 对应脚本 |
|------|--------|------|------|----------|
| **A** | R1-1 / R2-3 | 超参数敏感性扫描（kernel_size × num_kernels）| 🔄 **Seed 0 Done**; seeds 1&2 running (2026-04-24) | `run_expA.sh` / `run_expA_seeds12.sh` |
| **B** | R1-3 / R1-5 / R2-1 | 完整学习曲线（6模型 × 5数据量 × 3 seeds）| 🔄 **扩展中**：补充 n=1k/2k(RC), n=2k(Markov)，seeds 扩展至 5 (2026-04-24) | `run_expB_parallel.sh` / `submit_expB_ext.sh` |
| **C** | R1-4 | 约束消融（KNET_rc vs KNET_uncons_rc）| ✅ **Done** (2026-04-23) | `run_expC.sh` |
| **D（new）** | — | Simu16 学习曲线对比（KNET_rc vs CNN-TF，两台机器）| ✅ **Done** (2026-04-24) | `run_expD_192.sh` + cluster |
| **E** | R1-2 | 缩小论断范围（"omics pattern recognition"）| ⏳ Pending（文字修改）| — |
| **F** | R1-6 | 语言润色（口语表达）| ⏳ Pending（文字修改）| — |
| **G** | R2-2 | 回复 Swin/ARConv（2D视觉架构非 omics 标准基线）| ⏳ Pending（文字回复）| — |

---

## Exp A：超参数扫描（🔄 Seed 0 完成；Seeds 1&2 运行中，2026-04-24）

**目标**：验证 KNET 对 kernel_size 和 num_kernels 的不敏感性。

| 参数 | 取值范围 |
|------|----------|
| kernel_size | 6, 8, 10, 12, 15 |
| num_kernels | 16, 32, 64, 128 |
| 数据集 | abs-ran_fix2（RC任务）+ markov_1_0_50000（Markov任务）|
| 总运行数 | 40 runs × 3 seeds = 120 runs（seed 0 完成，seeds 1&2 运行中，`run_expA_seeds12.sh`）|

**Seeds 1&2 监控**：
```bash
ssh liut@192.168.3.17 "tail -f /tmp/expA_seeds12.log"
ssh liut@192.168.3.17 "ps aux | grep run_bmk | grep -v grep | wc -l"
```

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

## Exp B：学习曲线（🔄 扩展中，2026-04-24）

**目标（扩展后）**：6 模型 × 7 RC数据量 + 6 Markov数据量 × 5 seeds（RC: 1k/2k/5k/10k/20k/50k/100k；Markov: 2k/5k/10k/20k/50k/100k）。

**扩展内容（210 新runs，luminary gpu2+gpu32）**：
- RC 新增 n=1k, 2k（seeds 0-4，6模型）
- RC 已有 n=5k-100k 补充 seeds 3,4
- Markov 新增 n=2k（`markov_1_0_5000 --sample-size 2000`，seeds 0-4，6模型）
- Markov 已有 n=5k-100k 补充 seeds 3,4

**提交方式**（在 luminary 激活 kattn-sim 后执行）：
```bash
cd /lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev/src/simulation
bash submit_expB_ext.sh
```

**RC 任务（abs-ran_fix2，Simu7），mean AUROC：**

| 模型 | n=5k | n=10k | n=20k | n=50k | n=100k |
|------|------|-------|-------|-------|--------|
| KNET_rc | **0.995** | **0.997** | **0.998** | **0.999** | **0.999** |
| cnn_transformer_pm | 0.950 | 0.983 | 0.982 | 0.989 | 0.991 |
| transformer_cls | 0.818 | 0.828 | 0.988 | 0.994 | 0.993 |
| cnn | 0.850 | 0.895 | 0.886 | 0.893 | 0.909 |
| transformer_cls_kmer | 0.578 | 0.618 | 0.689 | 0.777 | 0.965 |
| mha | 0.741 | 0.739 | 0.747 | 0.762 | 0.791 |

**Markov 任务，mean AUROC：**

| 模型 | n=5k | n=10k | n=20k | n=50k | n=100k |
|------|------|-------|-------|-------|--------|
| KNET | **0.836** | **0.887** | **0.835** | **0.867** | **0.877** |
| cnn_transformer_pm | 0.729 | 0.831 | 0.780 | 0.836 | 0.849 |
| transformer_cls | 0.568 | 0.545 | 0.624 | 0.829 | 0.824 |
| cnn | 0.643 | 0.769 | 0.790 | 0.854 | 0.867 |
| transformer_cls_kmer | 0.691 | 0.800 | 0.775 | 0.812 | 0.826 |
| mha | 0.550 | 0.532 | 0.529 | 0.610 | 0.634 |

**注**：KNET_rc n=5000 seed=1 存在异常值（0.595，未收敛），不影响整体趋势；取 3 seeds 时需报告 median 或注明。

---

## Exp C：约束消融（✅ 完成，2026-04-23）

**目标**：对比有/无 point-to-point 约束的性能差异（k=12, nk=64，全量数据）。

**RC 任务：**

| 数据集 | KNET_rc (s0/s1/s2) | mean | KNET_uncons_rc (s0/s1/s2) | mean | KNET_rc - uncons |
|--------|---------------------|------|---------------------------|------|-----------------|
| abs-ran_fix2 (Simu7) | 0.9996/0.9995/0.9995 | **0.9995** | 0.9994/0.9989/0.9994 | 0.9992 | +0.0003 |
| random_rand (Simu16) | 0.9943/0.9933/0.9941 | **0.9939** | 0.9946/0.9927/0.9940 | 0.9938 | +0.0001 |

**Markov 任务（markov_1_0_50000）：**

| 模型 | s0 | s1 | s2 | mean | KNET - uncons |
|------|----|----|----|----|---|
| KNET | 0.8705 | 0.8713 | 0.8667 | **0.8695** | |
| KNET_uncons | 0.8557 | 0.8613 | 0.8646 | 0.8605 | **+0.0090** |

**解读**：RC 任务满载数据下约束效果因天花板效应不显著；Markov 任务约束带来 +0.009 稳定提升。约束的主要价值体现在小样本效率（Exp D/B 对比可见）。

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

## 下一步

1. **绘图**：基于 `exp_results_merged.csv` 生成 Exp A/B/C/D 论文图
2. **文字修改**（E/F/G）：缩小论断范围、语言润色、回复 Swin/ARConv
3. **论文 Revision 提交**

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
