# Revision Experiments — BIB-26-0525

投稿期刊：*Briefings in Bioinformatics*  
论文：K-attention（面向 omics 序列分析的生物约束注意力算子）  
当前阶段：**大修（Major Revision）**

---

## 审稿人要求 & 实验状态

| 编号 | 审稿人 | 要求 | 状态 | 对应脚本 |
|------|--------|------|------|----------|
| **A** | R1-1 / R2-3 | 超参数敏感性扫描（kernel_size × num_kernels）| ✅ **Done** (2026-04-21) | `run_expA.sh` |
| **B** | R1-3 / R1-5 / R2-1 | 完整学习曲线（6模型 × 5数据量 × 3 seeds）| 🔄 **Running** (~Apr 22 19:00) | `run_expB_parallel.sh` |
| **C** | R1-4 | 约束消融（KNET_rc vs KNET_uncons_rc）| ⏳ Pending（Exp B 后启动）| `run_expC.sh` |
| **D** | R1-2 | 缩小论断范围（"omics pattern recognition"）| ⏳ Pending（文字修改）| — |
| **E** | R1-6 | 语言润色（口语表达）| ⏳ Pending（文字修改）| — |
| **F** | R2-2 | 回复 Swin/ARConv（2D视觉架构非 omics 标准基线）| ⏳ Pending（文字回复）| — |

---

## Exp A：超参数扫描（✅ 完成）

**目标**：验证 KNET 对 kernel_size 和 num_kernels 的不敏感性。

| 参数 | 取值范围 |
|------|----------|
| kernel_size | 6, 8, 10, 12, 15 |
| num_kernels | 16, 32, 64, 128 |
| 数据集 | abs-ran_fix2（RC任务）+ markov_1_0_50000（Markov任务）|
| 总运行数 | 40 runs（各 1 seed，lr=1e-2）|

**当前结果摘要（完整见 `results/exp_results.csv`）：**
- RC (abs-ran_fix2)：AUROC 范围 0.875–0.951，k=8/NK=128 最优
- Markov (markov_1_0_50000)：AUROC 范围 0.824–0.875，k=10/NK=128 最优
- 结论：超参数在合理范围内变化对性能影响较小（支持鲁棒性论断）

**查看结果：**
```bash
cd src/simulation && python analyze_results.py --exp A
```

---

## Exp B：学习曲线（🔄 运行中）

**目标**：展示 6 个模型在不同数据量下的 AUROC 学习曲线，3个随机 seed。

| 参数 | 取值 |
|------|------|
| 模型 | KNET_rc/KNET, cnn, transformer_cls, transformer_cls_kmer, mha, cnn_transformer_pm |
| 数据量 | 5k, 10k, 20k, 50k, 100k |
| Seeds | 0, 1, 2 |
| 总运行数 | RC 90 + Markov 90 = 180 runs |

**进度（截至 2026-04-21 20:30 CST）：**
- RC 完成：~68/90（KNET_rc, cnn, mha, cnn_transformer_pm 全部完成；transformer 运行中）
- Markov 完成：81/90（KNET, cnn, mha, cnn_transformer_pm 全部完成；transformer 1/30）
- **当前正在运行**：`run_expB_parallel.sh`（3并行 transformer jobs，batch=128）
- 日志：`/tmp/expB_parallel.log`
- **预计完成**：2026-04-22 ~19:00 CST

**监控运行状态：**
```bash
ps aux | grep run_bmk | grep -v grep
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
tail -f /tmp/expB_parallel.log
```

**查看结果：**
```bash
cd src/simulation && python analyze_results.py --exp B
```

---

## Exp C：约束消融（⏳ Exp B 后启动）

**目标**：对比有/无 point-to-point 约束的 KNET 性能差异。

| 对比 | 任务 | 数据集 |
|------|------|--------|
| KNET_rc vs KNET_uncons_rc | RC | abs-ran_fix2, relative_fix2, random_rand |
| KNET vs KNET_uncons | Markov | markov_1_0_50000 |

Seeds: 0,1,2；总运行数：3×2×3 + 1×2×3 = 24 runs

**启动命令：**
```bash
cd src/simulation && bash run_expC.sh
```

---

## 关键修改说明（相对原始代码）

### run_bmk.py 修改点
1. **新增 model_type**：`KNET_rc`（RC任务，`KattentionModel`）、`KNET_uncons_rc`、`KNET`（Markov，`KattentionModel_mask`）、`KNET_uncons`
2. **AUROCThresholdStop callback**：val_auroc ≥ 0.99 时提前终止
3. **新增参数**：`--batch-size`（默认512）、`--auroc-threshold`（默认0.99）
4. **CSV header fix**：`result_csv.stat().st_size == 0` 判断空文件

### 模型分工（重要）
- RC 任务 → `KNET_rc`（`KattentionModel`，无 mask）
- Markov 任务 → `KNET`（`KattentionModel_mask`，±2对角线 band mask）
- lr=1e-2 for all KNET variants（原 Snakefile 验证值，1e-4 会导致 AUROC ~0.8）

### 新增基线模型（kattn/cnns.py）
- `cnn_transformer_pm`：CNN(v→32→64) + TF(64h, 4head, 2层)，~78k 参数，参数量匹配 KNET

---

## 所有实验脚本位置

所有脚本均在 `src/simulation/` 目录，需从该目录运行：

```bash
cd /home/mnt/liut/K-attention/K-attention/Kattn-sim-dev/src/simulation
source env_setup.sh

bash run_expA.sh             # 超参数扫描（已完成）
bash run_expB.sh             # 学习曲线（原始，完整版）
bash run_expB_parallel.sh    # 学习曲线（当前运行，3并行，跳过已完成）
bash run_expC.sh             # 约束消融
bash run_validate.sh         # 快速验证（n=20000）

python analyze_results.py --exp A   # 查看 Exp A 结果
python analyze_results.py --exp B   # 查看 Exp B 结果
```

---

## 结果文件

| 文件 | 说明 |
|------|------|
| `results/exp_results.csv` | 所有实验结果（CSV），**不入 git**，本机运行产生 |
| `results/checkpoint/` | 模型权重，**不入 git**（5.3GB+） |
| `src/simulation/tb_logs/` | TensorBoard 日志，**不入 git** |

---

*下一步*：Exp B 完成后，运行 `python analyze_results.py --exp B` 生成学习曲线数据，然后启动 `run_expC.sh`。
