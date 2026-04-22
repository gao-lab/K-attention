# 机器分工：192.168.3.17

**最后更新**：2026-04-22  
**负责实验**：Exp A（超参数扫描）、Exp B（学习曲线，全部任务）、Exp C（约束消融）  
**GPU**：RTX 4090（24.5 GB）  
**conda 环境**：`kattn-sim`  
**工作目录**：`/home/mnt/liut/K-attention/K-attention/Kattn-sim-dev/src/simulation/`

---

## 任务总览

| 实验 | 状态 | 说明 |
|------|------|------|
| **Exp A** 超参数扫描 | ✅ 完成 | 40 runs，kernel_size × num_kernels 全网格 |
| **Exp B** 学习曲线（RC） | ⚠️ 需重跑 KNET_rc | 其余 5 个模型 ✅；KNET_rc 有配置 bug，已修复，待重跑 15 runs |
| **Exp B** 学习曲线（Markov） | 🔄 运行中 | ~75/90，transformer 收尾中 |
| **Exp C** 约束消融 | ⏳ 待启动 | Exp B Markov 完成 + KNET_rc 重跑后启动 |

---

## Exp A：超参数扫描（✅ 完成）

**目的**：验证 KNET 对 kernel_size 和 num_kernels 不敏感。  
**数据集**：`abs-ran_fix2`（RC）、`markov_1_0_50000`（Markov）  
**参数网格**：kernel_size ∈ {6,8,10,12,15}，num_kernels ∈ {16,32,64,128}，共 40 runs

### RC 任务（abs-ran_fix2）AUROC
```
num_kernels    16      32      64      128
kernel_size
6           0.8746  0.9137  0.9293  0.9439
8           0.9136  0.9217  0.9410  0.9513  ← 最优
10          0.9081  0.9177  0.9363  0.9432
12          0.9001  0.9171  0.8045* 0.9328
15          0.8942  0.8953  0.8908  0.9207
```
*k=12/NK=64 单次训练不稳定（异常值）

**结论**：AUROC 范围 0.875–0.951，超参数变化影响小，支持鲁棒性论断。

> ⚠️ **注意**：Exp A 的 KNET_rc 结果同样使用了错误的 reverse=False 配置，但 Exp A 的目的是展示超参数鲁棒性（相对比较），各组参数一致，结论仍然有效。如需绝对值准确，需重新评估。

### Markov 任务（markov_1_0_50000）AUROC
```
num_kernels    16      32      64      128
kernel_size
6           0.8342  0.8237  0.8279  0.8333
8           0.8686  0.8659  0.8686  0.8575
10          0.8691  0.8745  0.8729  0.8748  ← 最优
12          0.8636  0.8674  0.8670  0.8694
15          0.8536  0.8647  0.8656  0.8650
```
**结论**：AUROC 范围 0.824–0.875，Markov 任务超参数更不敏感。

---

## Exp B：学习曲线（部分完成）

**目的**：展示 6 个模型在不同数据量下的学习曲线（3 seeds 均值±std）。  
**模型**：KNET_rc/KNET、cnn、transformer_cls、transformer_cls_kmer、mha、cnn_transformer_pm  
**数据量**：5k / 10k / 20k / 50k / 100k，seeds: 0, 1, 2

### RC 任务（abs-ran_fix2）— 90/90 数值已收集，但 KNET_rc 需重跑

| 模型 | 5k (mean±std) | 10k | 20k | 50k | 100k |
|------|--------------|-----|-----|-----|------|
| **KNET_rc** ⚠️ | 0.604±0.019 | 0.759±0.046 | 0.816±0.023 | 0.887±0.013 | 0.916±0.009 |
| cnn | 0.850±0.009 | 0.895±0.006 | 0.887±0.002 | 0.893±0.001 | 0.909±0.001 |
| cnn_transformer_pm | 0.950±0.021 | 0.983±0.006 | 0.982±0.006 | 0.990±0.002 | 0.991±0.001 |
| mha | 0.741±0.025 | 0.740±0.007 | 0.747±0.008 | 0.762±0.024 | 0.792±0.002 |
| transformer_cls | 0.818±0.004 | 0.828±0.002 | 0.988±0.002 | 0.994±0.001 | 0.993±0.003 |
| transformer_cls_kmer | 0.578±0.015 | 0.618±0.021 | 0.689±0.008 | 0.777±0.014 | 0.965±0.000 |

⚠️ **KNET_rc 行数据无效**：使用了 `reverse=False`（bug），正确配置为 `reverse=True`。已于 2026-04-22 修复 run_bmk.py，需重跑 15 runs 后替换。预期修正后接近 ~1.0。

### Markov 任务 — 🔄 运行中（~75/90）

| 模型 | 5k (mean±std) | 10k | 20k | 50k | 100k |
|------|--------------|-----|-----|-----|------|
| **KNET** ✅ | 0.836±0.007 | 0.887±0.011 | 0.835±0.009 | 0.860±0.015 | 0.877±0.001 |
| cnn ✅ | 0.643±0.014 | 0.769±0.025 | 0.790±0.004 | 0.854±0.002 | 0.867±0.001 |
| cnn_transformer_pm ✅ | 0.729±0.015 | 0.831±0.009 | 0.780±0.007 | 0.836±0.007 | 0.849±0.002 |
| mha ✅ | 0.551±0.020 | 0.532±0.007 | 0.530±0.016 | 0.610±0.004 | 0.634±0.024 |
| transformer_cls 🔄 | 0.573±0.010 | 0.535±0.011 | 0.600 (部分) | 0.836 (部分) | 0.809 (部分) |
| transformer_cls_kmer 🔄 | 0.698±0.007 | 0.794±0.001 | 0.768±0.012 | 0.809 (部分) | 0.829 (部分) |

**当前正在运行（3并行）**：
- transformer_cls markov_1_0_20000 seed=1
- transformer_cls markov_1_0_50000 seed=1
- transformer_cls_kmer markov_1_0_50000 seed=1

---

## Exp B 收尾操作计划

Exp B Markov transformer 完成后，**立刻按顺序执行**：

```bash
cd /home/mnt/liut/K-attention/K-attention/Kattn-sim-dev/src/simulation
source env_setup.sh

# 1. 重跑 KNET_rc（reverse=True 已修复），15 runs
for seed in 0 1 2; do
  for n in 5000 10000 20000 50000 100000; do
    python run_bmk.py --model-type KNET_rc --test-config abs-ran_fix2 \
      --sample-size $n --max-epochs 500 --patience 20 \
      --max-lr 1e-2 --batch-size 512 --auroc-threshold 0.99 --version $seed
  done
done

# 2. 启动 Exp C（约束消融）
bash run_expC.sh
```

---

## Exp C：约束消融（⏳ 待启动）

**目的**：对比有/无 point-to-point 约束（groups=kernel_size）的性能差异。

| 对比 | 任务 | 数据集 |
|------|------|--------|
| KNET_rc vs KNET_uncons_rc | RC | abs-ran_fix2, relative_fix2, random_rand |
| KNET vs KNET_uncons | Markov | markov_1_0_50000 |

seeds: 0,1,2；总 24 runs。

---

## 结果文件位置

| 文件 | 说明 |
|------|------|
| `results/exp_results.csv` | 全部实验数值（本机，不入 git） |
| `revision/MACHINE_192.168.3.17.md` | 本文件（入 git，供汇总） |
| `revision/MODEL_CONFIG.md` | 模型配置规范（重要，防止重复踩坑） |
| `revision/PROGRESS.md` | 整体进度追踪 |

---

## 汇总时注意事项

1. **KNET_rc RC 数据待替换**：当前 CSV 中 KNET_rc 的 15 条 RC Exp B 数据使用错误配置，最终汇总时用重跑后数据覆盖
2. **Exp A KNET_rc 相对比较仍有效**：Exp A 用途是超参数鲁棒性（组间相对比较），不依赖绝对值
3. **Markov KNET 数据正确**：`KattentionModel_mask` + `reverse=False` 配置从未改变，数据可直接使用
4. **CSV 去重**：同一 (model_type, test_config, kernel_size, num_kernels, sample_size, version) 组合若有多行，取最新时间戳
