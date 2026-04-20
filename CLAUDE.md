# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 项目背景

**K-attention**（BIB-26-0525，投稿至 *Briefings in Bioinformatics*）提出一种面向 omics 序列分析的生物约束注意力算子，在三个任务上验证：RC/Markov 模拟实验、RNA-RBP 结合预测（172个RBP）、CRISPR gRNA 效率预测（11个数据集）。论文当前处于**大修阶段（revision）**，本工作目录的核心任务是**补充实验、回应审稿人意见**。

### 当前 Revision 目标（按优先级）

| 编号 | 审稿人 | 要求 | 状态 |
|------|--------|------|------|
| A | R1-1/R2-3 | **超参数敏感性扫描**（kernel_size × num_kernels on abs-ran_fix2 + markov_1_0_50000）| ✅ 正在运行 `run_expA.sh` |
| B | R1-3/R1-5/R2-1 | **完整学习曲线**（多数据量 × 多seeds，含 cnn_transformer_pm 新基线）| 待 Exp A 完成后启动 `run_expB.sh` |
| C | R1-4 | **点对点约束消融**（KNET vs KNET_uncons on RBP）| 结果可复用，待整理 |
| D | R1-2 | 缩小论断范围（"omics pattern recognition"）| 文字修改 |
| E | R1-6 | 语言润色（口语表达）| 文字修改 |
| F | R2-2 | 回复 Swin/ARConv 无需新实验（2D视觉架构非 omics 标准基线）| 文字回复 |

---

## 环境配置

**重要：kattn-sim conda 环境内置的路径变量均指向 Docker 容器路径 `/Kattn-sim-dev`，在本机须覆盖。**

```bash
# 激活环境后必须 source 以下脚本修正路径
cd /home/mnt/liut/K-attention/K-attention/Kattn-sim-dev/src/simulation
source env_setup.sh

# env_setup.sh 设置的变量：
# KATTN_SRC_DIR=/home/mnt/liut/K-attention/K-attention/Kattn-sim-dev/src
# KATTN_RESOURCES_DIR=/home/mnt/liut/K-attention/K-attention/Kattn-sim-dev/resources
# HF_DATASETS_CACHE=/home/mnt/liut/K-attention/K-attention/Kattn-sim-dev/.hf_cache
# HF_DATASETS_OFFLINE=1
# PYTHONPATH=$KATTN_SRC_DIR:...
```

```bash
# RBP 实验环境（单独）
conda activate kattn-rbp
```

---

## 常用命令

### 模拟数据训练（工作目录：`Kattn-sim-dev/src/simulation/`）

```bash
source env_setup.sh   # 必须先执行

# 基础训练（K-attention主模型）
python run_bmk.py --model-type KNET --test-config abs-ran_fix2 \
    --kernel-size 12 --num-kernels 64 --max-lr 1e-4 --version 0

# 可用 model-type：
#   KNET (=KattentionModel_mask, 模拟主模型), KNET_uncons (消融)
#   cnn, transformer_cls, transformer_cls_kmer, mha
#   cnn_transformer, cnn_transformer_pm (新增CNN-TF hybrid)

# 预处理缓存数据集（必须在训练前执行一次）
python run_bmk.py --model-type KNET --test-config abs-ran_fix2 --cache-run

# 训练时子采样（学习曲线用）
python run_bmk.py --model-type KNET --test-config abs-ran_fix2 --sample-size 5000

# 超参数扫描
python run_bmk.py --model-type KNET --test-config abs-ran_fix2 \
    --kernel-size 6 --num-kernels 16

# 批量实验脚本
bash run_expA.sh    # 超参数扫描
bash run_expB.sh    # 学习曲线（Exp A 完成后运行）

# 查看实验结果
python analyze_results.py --exp A
python analyze_results.py --exp B
# 结果 CSV 位于：../../results/exp_results.csv
```

### RBP 任务训练（工作目录：`Kattention_aten_test/scripts/RBP/`）

```bash
python -u main.py <data> <kernel_size> <num_kernels> <seed> <model> <optimizer>
# model 可选：KNET_plus_seq2、KNET_plus_ic、KNET_uncons_seq2 等
```

---

## Simu1-16 数据集映射

论文中的 Simu1-16 对应关系（Feature 为外层分组，Position 为内层）：

| Simu | Config name | Feature | Position |
|------|-------------|---------|----------|
| 1-4  | {absolute,random,abs-ran,relative}_pwm | pure PWM | 4种 |
| 5-8  | {absolute,random,abs-ran,relative}_fix2 | 40% PWM | 4种 |
| 9-12 | {absolute,random,abs-ran,relative}_fix1 | 20% PWM | 4种 |
| 13-16| {absolute,random,abs-ran,relative}_rand | 0% PWM | 4种 |

**Simu7 = `abs-ran_fix2`**：学习曲线代表数据集（中等难度：CNN=0.966, KNET=1.0）。
`abs-ran_fix1.fa`（Simu11）文件缺失，可用 `simu_main.py --position abs-ran --feature fix1` 生成。

---

## 代码架构

```
K-attention/
└── K-attention/
    ├── Kattn-sim-dev/            # 模拟 + CRISPR 模块
    │   ├── resources/            # FASTA 数据（RC 全量 + Markov 多数据量版本）
    │   ├── .hf_cache/            # HuggingFace datasets 本地缓存
    │   └── src/
    │       ├── kattn/            # 核心算子库
    │       └── simulation/       # 模拟任务脚本（训练 + 实验批量脚本）
    └── Kattention_aten_test/     # RBP 预测模块
        └── scripts/RBP/
```

### 核心算子：`kattn/kattention.py`

**KattentionV4** 是论文主体实现：
- `torch.unfold` 展开为局部窗口（感受野 = kernel_size = motif 长度）
- `groups=kernel_size` 分组卷积强制 point-to-point 约束
- 接口：`KattentionV4(channel_size, kernel_size, num_kernels, reverse=False)`
- **forward 返回 dict**，用 `output["attn_logits"]` 取输出

**模拟实验用类（重要区分）：**

| 类名 | 用途 | forward 签名 |
|------|------|-------------|
| `KattentionModel_mask` | 模拟主模型（有约束+对角线mask）→ model_type=`KNET` | `(input_ids, cls_labels, key_padding_mask=None)` |
| `KattentionModel_uncons_mask` | 消融版→ model_type=`KNET_uncons` | 同上 |
| `KNET`（line 1131） | **RBP 专用**，有4路并行CNN，不用于模拟实验 | `(input_ids, key_padding_mask, cls_labels)` 必须 |
| `KNET_uncons`（line 1184） | RBP 消融专用 | 同上 |

### CNN-TF Hybrid baseline（新增，`kattn/cnns.py`）

| 模型类 | model-type | 参数量(vocab=7) | 结构 |
|--------|-----------|----------------|------|
| `CNNTransformerModel` | `cnn_transformer` | ~334k | CNN(v→32→64→128) + TF(128h, 4head, 2层) |
| `CNNTransformerModelMatched` | `cnn_transformer_pm` | ~78k | CNN(v→32→64) + TF(64h, 4head, 2层) |

### 数据加载

- 注册表：`kattn/general_fasta/general_fasta.py` 中的 `FASTA_FILES_DIRS` 字典
- 训练缓存：`simulation/_cache_dsts/<config_name>/` 目录（自动生成）
- subsample 缓存：`simulation/_cache_dsts/<config_name>_n<N>/` 目录

---

## 重要实现注意事项

- `KattentionV4.forward` 返回 **dict**（非 tensor），必须通过 `output["attn_logits"]` 取值
- `KattentionModel_uncons_mask.forward` 内有 `mask.cuda()` 硬编码，确保 GPU 可用
- `modules.py` 中 `BaseClassifier_reg = BaseClassifier`（别名，支持 CRISPR 模块导入）
- Markov 任务 KNET 使用 `KattentionModel_mask` 的对角线 mask（强调相邻偏移对，对应一阶 Markov 先验）
- HuggingFace datasets 离线模式（`HF_DATASETS_OFFLINE=1`）：数据必须先 `--cache-run` 预处理
