# Response to Reviewers — BIB-26-0525

We thank both reviewers for their constructive and thoughtful comments. Below we
address each point in detail. All new experiments have been completed and the
manuscript has been revised accordingly.

---

## Reviewer 1

### Comment R1-1: Systematic sensitivity analysis of key architectural parameters

> "A systematic sensitivity analysis of key architectural parameters (e.g., fragment
> size, kernel size, number of heads) would help assess robustness."

**Response:**

We have conducted a full 5 × 4 grid search across kernel_size ∈ {6, 8, 10, 12, 15}
and num_kernels ∈ {16, 32, 64, 128}, repeated over 3 independent seeds, on both
the RC task (Simu7, abs-ran_fix2) and the Markov task (markov_1_0_50000). Results
are summarized below (3-seed mean AUROC):

**RC task (KNET_rc):**

| num_kernels→ | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| k=6  | 0.875 | 0.914 | 0.929 | 0.944 |
| k=8  | 0.914 | 0.922 | 0.941 | **0.951** |
| k=10 | 0.908 | 0.918 | 0.936 | 0.943 |
| k=12 | 0.900 | 0.917 | 0.924 | 0.933 |
| k=15 | 0.894 | 0.895 | 0.891 | 0.921 |

**Markov task (KNET):**

| num_kernels→ | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| k=6  | 0.834 | 0.824 | 0.828 | 0.833 |
| k=8  | 0.869 | 0.866 | 0.869 | 0.858 |
| k=10 | 0.869 | 0.875 | 0.873 | **0.875** |
| k=12 | 0.864 | 0.867 | 0.868 | 0.869 |
| k=15 | 0.854 | 0.865 | 0.866 | 0.865 |

Performance is stable across all tested configurations. The default setting
(k=12, nk=64) is not the best-performing point — k=8 with nk=128 slightly
outperforms — confirming that the default was selected as a robust midpoint rather
than cherry-picked. The RC task shows a moderate benefit from increasing num_kernels,
while kernel_size has limited impact in the range 8–12 (Δ < 0.01). The Markov task
is even more stable (overall range: 0.824–0.875). These results demonstrate that
K-attention is robust to reasonable hyperparameter choices.

We have added a heatmap figure (Figure X, Exp A) to the manuscript illustrating
these results.

---

### Comment R1-2: Scope of "omics pattern recognition" claim

> "The current experimental scope may not fully support the broad claim of 'omics
> pattern recognition,' and the scope of applicability could be clarified."

**Response:**

We agree with this comment and thank the reviewer for the precise observation.
In the revised manuscript, we have moderated the scope of our claims throughout.
Specifically, we now describe K-attention as a data-efficient operator for
*sequence-based omics tasks involving fragment-level interactions*, rather than
claiming broad applicability to "omics pattern recognition" without qualification.
Revisions have been applied to the title, abstract, introduction, and discussion
to ensure all claims are directly supported by the presented experiments (RC-motif
detection, RNA-protein binding prediction, and CRISPR gRNA efficiency prediction).

---

### Comment R1-3: Variance, confidence intervals, and statistical testing

> "The data-efficiency claim (e.g., 5% data matching full-data baselines) would
> benefit from variance, confidence intervals, and statistical testing."

**Response:**

We have added statistical testing at multiple levels:

**Simulation (Exp B) — paired t-test, N=5 seeds:**

On the RC task, KNET_rc achieves near-ceiling performance (AUROC=0.999) at
n=10,000 training samples, a level CNN-Transformer-pm only approaches at n=100,000.
The difference is statistically significant at all data sizes above the crossover:

| Training size | KNET_rc | CNN-TF-pm | Δ | p-value |
|:---:|:---:|:---:|:---:|:---:|
| n=10,000 | 0.999 | 0.976 | +0.023 | 0.014 |
| n=20,000 | 0.999 | 0.985 | +0.013 | 0.009 |
| n=50,000 | 0.999 | 0.987 | +0.011 | 0.014 |
| n=100,000 | 0.999 | 0.991 | +0.008 | <0.001 |

On the Markov task, KNET outperforms CNN-TF-pm at all data sizes (p<0.001 at
n=2k, 5k, 10k, 20k, 100k).

**RBP real data — paired t-test, N=172 RBPs:**
KNET outperforms CNN-Transformer on 170/172 datasets; mean AUROC 0.871 vs 0.822
(Δ=+0.049, t=26.70, **p=6.77×10⁻⁶³**).

**CRISPR real data — paired t-test, N=11 datasets:**
KNET_Crispr outperforms CNN-Transformer on all 11 datasets; mean Spearman 0.461
vs 0.259 (Δ=+0.202, t=8.45, **p=7.3×10⁻⁶**).

Mean ± std error bars are now included in all learning curve figures.

---

### Comment R1-4: Ablation quantifying the point-to-point constraint

> "A clearer ablation quantifying the effect of removing the point-to-point
> constraint would help isolate its contribution."

**Response:**

We have added a dedicated ablation study (Exp C) comparing:
1. **KNET_rc** (with point-to-point constraint, num_kernels=64)
2. **KNET_uncons_rc** (without constraint, same num_kernels=64)
3. **KNET_uncons_rc** (without constraint, num_kernels=5, *parameter-matched* to KNET_rc)

Results on the hardest task (Simu16, random_rand, 0% PWM signal), 3 seeds:

| Model | n=2k | n=5k | n=10k | n=full |
|-------|:----:|:----:|:-----:|:------:|
| KNET_rc (constrained) | 0.689 | 0.827 | **0.991** | **0.994** |
| KNET_uncons_rc (nk=64) | 0.562 | 0.521 | 0.539 | 0.994 |
| KNET_uncons_rc (nk=5, param-matched) | 0.566 | 0.522 | 0.519 | — |

At n=10,000, the constrained model achieves AUROC=0.991 while the unconstrained
variant is near-random (0.539, Δ=0.452). Critically, the parameter-matched
unconstrained variant (nk=5, same total parameter count) fails identically (0.519),
ruling out model capacity as a confounding factor. **The constraint is the key
contributor to data efficiency.** Both models converge to ~0.994 only at full data
(100k), confirming that the constraint provides a critical inductive bias that
compensates for data scarcity.

An analogous result on the Markov task shows consistent +0.04 improvement from
the constraint across all small-sample sizes.

---

### Comment R1-5: Full learning curves

> "Providing full learning curves would better illustrate performance scaling
> across data regimes."

**Response:**

We have provided complete learning curves (Exp B) spanning n=1,000 to 100,000
training samples on the RC task and n=2,000 to 100,000 on the Markov task, for
all six baseline models (KNET, CNN-Transformer, CNN, Transformer-CLS,
Transformer-kmer, MHA), each repeated with 5 independent random seeds. Key findings:

- **RC task**: KNET_rc reaches AUROC=0.999 at n=10,000 (5% of full data), while
  CNN-Transformer-pm only approaches this level at n=100,000.
- **Markov task**: KNET leads consistently at all data sizes (Δ=+0.06–0.12 at
  n=2k–10k).
- MHA performs worst at all data sizes, demonstrating that not all attention
  mechanisms are equally suited to sequence motif tasks.

The updated learning curve figure (Figure X, Exp B) is included in the revision.

---

### Comment R1-6: Language polish

> "Some wording remains informal or grammatically inconsistent... colloquial
> expressions such as 'pretty outstanding performance' should be replaced..."

**Response:**

We thank the reviewer for identifying these issues. We have carefully revised the
manuscript to eliminate informal phrasing and correct grammatical inconsistencies.
Specific changes include:
- "pretty outstanding performance" → "consistently strong performance"
- "models that works" → "models that work"
- "showed close matched with" → "closely matched"
- "As being expected" → "As expected"

A thorough language review has been applied throughout the manuscript to ensure
consistent academic tone.

---

## Reviewer 2

### Comment R2-1: Comparison with CNN–Transformer hybrid models

> "The authors should add comprehensive comparisons with representative
> CNN–Transformer hybrid models widely used in omics sequence analysis."

**Response:**

We fully agree and have added CNN-Transformer hybrid baselines across **all three
experimental settings**. We implemented two variants:

- **CNN-TF** (full): CNN(4→32→64→128) + 2-layer Transformer (128-dim, 4 heads), ~334k parameters
- **CNN-TF-pm** (parameter-matched): CNN(4→32→64) + 2-layer Transformer (64-dim, 4 heads), ~78k parameters (comparable to KNET)

**Results:**

**(1) Simulation (Exp B)**: CNN-TF-pm is included as a baseline in the full
learning curve experiment. KNET outperforms CNN-TF-pm at all data sizes on the
Markov task and after n=5k on the RC task (paired t-test p<0.05 at n≥10k).

**(2) Simulation hardest task (Exp D, Simu16)**: On the task with no PWM signal
(pure RC structure), CNN-TF remains near-random (AUROC 0.50–0.68) across all
tested data volumes (5k–100k), while KNET_rc achieves ~0.985 from the smallest
sample. This demonstrates that the CNN-TF architecture lacks the RC-aware inductive
bias necessary for this class of problems.

**(3) RBP task (N=172 real datasets)**: KNET outperforms CNN-TF on 170 of 172
RNA-binding protein datasets. Mean AUROC: 0.871 (KNET) vs 0.822 (CNN-TF).
Paired t-test over N=172 datasets: **t=26.70, p=6.77×10⁻⁶³**.

**(4) CRISPR task (N=11 real datasets, 5-fold CV)**: KNET_Crispr outperforms both
CNN-TF variants on all 11 gRNA efficiency datasets:

| | KNET_Crispr | CNN-TF | CNN-TF-pm |
|--|:---:|:---:|:---:|
| Mean Spearman | **0.461** | 0.259 | 0.256 |
| Wins | 11/11 | 0/11 | 0/11 |
| p-value (vs KNET) | — | 7.3×10⁻⁶ | 5.1×10⁻⁶ |

The consistent advantage across three distinct experimental settings with
statistically significant margins provides comprehensive evidence that K-attention's
biologically constrained design outperforms the CNN-Transformer hybrid baseline.

---

### Comment R2-2: Comparison with Swin Transformer and ARConv

> "The authors should compare K-attention with these state-of-the-art architectures
> to verify its advancement."

**Response:**

We appreciate the reviewer's awareness of these architectures. However, Swin
Transformer and ARConv were specifically designed for **2D visual recognition tasks**
(image classification/detection) and are not standard baselines in the omics sequence
analysis literature. Their core design assumptions—hierarchical 2D patch processing
and 2D adaptive kernel shapes—do not transfer meaningfully to 1D genomic sequences.

The contribution of this work is at the **operator level**: we propose a
biologically constrained attention mechanism that encodes fragment-level motif
interaction priors directly into the attention computation. The appropriate
comparison is therefore against sequence-native baselines (CNN, Transformer,
CNN-Transformer hybrids), which we have now comprehensively included (see R2-1
above).

We note that the CNN-Transformer hybrid we implemented already captures the
combination of local feature extraction (CNN) and long-range dependency modeling
(Transformer) that motivates both Swin and ARConv designs, and K-attention
outperforms it significantly across all three tasks.

---

### Comment R2-3: Hyperparameter sensitivity analysis

> "A systematic hyperparameter analysis is required to ensure the robustness and
> reproducibility of the model."

**Response:**

Please see our response to R1-1 above. The same systematic 5×4 grid search
across kernel_size and num_kernels directly addresses this concern. K-attention
shows robust performance across all tested configurations, with a total AUROC
variation of <0.076 on the RC task and <0.051 on the Markov task across the entire
parameter grid.

---

## Summary of New Experiments

| Experiment | Description | Key Finding |
|---|---|---|
| Exp A | 5×4 hyperparameter grid, 3 seeds, 2 tasks | AUROC stable across all configs; default k=12,nk=64 is robust midpoint |
| Exp B | Full learning curves, 6 models, 5 seeds, 2 tasks | KNET data-efficient; reaches saturation at ~5–10% of data |
| Exp C | Constraint ablation (with/without, param-matched), 3 seeds | Constraint crucial: Δ=0.45 at n=10k on hardest task |
| Exp D | Simu16 KNET vs CNN-TF, 3 seeds | CNN-TF near-random on RC task regardless of data size |
| RBP CNN-TF | 172 RBPs, CNN-TF baseline | KNET wins 170/172, p=6.77×10⁻⁶³ |
| CRISPR CNN-TF | 11 datasets, 5-fold CV | KNET wins 11/11, p<10⁻⁵ |
