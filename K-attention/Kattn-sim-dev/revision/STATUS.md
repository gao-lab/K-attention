# Revision Status Snapshot — BIB-26-0525
*Last updated: 2026-04-30 (RC Tasks 1/2 complete, Markov in progress)*

This file is a machine-readable snapshot for agents picking up this work.
Full detail: `revision/PROGRESS.md`

---

## Experiment Status

| Exp | Description | Status | Seeds done | Pending action |
|-----|-------------|--------|-----------|----------------|
| A | Hyperparameter scan (kernel_size × num_kernels) | ✅ Done | seeds 0,1,2 (120 runs) | — |
| B | Learning curves (abs-ran_fix2 + markov_1_0) | ✅ Done | seeds 0–4 | — |
| C | Constraint ablation (KNET vs KNET_uncons) | ✅ Done | seeds 0–4 | — |
| D | Simu16 learning curves (KNET_rc vs CNN-TF) | ✅ Done | seeds 0–2 | — |
| H | RC multi-difficulty (random_fix2/fix1/rand) | 🔄 partial | fix2:177/180, fix1:180/180, rand:108/148+ | Wait for rand completion + CNN/MHA on rand |
| I | Markov multi-entropy (0_75/1_0/1_25) | 🔄 partial | 1_0:complete; 0_75/1_25: only 3/6 models | Wait for KNET/cnn_transformer_pm/transformer_cls |
| E | Narrow claims ("omics pattern recognition") | ⏳ Pending | — | Text revision only |
| F | Language polish | ⏳ Pending | — | Text revision only |
| G | Reply re Swin/ARConv | ⏳ Pending | — | Text reply only |

---

## Data Files

| File | Location | Rows | Notes |
|------|----------|------|-------|
| `exp_results_merged.csv` | `results/exp_results_merged.csv` | 855 | All experiments, 2026-04-30 merge |

---

## Figures

All figures at `revision/figures/`. **Exp B and C need re-plotting with new 5-seed data.**

| File | Status | Last generated |
|------|--------|----------------|
| `exp_a_heatmap.pdf/png` | ✅ Final (3-seed mean) | 2026-04-24 |
| `exp_b_learning_curve.pdf/png` | ✅ Final (5 seeds, n=1k–100k) | 2026-04-24 |
| `exp_c_ablation.pdf/png` | ✅ Final (small-sample learning curve) | 2026-04-24 |
| `exp_d_simu16.pdf/png` | ✅ Final (3 seeds) | 2026-04-24 |

---

## Key Results Summary

### Exp A — Hyperparameter sensitivity (3-seed mean)

**RC task (KNET_rc, abs-ran_fix2):**

| num_kernels→ | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| k=6 | 0.875 | 0.914 | 0.929 | 0.944 |
| k=8 | 0.914 | 0.922 | 0.941 | **0.951** |
| k=10 | 0.908 | 0.918 | 0.936 | 0.943 |
| k=12 | 0.900 | 0.917 | 0.924 | 0.933 |
| k=15 | 0.894 | 0.895 | 0.891 | 0.921 |

**Markov task (KNET, markov_1_0_50000):**

| num_kernels→ | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| k=6 | 0.834 | 0.824 | 0.828 | 0.833 |
| k=8 | 0.869 | 0.866 | 0.869 | 0.858 |
| k=10 | 0.869 | 0.875 | 0.873 | **0.875** |
| k=12 | 0.864 | 0.867 | 0.868 | 0.869 |
| k=15 | 0.854 | 0.865 | 0.866 | 0.865 |

### Exp B — Learning curves (5-seed mean AUROC)

**RC (abs-ran_fix2):**

| Model | n=1k | n=2k | n=5k | n=10k | n=20k | n=50k | n=100k |
|-------|------|------|------|-------|-------|-------|--------|
| KNET_rc | 0.656 | 0.730 | 0.916* | **0.999** | **0.999** | **0.999** | **0.999** |
| cnn_transformer_pm | 0.631 | 0.846 | 0.942 | 0.976 | 0.985 | 0.987 | 0.991 |
| transformer_cls | 0.800 | 0.825 | 0.819 | 0.829 | 0.988 | 0.993 | 0.993 |
| cnn | 0.715 | 0.799 | 0.852 | 0.898 | 0.887 | 0.893 | 0.909 |
| transformer_cls_kmer | 0.595 | 0.517 | 0.602 | 0.621 | 0.690 | 0.782 | 0.965 |
| mha | 0.639 | 0.714 | 0.718 | 0.743 | 0.743 | 0.765 | 0.795 |

*KNET_rc n=5k: seed=1 anomaly (0.595); other 4 seeds ~0.995

**Markov:**

| Model | n=2k | n=5k | n=10k | n=20k | n=50k | n=100k |
|-------|------|------|-------|-------|-------|--------|
| KNET | 0.677 | 0.822 | **0.888** | **0.835** | 0.859 | 0.878 |
| cnn_transformer_pm | 0.553 | 0.728 | 0.825 | 0.783 | 0.840 | 0.852 |
| transformer_cls | 0.574 | 0.564 | 0.542 | 0.659 | 0.830 | 0.826 |
| cnn | 0.548 | 0.640 | 0.780 | 0.789 | 0.854 | 0.868 |
| transformer_cls_kmer | 0.538 | 0.707 | 0.809 | 0.772 | 0.813 | 0.826 |
| mha | 0.590 | 0.549 | 0.537 | 0.535 | 0.609 | 0.625 |

### Exp C — Constraint ablation (small-sample, random_rand)

**RC task (3 seeds):**

| Model | n=2k | n=5k | n=10k | n=full |
|-------|------|------|-------|--------|
| KNET_rc (constrained, nk=64) | 0.689 | 0.827 | **0.991** | **0.994** |
| KNET_uncons_rc (nk=64, 12×params) | 0.562 | 0.521 | 0.539 | 0.994 |
| KNET_uncons_rc (nk=5, param-matched) | 0.566 | 0.522 | 0.519 | — |

**Markov task (3 seeds):**

| Model | n=2k | n=5k | n=10k |
|-------|------|------|-------|
| KNET (constrained, nk=64) | 0.677 | 0.822 | **0.888** |
| KNET_uncons (nk=64) | 0.638 | 0.779 | 0.847 |
| KNET_uncons (nk=5) | 0.614 | 0.737 | 0.865 |

### Exp D — Simu16 learning curves (random_rand)

| n | KNET_rc | CNN-TF (matched) | CNN-TF (full) |
|---|---------|-----------------|---------------|
| 5,000 | **0.985** | 0.505 | 0.522 |
| 10,000 | **0.991** | 0.541 | 0.510 |
| 20,000 | **0.992** | 0.530 | 0.511 |
| 50,000 | **0.993** | 0.568 | 0.553 |
| 100,000 | **0.994** | 0.683 | 0.564 |

---

## Infrastructure

| Item | Detail |
|------|--------|
| Main compute | 192.168.3.17 (liut), RTX 4090, direct SSH |
| Cluster | luminary (172.18.18.7), via bastion 172.18.18.11, Slurm |
| Conda env | `kattn-sim` at `/rd1/liut/miniconda3/envs/kattn-sim/` (on 192.168.3.17) |
| Results CSV | `results/exp_results_merged.csv` (local, 855 rows) |
| Analysis script | `src/simulation/analyze_results.py` (updated 2026-04-30 for multi-task) |
| Merge script | `src/simulation/merge_results.py` |
| Figures | `revision/figures/` (local) |
| Plot script | `src/simulation/plot_revision.py` (use kattn-sim Python on 192.168.3.17) |

---

## Immediate Next Actions

1. **Check cluster progress**: Markov H=0.75/1.25 KNET/cnn_transformer_pm/transformer_cls still running (`squeue -u liut` on luminary)
2. **Re-merge + re-analyze** after new data arrives: `python3 analyze_results.py --merged --completeness`
3. **Generate figures** for new experiments: Exp H (RC multi-difficulty), Exp I (Markov multi-entropy)
4. **Text revisions E/F/G** — manuscript edits
   - E: Narrow claims to "omics pattern recognition"
   - F: Language polish
   - G: Reply re Swin/ARConv (2D visual architecture, not omics standard baseline)
