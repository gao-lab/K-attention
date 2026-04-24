# Revision Status Snapshot — BIB-26-0525
*Last updated: 2026-04-24*

This file is a machine-readable snapshot for agents picking up this work.
Full detail: `revision/PROGRESS.md`

---

## Experiment Status

| Exp | Description | Status | Seeds done | Pending action |
|-----|-------------|--------|-----------|----------------|
| A | Hyperparameter scan (kernel_size × num_kernels) | 🔄 Partial | seed 0 (40 runs) | Wait for seeds 1&2 (80 runs, running on 192.168.3.17 via `run_expA_seeds12.sh`) |
| B | Learning curves (6 models × 7 RC sizes + 6 Markov sizes × 5 seeds) | 🔄 Extending | seeds 0-2 done; seeds 3-4 + new sizes (RC 1k/2k, Markov 2k) pending on luminary | Run `bash submit_expB_ext.sh` on luminary |
| C | Constraint ablation (KNET vs KNET_uncons) | ✅ Done | seeds 0,1,2 | — |
| D | Simu16 learning curves (KNET_rc vs CNN-TF) | ✅ Done | seeds 0,1,2 | — |
| E | Narrow claims ("omics pattern recognition") | ⏳ Pending | — | Text revision only |
| F | Language polish | ⏳ Pending | — | Text revision only |
| G | Reply re Swin/ARConv | ⏳ Pending | — | Text reply only |

---

## Data Files

| File | Location | Rows | Notes |
|------|----------|------|-------|
| `exp_results_merged.csv` | `results/exp_results_merged.csv` | 278 | Seed 0 only complete for Exp A |
| Remote results (192.168.3.17) | `/rd1/liut/K-attention/K-attention/Kattn-sim-dev/results/exp_results.csv` | TBD | Will contain seeds 1&2 when done |

**After Exp A seeds 1&2 finish:**
1. Retrieve remote CSV: SSH to 192.168.3.17, copy `/rd1/liut/.../results/exp_results.csv`
2. Merge: `python3 src/simulation/merge_results.py --inputs results/exp_results_merged.csv <remote>.csv --output results/exp_results_merged.csv`
3. Re-run: `python3 src/simulation/plot_revision.py --exp A` (on 192.168.3.17 or with kattn-sim env)
4. Re-transfer figures: `revision/figures/exp_a_heatmap.{pdf,png}`

---

## Figures

All 8 figures generated and stored locally at `revision/figures/`:

| File | Status | Last generated |
|------|--------|----------------|
| `exp_a_heatmap.pdf/png` | ✅ Seed 0 only (will need re-plot after seeds 1&2) | 2026-04-24 |
| `exp_b_learning_curve.pdf/png` | ✅ Final (3 seeds) | 2026-04-24 |
| `exp_c_ablation.pdf/png` | ✅ Final (3 seeds) | 2026-04-24 |
| `exp_d_simu16.pdf/png` | ✅ Final (3 seeds) | 2026-04-24 |

---

## Key Results Summary

### Exp A — Hyperparameter sensitivity (seed 0 only)

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

*Seeds 1&2 will provide error bars; re-plot needed.*

### Exp B — Learning curves (mean AUROC, 3 seeds)

**RC (Simu7, abs-ran_fix2):**

| Model | n=5k | n=10k | n=20k | n=50k | n=100k |
|-------|------|-------|-------|-------|--------|
| KNET_rc | **0.995** | **0.997** | **0.998** | **0.999** | **0.999** |
| CNN-TF (matched) | 0.950 | 0.983 | 0.982 | 0.989 | 0.991 |
| Transformer | 0.818 | 0.828 | 0.988 | 0.994 | 0.993 |
| CNN | 0.850 | 0.895 | 0.886 | 0.893 | 0.909 |
| Transformer (k-mer) | 0.578 | 0.618 | 0.689 | 0.777 | 0.965 |
| MHA | 0.741 | 0.739 | 0.747 | 0.762 | 0.791 |

*Anomaly excluded: KNET_rc n=5000 seed=1 (AUROC=0.595)*

**Markov (markov_1_0_N):**

| Model | n=5k | n=10k | n=20k | n=50k | n=100k |
|-------|------|-------|-------|-------|--------|
| KNET | **0.836** | **0.887** | **0.835** | **0.867** | **0.877** |
| CNN-TF (matched) | 0.729 | 0.831 | 0.780 | 0.836 | 0.849 |
| Transformer | 0.568 | 0.545 | 0.624 | 0.829 | 0.824 |
| CNN | 0.643 | 0.769 | 0.790 | 0.854 | 0.867 |
| Transformer (k-mer) | 0.691 | 0.800 | 0.775 | 0.812 | 0.826 |
| MHA | 0.550 | 0.532 | 0.529 | 0.610 | 0.634 |

### Exp C — Constraint ablation (k=12, nk=64, full dataset)

**RC task:**

| Dataset | KNET_rc mean | KNET_uncons_rc mean | Δ |
|---------|-------------|---------------------|---|
| abs-ran_fix2 (Simu7) | **0.9995** | 0.9992 | +0.0003 |
| random_rand (Simu16) | **0.9939** | 0.9938 | +0.0001 |

**Markov task:**

| Model | mean AUROC | Δ vs uncons |
|-------|-----------|-------------|
| KNET | **0.8695** | +0.0090 |
| KNET_uncons | 0.8605 | — |

### Exp D — Simu16 learning curves (random_rand)

| n | KNET_rc | CNN-TF (matched) | CNN-TF (full) |
|---|---------|-----------------|---------------|
| 5,000 | **0.985** | 0.505 | 0.522 |
| 10,000 | **0.991** | 0.541 | 0.510 |
| 20,000 | **0.992** | 0.530 | 0.511 |
| 50,000 | **0.993** | 0.568 | 0.553 |
| 100,000 | **0.994** | 0.683 | 0.564 |

*Anomaly excluded: KNET_rc n=5000 seed=0 (AUROC=0.509)*

---

## Infrastructure

| Item | Detail |
|------|--------|
| Main compute | 192.168.3.17 (liut), RTX 4090, direct SSH |
| Cluster | luminary (172.18.18.7), via bastion 172.18.18.11, Slurm |
| Conda env | `kattn-sim` at `/rd1/liut/miniconda3/envs/kattn-sim/` (on 192.168.3.17) |
| Results CSV | `results/exp_results_merged.csv` (local) |
| Figures | `revision/figures/` (local) |
| Plot script | `src/simulation/plot_revision.py` (use kattn-sim Python) |
| Style module | `src/simulation/figure_style.py` (Wong palette, fonttype=42) |
| Skills | `.claude/skills/plot-revision.md`, `.claude/skills/figure-style.md` |

---

## Immediate Next Actions (for next agent)

1. **[Blocking] Wait for Exp A seeds 1&2** on 192.168.3.17
   - Monitor: `ssh liut@192.168.3.17 "tail -5 /tmp/expA_seeds12.log"`
   - When done: merge CSV → re-plot `--exp A` on 192.168.3.17 → transfer figures back

2. **[Blocking] Exp B extension on luminary** (210 runs, user submits manually)
   - Script ready: `src/simulation/submit_expB_ext.sh` + `job_expB_ext.sh`
   - Merges rc n=1k/2k (all 5 seeds) + existing rc sizes (seeds 3,4) + markov n=2k (all 5 seeds) + existing markov sizes (seeds 3,4)
   - Markov n=2k uses `markov_1_0_5000 --sample-size 2000` (no new FASTA needed)
   - `plot_revision.py` already updated to handle sample_size>0 for Markov
   - When done: download luminary CSV → merge → re-plot `--exp B` → transfer figures back

3. **Text revisions E/F/G** — no experiments needed, only manuscript edits
