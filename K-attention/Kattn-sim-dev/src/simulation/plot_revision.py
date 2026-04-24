#!/usr/bin/env python3
"""
plot_revision.py — Revision figures for K-attention (BIB-26-0525).

Usage:
  python plot_revision.py              # all figures
  python plot_revision.py --exp A      # Exp A heatmap only
  python plot_revision.py --exp B      # Exp B learning curves only
  python plot_revision.py --exp C      # Exp C ablation only
  python plot_revision.py --exp D      # Exp D Simu16 curve only

Output: ../../revision/figures/exp_{a,b,c,d}_*.{pdf,png}
Style:  figure_style.py (Wong palette, Arial, fonttype=42, 300 dpi)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import figure_style as fs
fs.apply()   # must be called before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
RESULTS_CSV = SCRIPT_DIR / "../../results/exp_results_merged.csv"
FIGURES_DIR = SCRIPT_DIR / "../../revision/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        sys.exit(f"ERROR: {RESULTS_CSV} not found. Run experiments and merge first.")
    df = pd.read_csv(RESULTS_CSV)
    df["val_auroc"]  = pd.to_numeric(df["val_auroc"],  errors="coerce")
    df["sample_size"] = pd.to_numeric(df["sample_size"], errors="coerce").fillna(-1).astype(int)
    print(f"Loaded {len(df)} rows from {RESULTS_CSV.resolve()}")
    return df


# ── Exp A — Hyperparameter sensitivity heatmap ─────────────────────────────

def plot_exp_a(df: pd.DataFrame):
    """2-panel annotated heatmap: kernel_size × num_kernels → AUROC."""
    fig, axes = plt.subplots(1, 2, figsize=fs.FIG_2COL_TALL)

    panels = [
        ("RC task (Simu7)",    "KNET_rc", "abs-ran_fix2"),
        ("Markov task",        "KNET",    "markov_1_0_50000"),
    ]

    for ax, (title, model, config) in zip(axes, panels):
        sub = df[
            (df["model_type"] == model) &
            (df["test_config"] == config) &
            (df["sample_size"] == -1)
        ]
        if sub.empty:
            ax.text(0.5, 0.5, "[no data]", transform=ax.transAxes,
                    ha="center", va="center", fontsize=fs.FS_LABEL)
            ax.set_title(title, fontsize=fs.FS_TITLE)
            continue

        pivot = sub.pivot_table(
            index="kernel_size", columns="num_kernels",
            values="val_auroc", aggfunc="mean"
        ).sort_index().sort_index(axis=1)

        vmin = max(0.5,  pivot.values.min() - 0.02)
        vmax = min(1.0,  pivot.values.max() + 0.01)

        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".3f",
            cmap="YlOrRd", linewidths=0.4, linecolor="#dddddd",
            vmin=vmin, vmax=vmax,
            cbar_kws={"label": "AUROC", "shrink": 0.85},
            annot_kws={"size": fs.FS_ANNOT + 1},
        )
        ax.set_title(title, fontsize=fs.FS_TITLE, pad=6)
        ax.set_xlabel("num_kernels", fontsize=fs.FS_LABEL)
        ax.set_ylabel("kernel_size", fontsize=fs.FS_LABEL)
        ax.tick_params(axis="both", labelsize=fs.FS_TICK)
        ax.collections[0].colorbar.ax.tick_params(labelsize=fs.FS_TICK)
        ax.collections[0].colorbar.set_label("AUROC", fontsize=fs.FS_LABEL)

    fig.suptitle("Hyperparameter sensitivity (AUROC)", fontsize=fs.FS_TITLE, y=1.02)
    fig.tight_layout(w_pad=3)
    fs.save(fig, FIGURES_DIR / "exp_a_heatmap")
    plt.close(fig)


# ── Exp B — Learning curves ────────────────────────────────────────────────

def _learning_curve_panel(ax, sub_df, models, x_col, title):
    """Plot mean ± 1 std learning curves for a list of models onto ax."""
    for model in models:
        msub = sub_df[sub_df["model_type"] == model]
        if msub.empty:
            continue
        stats = (msub.groupby(x_col)["val_auroc"]
                     .agg(["mean", "std"])
                     .reset_index()
                     .sort_values(x_col))
        color = fs.MODEL_COLORS.get(model, "#333333")
        label = fs.MODEL_NAMES.get(model, model)
        ax.plot(stats[x_col], stats["mean"], marker="o",
                label=label, color=color,
                linewidth=fs.LW, markersize=fs.MS)
        ax.fill_between(
            stats[x_col],
            stats["mean"] - stats["std"].fillna(0),
            stats["mean"] + stats["std"].fillna(0),
            alpha=0.12, color=color, linewidth=0,
        )
    ax.set_xscale("log")
    ax.set_title(title, fontsize=fs.FS_TITLE, pad=6)
    ax.set_xlabel("Training set size", fontsize=fs.FS_LABEL)
    ax.set_ylabel("AUROC", fontsize=fs.FS_LABEL)
    ax.set_ylim(0.45, 1.05)
    ax.tick_params(axis="both", labelsize=fs.FS_TICK)
    ax.legend(fontsize=fs.FS_LEGEND, loc="lower right",
              handlelength=1.5, borderpad=0.3)


def plot_exp_b(df: pd.DataFrame):
    """Two-panel learning curves (RC + Markov), mean ± 1 std across 3 seeds."""
    fig, axes = plt.subplots(1, 2, figsize=fs.FIG_2COL_TALL)

    # RC panel
    rc_models = ["KNET_rc", "cnn_transformer_pm", "transformer_cls",
                 "transformer_cls_kmer", "cnn", "mha"]
    rc_df = df[
        (df["test_config"] == "abs-ran_fix2") &
        (df["model_type"].isin(rc_models)) &
        (df["sample_size"] > 0)
    ].copy()
    # exclude anomalous KNET_rc n=5000 seed=1 (AUROC≈0.595)
    rc_df = rc_df[~(
        (rc_df["model_type"] == "KNET_rc") &
        (rc_df["sample_size"] == 5000) &
        (rc_df["val_auroc"] < 0.7)
    )]
    _learning_curve_panel(axes[0], rc_df, rc_models, "sample_size",
                          "RC task (Simu7, abs-ran_fix2)")

    # Markov panel — data size from sample_size (if subsampled) or test_config suffix
    mk_models = ["KNET", "cnn_transformer_pm", "transformer_cls",
                 "transformer_cls_kmer", "cnn", "mha"]
    mk_df = df[
        df["test_config"].str.startswith("markov_1_0_") &
        df["model_type"].isin(mk_models)
    ].copy()
    from_config = mk_df["test_config"].str.extract(r"_(\d+)$")[0].astype(int)
    mk_df["data_size"] = mk_df["sample_size"].where(mk_df["sample_size"] > 0, from_config)
    _learning_curve_panel(axes[1], mk_df, mk_models, "data_size",
                          "Markov task")

    fig.suptitle("Learning curves (mean ± 1 std, 3 seeds)", fontsize=fs.FS_TITLE, y=1.02)
    fig.tight_layout(w_pad=3)
    fs.save(fig, FIGURES_DIR / "exp_b_learning_curve")
    plt.close(fig)


# ── Exp C — Constraint ablation bar charts ─────────────────────────────────

def _bar_panel(ax, data_df, group_col, hue_col, groups, hues, group_labels, title):
    """Grouped bar chart with std error bars and per-seed scatter overlay."""
    n_hues = len(hues)
    bar_w  = 0.32
    gap    = 0.12
    group_w = n_hues * bar_w + gap

    xtick_pos, xtick_label = [], []

    for gi, group in enumerate(groups):
        center = gi * group_w
        xtick_pos.append(center + (n_hues - 1) * bar_w / 2)
        xtick_label.append(group_labels.get(group, group))

        for hi, hue in enumerate(hues):
            sub   = data_df[(data_df[group_col] == group) & (data_df[hue_col] == hue)]
            x     = center + hi * bar_w
            if sub.empty:
                continue
            mean  = sub["val_auroc"].mean()
            std   = sub["val_auroc"].std(ddof=1) if len(sub) > 1 else 0.0
            color = fs.MODEL_COLORS.get(hue, f"C{hi}")
            label = fs.MODEL_NAMES.get(hue, hue) if gi == 0 else None

            ax.bar(x, mean, width=bar_w * 0.85,
                   color=color, alpha=0.72, label=label, zorder=2)
            ax.errorbar(x, mean, yerr=std, fmt="none",
                        color="#333333", capsize=fs.CAPSIZE,
                        linewidth=fs.LW_THIN, zorder=3)
            jitter = np.linspace(-0.05, 0.05, len(sub))
            for val, jit in zip(sub["val_auroc"].values, jitter):
                ax.scatter(x + jit, val, color="#222222",
                           s=fs.SCATTER_S, zorder=4, alpha=0.75,
                           linewidths=0)

    if not data_df.empty:
        ymin = max(0.0, data_df["val_auroc"].min() - 0.06)
        ymax = min(1.02, data_df["val_auroc"].max() + 0.05)
        ax.set_ylim(ymin, ymax)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_label, fontsize=fs.FS_TICK)
    ax.set_title(title, fontsize=fs.FS_TITLE, pad=6)
    ax.set_ylabel("AUROC", fontsize=fs.FS_LABEL)
    ax.tick_params(axis="y", labelsize=fs.FS_TICK)
    ax.legend(fontsize=fs.FS_LEGEND, borderpad=0.3)


def plot_exp_c(df: pd.DataFrame):
    """Two-panel constraint ablation: RC (left) and Markov (right)."""
    fig, axes = plt.subplots(1, 2, figsize=fs.FIG_2COL_TALL)

    # RC
    rc_models  = ["KNET_rc", "KNET_uncons_rc"]
    rc_configs = ["abs-ran_fix2", "random_rand"]
    rc = df[
        df["model_type"].isin(rc_models) &
        df["test_config"].isin(rc_configs) &
        (df["sample_size"] == -1) &
        (df["kernel_size"] == 12) &
        (df["num_kernels"] == 64)
    ].copy()
    _bar_panel(axes[0], rc,
               group_col="test_config", hue_col="model_type",
               groups=rc_configs, hues=rc_models,
               group_labels={"abs-ran_fix2": "Simu7\n(abs-ran_fix2)",
                             "random_rand":  "Simu16\n(random_rand)"},
               title="RC task")

    # Markov
    mk_models = ["KNET", "KNET_uncons"]
    mk_cfg    = "markov_1_0_50000"
    mk = df[
        df["model_type"].isin(mk_models) &
        (df["test_config"] == mk_cfg) &
        (df["sample_size"] == -1) &
        (df["kernel_size"] == 12) &
        (df["num_kernels"] == 64)
    ].copy()
    _bar_panel(axes[1], mk,
               group_col="test_config", hue_col="model_type",
               groups=[mk_cfg], hues=mk_models,
               group_labels={mk_cfg: "Markov\n(n=50 000)"},
               title="Markov task")

    fig.suptitle("Point-to-point constraint ablation (k=12, full dataset)",
                 fontsize=fs.FS_TITLE, y=1.02)
    fig.tight_layout(w_pad=3)
    fs.save(fig, FIGURES_DIR / "exp_c_ablation")
    plt.close(fig)


# ── Exp D — Simu16 learning curves ────────────────────────────────────────

def plot_exp_d(df: pd.DataFrame):
    """Single-panel learning curves on Simu16 (random_rand) for 3 models."""
    fig, ax = plt.subplots(figsize=(fs.FIG_1COL[0] * 1.3, fs.FIG_1COL[1] * 1.4))

    models = ["KNET_rc", "cnn_transformer_pm", "cnn_transformer"]
    sub = df[
        df["model_type"].isin(models) &
        (df["test_config"] == "random_rand") &
        (df["sample_size"] > 0)
    ].copy()
    # exclude anomalous KNET_rc n=5000 seed=0 (AUROC≈0.509)
    sub = sub[~(
        (sub["model_type"] == "KNET_rc") &
        (sub["sample_size"] == 5000) &
        (sub["val_auroc"] < 0.6)
    )]

    _learning_curve_panel(ax, sub, models, "sample_size",
                          "Simu16 (random_rand)")
    fig.tight_layout()
    fs.save(fig, FIGURES_DIR / "exp_d_simu16")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate revision figures.")
    parser.add_argument("--exp", choices=["A", "B", "C", "D"],
                        help="Experiment to plot (default: all)")
    args = parser.parse_args()

    df      = load_data()
    run_all = args.exp is None

    if run_all or args.exp == "A":
        print("Plotting Exp A …")
        plot_exp_a(df)
    if run_all or args.exp == "B":
        print("Plotting Exp B …")
        plot_exp_b(df)
    if run_all or args.exp == "C":
        print("Plotting Exp C …")
        plot_exp_c(df)
    if run_all or args.exp == "D":
        print("Plotting Exp D …")
        plot_exp_d(df)

    print(f"\nDone. Figures in: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
