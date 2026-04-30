#!/usr/bin/env python3
"""
plot_revision.py — Revision figures for K-attention (BIB-26-0525).

Usage:
  python plot_revision.py              # all figures
  python plot_revision.py --exp A      # Exp A heatmap only
  python plot_revision.py --exp B      # Exp B learning curves only
  python plot_revision.py --exp C      # Exp C ablation only
  python plot_revision.py --exp D      # Exp D Simu16 curve only
  python plot_revision.py --exp E      # Exp E RBP scatter only
  python plot_revision.py --exp F      # Exp F CRISPR grouped bar only

Output: ../../revision/figures/exp_{a,b,c,d,e,f}_*.{pdf,png}
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
SCRIPT_DIR    = Path(__file__).parent
RESULTS_CSV   = SCRIPT_DIR / "../../results/exp_results_merged.csv"
FIGURES_DIR   = SCRIPT_DIR / "../../revision/figures"
DATA_DIR      = SCRIPT_DIR / "../../revision/data"
RBP_KNET_TSV  = SCRIPT_DIR / "../../../Kattention_aten_test/scripts/RBP/log/Train_KNET_plus_seq_valid_test.tsv"
RBP_CNNTF_CSV = DATA_DIR / "rbp_cnntf_seed666.csv"
CRISPR_CSV    = DATA_DIR / "crispr_comparison.csv"
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
    # anomalous KNET_rc n=5000 version=1 has been replaced in CSV; no exclusion needed.
    _learning_curve_panel(axes[0], rc_df, rc_models, "sample_size",
                          "RC task (Simu7)")

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
                          "Markov task (Simu_M7)")

    fig.suptitle("Learning curves (mean ± 1 std, 5 seeds)", fontsize=fs.FS_TITLE, y=1.02)
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
    """Two-panel constraint ablation: learning curves at small sample sizes.

    Left:  RC task (Simu16, random_rand) — KNET_rc vs KNET_uncons_rc (nk=64)
    Right: Markov task — KNET vs KNET_uncons (nk=64)

    Small-sample points (n=2k,5k,10k) from run_expC_ablation.sh;
    full-data point from run_expC.sh (sample_size=-1 → mapped to 50 000).
    """
    fig, axes = plt.subplots(1, 2, figsize=fs.FIG_2COL_TALL)

    # ── RC panel (random_rand, Simu16) ────────────────────────────────────
    rc_models = ["KNET_rc", "KNET_uncons_rc"]
    rc_display = {
        "KNET_rc":       "KNET (RC)",
        "KNET_uncons_rc": "KNET_uncons (RC)",
    }
    rc_df = df[
        (df["test_config"] == "random_rand") &
        (df["model_type"].isin(rc_models)) &
        (df["kernel_size"] == 12) &
        (df["num_kernels"] == 64) &
        (df["sample_size"].isin([2000, 5000, 10000, -1]))
    ].copy()
    # Map full dataset (-1) to representative size 50 000 for x-axis
    rc_df["plot_size"] = rc_df["sample_size"].replace(-1, 50000)

    for model in rc_models:
        msub = rc_df[rc_df["model_type"] == model]
        if msub.empty:
            continue
        stats = (msub.groupby("plot_size")["val_auroc"]
                     .agg(["mean", "std"])
                     .reset_index()
                     .sort_values("plot_size"))
        color = fs.MODEL_COLORS.get(model, "#333333")
        label = rc_display.get(model, model)
        axes[0].plot(stats["plot_size"], stats["mean"], marker="o",
                     label=label, color=color,
                     linewidth=fs.LW, markersize=fs.MS)
        axes[0].fill_between(
            stats["plot_size"],
            stats["mean"] - stats["std"].fillna(0),
            stats["mean"] + stats["std"].fillna(0),
            alpha=0.15, color=color, linewidth=0,
        )

    axes[0].set_xscale("log")
    axes[0].set_xticks([2000, 5000, 10000, 50000])
    axes[0].set_xticklabels(["2k", "5k", "10k", "full\n(~50k)"],
                             fontsize=fs.FS_TICK)
    axes[0].set_ylim(0.45, 1.05)
    axes[0].set_title("RC task (Simu16)", fontsize=fs.FS_TITLE, pad=6)
    axes[0].set_xlabel("Training set size", fontsize=fs.FS_LABEL)
    axes[0].set_ylabel("AUROC", fontsize=fs.FS_LABEL)
    axes[0].tick_params(axis="y", labelsize=fs.FS_TICK)
    axes[0].legend(fontsize=fs.FS_LEGEND, loc="lower right",
                   handlelength=1.5, borderpad=0.3)

    # ── Markov panel ──────────────────────────────────────────────────────
    # Only show KNET (full constraints) vs KNET_uncons_nomask (truly unconstrained)
    # Display names override: "KNET (Markov)" and "KNET_uncons (Markov)"
    mk_models = ["KNET", "KNET_uncons_nomask"]
    mk_display = {
        "KNET":               "KNET (Markov)",
        "KNET_uncons_nomask": "KNET_uncons (Markov)",
    }
    mk_df = df[
        df["test_config"].str.startswith("markov_1_0_") &
        (df["model_type"].isin(mk_models)) &
        (df["kernel_size"] == 12) &
        (df["num_kernels"] == 64)
    ].copy()
    from_config = mk_df["test_config"].str.extract(r"_(\d+)$")[0].astype(int)
    mk_df["data_size"] = mk_df["sample_size"].where(mk_df["sample_size"] > 0, from_config)
    mk_df = mk_df[mk_df["data_size"].isin([2000, 5000, 10000, 50000])]

    for model in mk_models:
        msub = mk_df[mk_df["model_type"] == model]
        if msub.empty:
            continue
        stats = (msub.groupby("data_size")["val_auroc"]
                     .agg(["mean", "std"])
                     .reset_index()
                     .sort_values("data_size"))
        color = fs.MODEL_COLORS.get(model, "#333333")
        label = mk_display.get(model, model)
        axes[1].plot(stats["data_size"], stats["mean"], marker="o",
                     label=label, color=color,
                     linewidth=fs.LW, markersize=fs.MS)
        axes[1].fill_between(
            stats["data_size"],
            stats["mean"] - stats["std"].fillna(0),
            stats["mean"] + stats["std"].fillna(0),
            alpha=0.15, color=color, linewidth=0,
        )

    axes[1].set_xscale("log")
    axes[1].set_xticks([2000, 5000, 10000, 50000])
    axes[1].set_xticklabels(["2k", "5k", "10k", "50k"],
                             fontsize=fs.FS_TICK)
    axes[1].set_ylim(0.45, 1.05)
    axes[1].set_title("Markov task (Simu_M7)", fontsize=fs.FS_TITLE, pad=6)
    axes[1].set_xlabel("Training set size", fontsize=fs.FS_LABEL)
    axes[1].set_ylabel("AUROC", fontsize=fs.FS_LABEL)
    axes[1].tick_params(axis="both", labelsize=fs.FS_TICK)
    axes[1].legend(fontsize=fs.FS_LEGEND, loc="lower right",
                   handlelength=1.5, borderpad=0.3)

    fig.suptitle("Point-to-point constraint ablation (kernel_length=12, num_kernels=64, 3 seeds)",
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

    _learning_curve_panel(ax, sub, models, "sample_size",
                          "Simu16 (random_rand)")
    fig.tight_layout()
    fs.save(fig, FIGURES_DIR / "exp_d_simu16")
    plt.close(fig)


# ── Exp E — RBP comparison scatter plot ───────────────────────────────────

def plot_exp_e_rbp():
    """Scatter plot: KNET vs CNN-TF AUROC per RBP (N=172).

    X axis: CNN-TF (seed=666)
    Y axis: KNET (k=16, mean over 4 seeds)
    Reference line: y = x
    """
    # Load KNET RBP results
    if not RBP_KNET_TSV.exists():
        print(f"WARNING: {RBP_KNET_TSV} not found. Skipping Exp E.")
        return
    knet_df = pd.read_csv(
        RBP_KNET_TSV, sep="\t", header=None,
        names=["record", "rbp", "model", "kernel_size", "seed", "auroc", "loss"]
    )
    knet_df = knet_df[knet_df["kernel_size"] == 16]
    knet_mean = knet_df.groupby("rbp")["auroc"].mean().reset_index()
    knet_mean.columns = ["rbp", "knet_auroc"]

    # Load CNN-TF RBP results
    if not RBP_CNNTF_CSV.exists():
        print(f"WARNING: {RBP_CNNTF_CSV} not found. Skipping Exp E.")
        return
    cnntf_df = pd.read_csv(RBP_CNNTF_CSV)
    cnntf_df.columns = ["rbp", "cnntf_auroc"]

    merged = pd.merge(knet_mean, cnntf_df, on="rbp")
    n = len(merged)
    wins = (merged["knet_auroc"] > merged["cnntf_auroc"]).sum()
    delta = (merged["knet_auroc"] - merged["cnntf_auroc"]).mean()

    fig, ax = plt.subplots(figsize=(fs.FIG_SQUARE[0] * 1.2, fs.FIG_SQUARE[1] * 1.2))

    ax.scatter(
        merged["cnntf_auroc"], merged["knet_auroc"],
        color=fs.C["blue"], alpha=0.45, s=fs.SCATTER_S * 1.8,
        linewidths=0, zorder=3,
    )

    # y = x reference diagonal
    lim_min = min(merged["cnntf_auroc"].min(), merged["knet_auroc"].min()) - 0.02
    lim_max = max(merged["cnntf_auroc"].max(), merged["knet_auroc"].max()) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color=fs.C["gray"], linewidth=fs.LW_THIN, linestyle="--",
            zorder=2, label="y = x")

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")

    # Annotate summary statistics
    ax.text(
        0.04, 0.97,
        f"N={n}, wins={wins}/{n}\nmean Δ={delta:+.3f}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=fs.FS_ANNOT + 1,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85),
    )

    ax.set_xlabel("CNN-TF AUROC", fontsize=fs.FS_LABEL)
    ax.set_ylabel("KNET AUROC", fontsize=fs.FS_LABEL)
    ax.set_title("RBP binding prediction (N=172 datasets)", fontsize=fs.FS_TITLE, pad=6)
    ax.tick_params(axis="both", labelsize=fs.FS_TICK)

    fig.tight_layout()
    fs.save(fig, FIGURES_DIR / "exp_e_rbp_scatter")
    plt.close(fig)


# ── Exp F — CRISPR comparison grouped bar chart ───────────────────────────

_CRISPR_DATASET_SHORT = {
    "chari2015Train293T":      "Chari'15",
    "doench2014-Hs":           "Doench'14-Hs",
    "doench2014-Mm":           "Doench'14-Mm",
    "doench2016_hg19":         "Doench'16",
    "hart2016-Hct1162lib1Avg": "Hart'16-Hct",
    "hart2016-HelaLib1Avg":    "Hart'16-Hela1",
    "hart2016-HelaLib2Avg":    "Hart'16-Hela2",
    "hart2016-Rpe1Avg":        "Hart'16-Rpe1",
    "morenoMateos2015":        "Moreno'15",
    "xu2015TrainHl60":         "Xu'15-Hl60",
    "xu2015TrainKbm7":         "Xu'15-Kbm7",
}

_CRISPR_MODEL_COLORS = {
    "KNET_Crispr":      fs.C["blue"],
    "CRISPRon_base":    fs.C["green"],
    "transformer_cls":  fs.C["purple"],
    "cnn_transformer":  fs.C["orange"],
    "cnn_transformer_pm": fs.C["sky"],
}
_CRISPR_MODEL_NAMES = {
    "KNET_Crispr":      "KNET",
    "CRISPRon_base":    "CRISPRon",
    "transformer_cls":  "Transformer",
    "cnn_transformer":  "CNN-TF",
    "cnn_transformer_pm": "CNN-TF (matched)",
}
_CRISPR_MODELS = [
    "KNET_Crispr", "CRISPRon_base", "transformer_cls",
    "cnn_transformer", "cnn_transformer_pm",
]


def plot_exp_f_crispr():
    """Two-panel CRISPR comparison figure.

    Left:  grouped bar chart — per-dataset Spearman r for 5 models
    Right: summary bar chart — mean ± SEM across 11 datasets
    """
    if not CRISPR_CSV.exists():
        print(f"WARNING: {CRISPR_CSV} not found. Skipping Exp F.")
        return
    df = pd.read_csv(CRISPR_CSV)

    # Melt to long form
    df_long = df.melt(id_vars="dataset", var_name="model", value_name="spearman")
    datasets = list(df["dataset"])
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(fs.FIG_2COL_TALL[0] * 1.15, fs.FIG_2COL_TALL[1]),
                             gridspec_kw={"width_ratios": [3.5, 1]})

    # ── Left panel: per-dataset grouped bars ─────────────────────────────
    ax = axes[0]
    n_models = len(_CRISPR_MODELS)
    bar_w = 0.14
    gap = 0.06
    group_w = n_models * bar_w + gap

    xtick_pos, xtick_labels = [], []
    for di, ds in enumerate(datasets):
        center = di * group_w
        xtick_pos.append(center + (n_models - 1) * bar_w / 2)
        xtick_labels.append(_CRISPR_DATASET_SHORT.get(ds, ds))

        for mi, model in enumerate(_CRISPR_MODELS):
            val = df.loc[df["dataset"] == ds, model].values
            if len(val) == 0:
                continue
            x = center + mi * bar_w
            color = _CRISPR_MODEL_COLORS.get(model, f"C{mi}")
            label = _CRISPR_MODEL_NAMES.get(model, model) if di == 0 else None
            ax.bar(x, val[0], width=bar_w * 0.88,
                   color=color, alpha=0.78, label=label, zorder=2)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=fs.FS_TICK - 0.5,
                       rotation=35, ha="right")
    ax.set_ylabel("Spearman r", fontsize=fs.FS_LABEL)
    ax.set_title("CRISPR gRNA efficiency (per dataset)", fontsize=fs.FS_TITLE, pad=6)
    ax.tick_params(axis="y", labelsize=fs.FS_TICK)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=fs.FS_LEGEND, loc="upper right",
              borderpad=0.3, ncol=1)

    # ── Right panel: mean ± SEM summary bars ─────────────────────────────
    ax2 = axes[1]
    means, sems = [], []
    for model in _CRISPR_MODELS:
        vals = df[model].values
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))

    xs = np.arange(n_models)
    colors = [_CRISPR_MODEL_COLORS.get(m, f"C{i}") for i, m in enumerate(_CRISPR_MODELS)]
    bars = ax2.bar(xs, means, yerr=sems, width=0.6,
                   color=colors, alpha=0.82, capsize=fs.CAPSIZE,
                   error_kw={"linewidth": fs.LW_THIN, "ecolor": "#333333"},
                   zorder=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(
        [_CRISPR_MODEL_NAMES.get(m, m) for m in _CRISPR_MODELS],
        fontsize=fs.FS_TICK - 0.5, rotation=45, ha="right",
    )
    ax2.set_ylabel("Mean Spearman r", fontsize=fs.FS_LABEL)
    ax2.set_title("Summary\n(N=11)", fontsize=fs.FS_TITLE, pad=6)
    ax2.tick_params(axis="y", labelsize=fs.FS_TICK)
    ax2.set_ylim(0, 0.58)

    # Annotate mean values above bars
    for x, mean in zip(xs, means):
        ax2.text(x, mean + 0.012, f"{mean:.3f}",
                 ha="center", va="bottom", fontsize=fs.FS_ANNOT,
                 fontweight="bold")

    fig.suptitle("CRISPR gRNA efficiency prediction (5-fold CV, Spearman r)",
                 fontsize=fs.FS_TITLE, y=1.02)
    fig.tight_layout(w_pad=2)
    fs.save(fig, FIGURES_DIR / "exp_f_crispr_bar")
    plt.close(fig)


# ── Supp — RBFOX2 kernel_length sensitivity ───────────────────────────────

def plot_rbfox2_kernel():
    """Bar plot: RBFOX2_HepG2 AUROC vs kernel_length (12/16/20/24), mean ± std."""
    knet_tsv = RBP_KNET_TSV
    if not knet_tsv.exists():
        print(f"WARNING: {knet_tsv} not found. Skipping RBFOX2 plot.")
        return

    df = pd.read_csv(knet_tsv, sep="\t", header=None,
                     names=["record", "rbp", "model", "kernel_size", "seed", "auroc", "loss"])
    rbfox2 = df[df["rbp"] == "RBFOX2_HepG2"].copy()

    ks    = sorted(rbfox2["kernel_size"].unique())
    means = [rbfox2[rbfox2["kernel_size"] == k]["auroc"].mean() for k in ks]
    stds  = [rbfox2[rbfox2["kernel_size"] == k]["auroc"].std(ddof=1) for k in ks]

    fig, ax = plt.subplots(figsize=(fs.FIG_1COL[0] * 1.2, fs.FIG_1COL[1] * 1.3))

    x = np.arange(len(ks))
    bars = ax.bar(x, means, yerr=stds, color=fs.C["blue"], alpha=0.85,
                  width=0.5, capsize=fs.CAPSIZE, error_kw={"linewidth": fs.LW_THIN},
                  zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks], fontsize=fs.FS_TICK)
    ax.set_xlabel("kernel_length", fontsize=fs.FS_LABEL)
    ax.set_ylabel("AUROC", fontsize=fs.FS_LABEL)
    ax.tick_params(axis="y", labelsize=fs.FS_TICK)
    ax.set_title("RBFOX2_HepG2 — kernel_length sensitivity",
                 fontsize=fs.FS_TITLE, pad=6)

    ymin = min(m - s for m, s in zip(means, stds))
    ymax = max(m + s for m, s in zip(means, stds))
    margin = (ymax - ymin) * 1.2
    ax.set_ylim(ymin - margin, ymax + margin)

    fig.tight_layout()
    fs.save(fig, FIGURES_DIR / "supp_rbfox2_kernel")
    plt.close(fig)


# ── RC Tasks — three-panel learning curves (Task1/2/3) ─────────────────────

def plot_rc_tasks(df: pd.DataFrame):
    """Three-panel learning curves for RC Task 1/2/3 (random_fix2/fix1/rand)."""
    rc_models = ["KNET_rc", "cnn_transformer_pm", "transformer_cls",
                 "transformer_cls_kmer", "cnn", "mha"]
    tasks = [
        ("random_fix2", "RC Task 1 (40% PWM, random pos)"),
        ("random_fix1", "RC Task 2 (20% PWM, random pos)"),
        ("random_rand", "RC Task 3 (0% PWM, random pos)"),
    ]
    # Target sizes for RC tasks (1K, 2K, 5K, 20K, 50K, 100K)
    RC_SIZES = [1000, 2000, 5000, 20000, 50000, 100000]

    fig, axes = plt.subplots(1, 3, figsize=(fs.FIG_2COL_TALL[0] * 1.5, fs.FIG_2COL_TALL[1]))

    for ax, (config, title) in zip(axes, tasks):
        sub = df[
            (df["test_config"] == config) &
            (df["model_type"].isin(rc_models))
        ].copy()
        # Map sample_size=-1 (full dataset) to 100000
        sub["plot_size"] = sub["sample_size"].replace(-1, 100000)
        sub = sub[sub["plot_size"].isin(RC_SIZES)]
        if sub.empty:
            ax.set_title(f"{title}\n(no data)", fontsize=fs.FS_TITLE, pad=6)
            ax.set_visible(True)
        else:
            _learning_curve_panel(ax, sub, rc_models, "plot_size", title)

    fig.suptitle("RC task learning curves (mean ± 1 std, 5 seeds)", fontsize=fs.FS_TITLE, y=1.02)
    fig.tight_layout(w_pad=3)
    fs.save(fig, FIGURES_DIR / "rc_tasks_learning_curve")
    plt.close(fig)


# ── Markov Tasks — three-panel learning curves (Task1/2/3) ─────────────────

def _mk_data_size(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve data_size: use sample_size if >0, else extract from config suffix."""
    df = df.copy()
    from_config = df["test_config"].str.extract(r"_(\d+)$")[0].astype(float).astype("Int64")
    df["data_size"] = df["sample_size"].where(df["sample_size"] > 0, from_config)
    return df


def plot_markov_tasks(df: pd.DataFrame):
    """Three-panel learning curves for Markov Task 1/2/3 (entropy 0.75/1.0/1.25)."""
    mk_models = ["KNET", "cnn_transformer_pm", "transformer_cls",
                 "transformer_cls_kmer", "cnn", "mha"]
    tasks = [
        ("markov_0_75", "Markov Task 1 (entropy=0.75, easy)"),
        ("markov_1_0",  "Markov Task 2 (entropy=1.0, medium)"),
        ("markov_1_25", "Markov Task 3 (entropy=1.25, hard)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(fs.FIG_2COL_TALL[0] * 1.5, fs.FIG_2COL_TALL[1]))

    for ax, (prefix, title) in zip(axes, tasks):
        sub = df[
            df["test_config"].str.startswith(prefix + "_") &
            df["model_type"].isin(mk_models)
        ].copy()
        if sub.empty:
            ax.set_title(f"{title}\n(no data)", fontsize=fs.FS_TITLE, pad=6)
            ax.set_visible(True)
        else:
            sub = _mk_data_size(sub)
            _learning_curve_panel(ax, sub, mk_models, "data_size", title)

    fig.suptitle("Markov task learning curves (mean ± 1 std, 5 seeds)", fontsize=fs.FS_TITLE, y=1.02)
    fig.tight_layout(w_pad=3)
    fs.save(fig, FIGURES_DIR / "markov_tasks_learning_curve")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate revision figures.")
    parser.add_argument("--exp", choices=["A", "B", "C", "D", "E", "F", "RBFOX2",
                                          "RC_TASKS", "MK_TASKS"],
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
    if run_all or args.exp == "E":
        print("Plotting Exp E (RBP scatter) …")
        plot_exp_e_rbp()
    if run_all or args.exp == "F":
        print("Plotting Exp F (CRISPR bar) …")
        plot_exp_f_crispr()
    if run_all or args.exp == "RBFOX2":
        print("Plotting RBFOX2 kernel_length sensitivity …")
        plot_rbfox2_kernel()
    if run_all or args.exp == "RC_TASKS":
        print("Plotting RC tasks learning curves …")
        plot_rc_tasks(df)
    if run_all or args.exp == "MK_TASKS":
        print("Plotting Markov tasks learning curves …")
        plot_markov_tasks(df)

    print(f"\nDone. Figures in: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
