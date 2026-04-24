#!/usr/bin/env python
"""Analyze experiment results from exp_results.csv.
Usage:
  python analyze_results.py --exp A   # hyperparameter scan pivot table
  python analyze_results.py --exp B   # learning curve table
  python analyze_results.py           # all
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_CSV = Path("../../results/exp_results.csv")
MERGED_CSV = Path("../../results/exp_results_merged.csv")


def load_results(merged=False):
    path = MERGED_CSV if (merged and MERGED_CSV.exists()) else RESULTS_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run experiments first.")
    df = pd.read_csv(path)
    df["val_auroc"] = pd.to_numeric(df["val_auroc"], errors="coerce")
    print(f"Loaded {len(df)} rows from {path}")
    return df


def exp_a_table(df):
    """Hyperparameter scan: kernel_size × num_kernels pivot for each dataset."""
    mask = df["model_type"].isin(["KNET", "KNET_rc"])
    sub = df[mask].copy()
    for dst in sub["test_config"].unique():
        d = sub[sub["test_config"] == dst]
        pivot = d.pivot_table(
            index="kernel_size", columns="num_kernels", values="val_auroc",
            aggfunc="mean"
        ).round(4)
        print(f"\n=== Exp A: KNET AUROC on {dst} ===")
        print(pivot.to_string())


def exp_b_table(df):
    """Learning curves: model × data_size, grouped by dataset."""
    sub = df.copy()
    # For RC: sample_size is the data size; for Markov: use test_config suffix
    sub["data_size"] = sub.apply(
        lambda r: r["sample_size"] if r["sample_size"] > 0
        else int(r["test_config"].rsplit("_", 1)[-1]) if r["test_config"].split("_")[-1].isdigit()
        else -1, axis=1
    )
    sub = sub[sub["data_size"] > 0]
    for dst_group in ["RC", "Markov"]:
        if dst_group == "RC":
            d = sub[sub["test_config"].str.startswith("abs-ran_fix2")]
        else:
            d = sub[sub["test_config"].str.startswith("markov_1_0")]
        if d.empty:
            continue
        pivot = d.pivot_table(
            index="model_type", columns="data_size", values="val_auroc",
            aggfunc=["mean", "std"]
        ).round(4)
        print(f"\n=== Exp B: {dst_group} Learning Curves ===")
        print(pivot.to_string())


def exp_c_table(df):
    """Constraint ablation: KNET_rc vs KNET_uncons_rc (RC); KNET vs KNET_uncons (Markov)."""
    # RC ablation
    rc_models = ["KNET_rc", "KNET_uncons_rc"]
    rc_configs = ["abs-ran_fix2", "random_rand"]
    rc = df[df["model_type"].isin(rc_models) & df["test_config"].isin(rc_configs) & (df["sample_size"] == -1)]
    if not rc.empty:
        print("\n=== Exp C: RC Constraint Ablation (full dataset) ===")
        pivot = rc.pivot_table(
            index="model_type", columns="test_config", values="val_auroc",
            aggfunc=["mean", "std"]
        ).round(4)
        print(pivot.to_string())
        # Per-seed detail
        for cfg in rc_configs:
            d = rc[rc["test_config"] == cfg].sort_values(["model_type", "version"])
            print(f"\n  {cfg} — per seed:")
            for _, row in d.iterrows():
                print(f"    {row['model_type']:20s} seed={int(row['version'])}  {row['val_auroc']:.4f}")

    # Markov ablation
    mk_models = ["KNET", "KNET_uncons"]
    mk_cfg = "markov_1_0_50000"
    mk = df[df["model_type"].isin(mk_models) & (df["test_config"] == mk_cfg) & (df["sample_size"] == -1)]
    if not mk.empty:
        print(f"\n=== Exp C: Markov Constraint Ablation ({mk_cfg}, full dataset) ===")
        pivot = mk.pivot_table(
            index="model_type", columns="version", values="val_auroc",
            aggfunc="mean"
        ).round(4)
        pivot["mean"] = mk.groupby("model_type")["val_auroc"].mean().round(4)
        print(pivot.to_string())


def exp_d_table(df):
    """Learning curve on Simu16 (random_rand): KNET_rc vs cnn_transformer_pm vs cnn_transformer."""
    models = ["KNET_rc", "cnn_transformer_pm", "cnn_transformer"]
    d = df[df["model_type"].isin(models) & (df["test_config"] == "random_rand") & (df["sample_size"] > 0)]
    if d.empty:
        print("\n=== Exp D: no data found ===")
        return
    # Exclude anomalous KNET_rc n=5000 seed=0 (AUROC < 0.6)
    d = d[~((d["model_type"] == "KNET_rc") & (d["sample_size"] == 5000) & (d["val_auroc"] < 0.6))]
    pivot_mean = d.pivot_table(
        index="model_type", columns="sample_size", values="val_auroc", aggfunc="mean"
    ).round(4)
    pivot_std = d.pivot_table(
        index="model_type", columns="sample_size", values="val_auroc", aggfunc="std"
    ).round(4)
    print("\n=== Exp D: Simu16 (random_rand) Learning Curve — mean AUROC ===")
    print(pivot_mean.to_string())
    print("\n=== Exp D: Simu16 (random_rand) Learning Curve — std AUROC ===")
    print(pivot_std.to_string())
    # Per-seed detail
    print("\n  Per-seed detail:")
    for model in models:
        for n in sorted(d["sample_size"].unique()):
            vals = d[(d["model_type"] == model) & (d["sample_size"] == n)]["val_auroc"].values
            vals_str = "  ".join(f"{v:.4f}" for v in sorted(vals))
            print(f"    {model:22s} n={n:>7d}  [{vals_str}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["A", "B", "C", "D"], default=None)
    parser.add_argument("--merged", action="store_true", help="Use merged CSV from all machines")
    args = parser.parse_args()

    df = load_results(merged=args.merged)

    if args.exp in (None, "A"):
        exp_a_table(df)
    if args.exp in (None, "B"):
        exp_b_table(df)
    if args.exp in (None, "C"):
        exp_c_table(df)
    if args.exp in (None, "D"):
        exp_d_table(df)


if __name__ == "__main__":
    main()
