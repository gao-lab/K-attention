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


def load_results():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"{RESULTS_CSV} not found. Run experiments first.")
    df = pd.read_csv(RESULTS_CSV)
    df["val_auroc"] = pd.to_numeric(df["val_auroc"], errors="coerce")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["A", "B"], default=None)
    args = parser.parse_args()

    df = load_results()
    print(f"Loaded {len(df)} result rows.")

    if args.exp in (None, "A"):
        exp_a_table(df)
    if args.exp in (None, "B"):
        exp_b_table(df)


if __name__ == "__main__":
    main()
