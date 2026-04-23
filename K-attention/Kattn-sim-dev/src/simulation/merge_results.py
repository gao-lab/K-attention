#!/usr/bin/env python3
"""
Merge exp_results.csv from multiple machines into one deduplicated file.

Usage:
    python merge_results.py --inputs /path/a/exp_results.csv /path/b/exp_results.csv \
                            --output ../../results/exp_results_merged.csv

Dedup key: (model_type, test_config, kernel_size, num_kernels, sample_size, max_lr, version)
When duplicates exist, keep the row with the higher val_auroc.
"""

import argparse
import csv
from pathlib import Path

KEY_COLS = ["model_type", "test_config", "kernel_size", "num_kernels",
            "sample_size", "max_lr", "version"]
HEADER = ["timestamp", "model_type", "test_config", "kernel_size", "num_kernels",
          "sample_size", "max_lr", "version", "val_auroc"]


def load_csv(path):
    rows = {}
    path = Path(path)
    if not path.exists():
        print(f"[skip] {path} not found")
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(row[c] for c in KEY_COLS)
            auroc = float(row["val_auroc"])
            if key not in rows or auroc > float(rows[key]["val_auroc"]):
                rows[key] = row
    print(f"[loaded] {path}: {len(rows)} unique runs")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="CSV files to merge (from different machines)")
    parser.add_argument("--output", default="../../results/exp_results_merged.csv",
                        help="Output merged CSV path")
    args = parser.parse_args()

    merged = {}
    for path in args.inputs:
        rows = load_csv(path)
        for key, row in rows.items():
            if key not in merged or float(row["val_auroc"]) > float(merged[key]["val_auroc"]):
                merged[key] = row

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for row in sorted(merged.values(),
                          key=lambda r: (r["model_type"], r["test_config"],
                                         r["sample_size"], r["version"])):
            writer.writerow(row)

    print(f"[done] merged {len(merged)} runs → {out}")


if __name__ == "__main__":
    main()
