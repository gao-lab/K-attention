#!/usr/bin/env python3
"""Analyze experiment results from exp_results.csv or exp_results_merged.csv.

Usage:
  python analyze_results.py --exp A          # hyperparameter scan pivot table
  python analyze_results.py --exp B          # RC + Markov learning curves (all configs)
  python analyze_results.py --exp C          # constraint ablation
  python analyze_results.py --exp D          # Simu16 learning curves
  python analyze_results.py --exp B --rc-only     # RC learning curves only
  python analyze_results.py --exp B --markov-only # Markov learning curves only
  python analyze_results.py --merged         # use merged CSV
"""
import argparse
from pathlib import Path
import csv
import sys
from collections import defaultdict

RESULTS_CSV = Path("../../results/exp_results.csv")
MERGED_CSV = Path("../../results/exp_results_merged.csv")


def load_results(merged=False):
    path = MERGED_CSV if (merged and MERGED_CSV.exists()) else RESULTS_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run experiments first.")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["val_auroc"] = float(row["val_auroc"])
                row["kernel_size"] = int(float(row["kernel_size"]))
                row["num_kernels"] = int(float(row["num_kernels"]))
                row["sample_size"] = int(float(row["sample_size"]))
                row["version"] = int(float(row["version"]))
                rows.append(row)
            except (ValueError, TypeError):
                pass
    print(f"Loaded {len(rows)} rows from {path}")
    return rows


def pivot_table(rows, index_key, col_key, val_key="val_auroc", agg="mean"):
    """Build a pivot table dict[index][col] -> aggregated value."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r[index_key]][r[col_key]].append(r[val_key])
    result = {}
    for idx, cols in groups.items():
        result[idx] = {}
        for col, vals in cols.items():
            if agg == "mean":
                result[idx][col] = round(sum(vals) / len(vals), 4)
            elif agg == "std":
                result[idx][col] = round(
                    (sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals)) ** 0.5, 4
                )
    return result


def print_pivot(pivot, title, indices=None, columns=None):
    """Pretty-print a pivot table dict[index][col] -> value."""
    print(f"\n=== {title} ===")
    if not pivot:
        print("  (no data)")
        return
    if indices is None:
        indices = sorted(pivot.keys())
    if columns is None:
        cols = set()
        for v in pivot.values():
            cols.update(v.keys())
        columns = sorted(cols)
    # Header
    header = f"{'':20s}"
    for c in columns:
        header += f" {str(c):>10s}"
    print(header)
    print("-" * len(header))
    for idx in indices:
        row_str = f"{str(idx):20s}"
        for c in columns:
            val = pivot.get(idx, {}).get(c, None)
            if val is not None:
                row_str += f" {val:10.4f}"
            else:
                row_str += f" {'-':>10s}"
        print(row_str)


# ═══════════════════════════════════════════════════════════════════════════════
# Exp A: Hyperparameter scan
# ═══════════════════════════════════════════════════════════════════════════════

def exp_a_table(rows):
    """Hyperparameter scan: kernel_size x num_kernels pivot for each dataset."""
    models = {"KNET", "KNET_rc"}
    sub = [r for r in rows if r["model_type"] in models]
    for dst in sorted(set(r["test_config"] for r in sub)):
        d = [r for r in sub if r["test_config"] == dst]
        # Take only full-dataset runs (sample_size == -1)
        d = [r for r in d if r["sample_size"] == -1]
        if not d:
            continue
        pivot = pivot_table(d, "kernel_size", "num_kernels")
        print_pivot(pivot, f"Exp A: {dst}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers for RC / Markov learning curves
# ═══════════════════════════════════════════════════════════════════════════════

RC_CONFIGS_MAP = {
    # Config name -> display label
    "abs-ran_fix2": "abs-ran_fix2 (Simu7, 40% PWM)",
    "random_fix2":  "random_fix2 (40% PWM)",
    "random_fix1":  "random_fix1 (20% PWM)",
    "random_rand":  "random_rand (Simu16, 0% PWM)",
}

RC_MODELS = ["KNET_rc", "cnn_transformer_pm", "cnn", "mha",
             "transformer_cls", "transformer_cls_kmer"]

MARKOV_MODELS = ["KNET", "cnn_transformer_pm", "cnn", "mha",
                 "transformer_cls", "transformer_cls_kmer"]

MARKOV_DIFFICULTY = {
    "0_75": "Markov H=0.75 (easy)",
    "1_0":  "Markov H=1.0 (medium)",
    "1_25": "Markov H=1.25 (hard)",
}


def get_markov_base_config(test_config, sample_size):
    """Normalize markov config to (entropy, data_size)."""
    parts = test_config.split("_")  # e.g. ["markov", "1", "0", "50000"]
    # Find entropy: parts after "markov"
    if len(parts) >= 4:
        entropy = f"{parts[1]}_{parts[2]}"  # "1_0"
        # Data size is from test_config suffix or sample_size
        suffix = parts[-1]
        if suffix.isdigit() and sample_size == -1:
            data_size = int(suffix)
        elif sample_size > 0:
            data_size = sample_size
        else:
            data_size = int(suffix) if suffix.isdigit() else -1
        return entropy, data_size
    return None, None


def rc_learning_curves(rows):
    """Learning curves for RC tasks: each config, each model's AUROC across data sizes."""
    for config, label in sorted(RC_CONFIGS_MAP.items()):
        sub = [r for r in rows if r["test_config"] == config
               and r["sample_size"] > 0
               and r["model_type"] in RC_MODELS]
        if not sub:
            continue
        pivot = pivot_table(sub, "model_type", "sample_size")
        # Show per-seed detail too
        seeds_by_model_size = defaultdict(lambda: defaultdict(list))
        for r in sub:
            seeds_by_model_size[r["model_type"]][r["sample_size"]].append(r["val_auroc"])

        print_pivot(pivot, f"Exp B RC: {label} — mean AUROC",
                    indices=RC_MODELS)

        # Per-seed detail for each model
        print(f"\n  Per-seed detail ({label}):")
        for model in RC_MODELS:
            sizes = sorted(seeds_by_model_size.get(model, {}).keys())
            if not sizes:
                continue
            for n in sizes:
                vals = seeds_by_model_size[model][n]
                if len(vals) >= 3:
                    vals_str = "  ".join(f"{v:.4f}" for v in sorted(vals))
                    mean = sum(vals) / len(vals)
                    print(f"    {model:22s} n={n:>7d}  mean={mean:.4f}  [{vals_str}]")
                else:
                    vals_str = "  ".join(f"{v:.4f}" for v in sorted(vals))
                    print(f"    {model:22s} n={n:>7d}  (n={len(vals)})  [{vals_str}]")


def markov_learning_curves(rows):
    """Learning curves for Markov tasks: group by entropy level."""
    markov_rows = [r for r in rows if r["test_config"].startswith("markov_")
                   and r["model_type"] in MARKOV_MODELS]

    by_entropy = defaultdict(list)
    for r in markov_rows:
        entropy, data_size = get_markov_base_config(r["test_config"], r["sample_size"])
        if entropy and data_size > 0:
            by_entropy[entropy].append({**r, "data_size": data_size})

    for entropy in sorted(by_entropy.keys()):
        label = MARKOV_DIFFICULTY.get(entropy, f"Markov H={entropy}")
        sub = by_entropy[entropy]
        pivot = pivot_table(sub, "model_type", "data_size")
        print_pivot(pivot, f"Exp B Markov: {label} — mean AUROC",
                    indices=MARKOV_MODELS)



def exp_b_table(rows, rc_only=False, markov_only=False):
    """Learning curves for all configurations."""
    if markov_only:
        markov_learning_curves(rows)
    elif rc_only:
        rc_learning_curves(rows)
    else:
        rc_learning_curves(rows)
        markov_learning_curves(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Exp C: Constraint ablation
# ═══════════════════════════════════════════════════════════════════════════════

def exp_c_table(rows):
    """Constraint ablation: KNET_rc vs KNET_uncons_rc (RC); KNET vs KNET_uncons (Markov)."""
    # RC ablation — full dataset
    rc_models = ["KNET_rc", "KNET_uncons_rc"]
    rc_configs = ["abs-ran_fix2", "random_rand"]
    rc = [r for r in rows if r["model_type"] in rc_models
          and r["test_config"] in rc_configs
          and r["sample_size"] == -1]
    if rc:
        print("\n=== Exp C: RC Constraint Ablation (full dataset) ===")
        pivot = pivot_table(rc, "model_type", "test_config")
        print_pivot(pivot, "mean AUROC", indices=rc_models, columns=rc_configs)
        # Per-seed
        for cfg in rc_configs:
            d = sorted(
                [r for r in rc if r["test_config"] == cfg],
                key=lambda r: (r["model_type"], r["version"])
            )
            print(f"\n  {cfg} — per seed:")
            for r in d:
                print(f"    {r['model_type']:20s} seed={r['version']}  {r['val_auroc']:.4f}")

    # RC ablation — small sample (random_rand)
    rc_small = [r for r in rows if r["model_type"] in rc_models
                and r["test_config"] == "random_rand"
                and r["sample_size"] > 0]
    if rc_small:
        print("\n=== Exp C: RC Small-Sample Ablation (random_rand) ===")
        pivot = pivot_table(rc_small, "model_type", "sample_size")
        print_pivot(pivot, "mean AUROC", indices=rc_models)

    # Markov ablation — full dataset
    mk_models = ["KNET", "KNET_uncons"]
    mk = [r for r in rows if r["model_type"] in mk_models
          and r["test_config"] == "markov_1_0_50000"
          and r["sample_size"] == -1]
    if mk:
        print("\n=== Exp C: Markov Constraint Ablation (markov_1_0_50000) ===")
        mean_by_model = {}
        for m in mk_models:
            vals = [r["val_auroc"] for r in mk if r["model_type"] == m]
            if vals:
                mean_by_model[m] = round(sum(vals) / len(vals), 4)
                print(f"  {m:20s}: {mean_by_model[m]:.4f}  (seeds: {[r['version'] for r in mk if r['model_type'] == m]})")

    # Markov small-sample ablation
    mk_small = [r for r in rows if r["model_type"] in mk_models
                and r["test_config"].startswith("markov_1_0")
                and r["sample_size"] > 0]
    if mk_small:
        print("\n=== Exp C: Markov Small-Sample Ablation ===")
        pivot = pivot_table(mk_small, "model_type", "sample_size")
        print_pivot(pivot, "mean AUROC", indices=mk_models)


# ═══════════════════════════════════════════════════════════════════════════════
# Exp D: Simu16 learning curves (KNET_rc vs CNN-TF on random_rand)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_d_table(rows):
    """Learning curve on Simu16 (random_rand): KNET_rc vs cnn_transformer_pm vs cnn_transformer."""
    models = ["KNET_rc", "cnn_transformer_pm", "cnn_transformer"]
    d = [r for r in rows if r["model_type"] in models
         and r["test_config"] == "random_rand"
         and r["sample_size"] > 0]
    if not d:
        print("\n=== Exp D: no data found ===")
        return

    # Exclude anomalous KNET_rc n=5000 seed=0 (AUROC < 0.6)
    d_filtered = [r for r in d if not (
        r["model_type"] == "KNET_rc" and r["sample_size"] == 5000 and r["val_auroc"] < 0.6
    )]

    pivot_mean = pivot_table(d_filtered, "model_type", "sample_size")
    pivot_std = pivot_table(d_filtered, "model_type", "sample_size", agg="std")

    print("\n=== Exp D: Simu16 (random_rand) Learning Curve — mean AUROC ===")
    print_pivot(pivot_mean, "mean", indices=models)
    print("\n=== Exp D: Simu16 (random_rand) Learning Curve — std AUROC ===")
    print_pivot(pivot_std, "std", indices=models)

    # Per-seed
    print("\n  Per-seed detail:")
    for model in models:
        sizes = sorted(set(r["sample_size"] for r in d if r["model_type"] == model))
        for n in sizes:
            vals = [r["val_auroc"] for r in d
                    if r["model_type"] == model and r["sample_size"] == n]
            vals_str = "  ".join(f"{v:.4f}" for v in sorted(vals))
            mean = sum(vals) / len(vals)
            print(f"    {model:22s} n={n:>7d}  mean={mean:.4f}  [{vals_str}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Exp E: Cross-config comparison (KNET_rc across all RC configs)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_cross_rc(rows):
    """KNET_rc performance across all RC difficulty levels at each data size."""
    model = "KNET_rc"
    sub = [r for r in rows if r["model_type"] == model
           and r["test_config"] in RC_CONFIGS_MAP
           and r["sample_size"] > 0]
    if not sub:
        return
    print("\n=== Cross-Config: KNET_rc across RC difficulty levels ===")
    pivot = pivot_table(sub, "test_config", "sample_size")
    print_pivot(pivot, "KNET_rc mean AUROC", columns=sorted(set(r["sample_size"] for r in sub)))


def exp_cross_markov(rows):
    """KNET performance across all Markov entropy levels at each data size."""
    model = "KNET"
    markov_rows = [r for r in rows if r["model_type"] == model
                   and r["test_config"].startswith("markov_")]
    normalized = []
    for r in markov_rows:
        entropy, data_size = get_markov_base_config(r["test_config"], r["sample_size"])
        if entropy and data_size > 0:
            normalized.append({**r, "entropy": f"H={entropy}", "data_size": data_size})
    if not normalized:
        return
    print("\n=== Cross-Config: KNET across Markov entropy levels ===")
    pivot = pivot_table(normalized, "entropy", "data_size")
    print_pivot(pivot, "KNET mean AUROC", columns=sorted(set(r["data_size"] for r in normalized)))


def exp_data_completeness(rows):
    """Print a completeness summary: (model, config, size) -> seed count."""
    from collections import Counter
    groups = Counter()
    for r in rows:
        key = (r["model_type"], r["test_config"], r["sample_size"])
        groups[key] += 1

    # Group by config, then model
    by_config = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for (model, config, size), count in groups.items():
        by_config[config][model][size] = count

    print("\n=== Data Completeness Summary ===")
    print(f"{'Config':30s} {'Model':22s} {'Sizes (N seeds)':s}")
    print("-" * 90)
    for config in sorted(by_config.keys()):
        models = by_config[config]
        for model in sorted(models.keys()):
            sizes = models[model]
            size_str = ", ".join(f"n={k}:{v}" for k, v in sorted(sizes.items()))
            print(f"{config:30s} {model:22s} {size_str}")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["A", "B", "C", "D", "E"], default=None,
                        help="Which experiment to analyze (E = completeness summary)")
    parser.add_argument("--merged", action="store_true",
                        help="Use merged CSV from all machines")
    parser.add_argument("--rc-only", action="store_true",
                        help="Exp B: RC learning curves only")
    parser.add_argument("--markov-only", action="store_true",
                        help="Exp B: Markov learning curves only")
    parser.add_argument("--completeness", action="store_true",
                        help="Print data completeness summary")
    args = parser.parse_args()

    rows = load_results(merged=args.merged)

    if args.completeness:
        exp_data_completeness(rows)
        return

    ran = False
    if args.exp in (None, "A"):
        exp_a_table(rows); ran = True
    if args.exp in (None, "B"):
        exp_b_table(rows, rc_only=args.rc_only, markov_only=args.markov_only); ran = True
    if args.exp in (None, "C"):
        exp_c_table(rows); ran = True
    if args.exp in (None, "D"):
        exp_d_table(rows); ran = True
    if args.exp in (None, "E"):
        exp_cross_rc(rows); exp_cross_markov(rows); ran = True

    # Default: show completeness
    if not ran or args.exp is None:
        exp_data_completeness(rows)


if __name__ == "__main__":
    main()
