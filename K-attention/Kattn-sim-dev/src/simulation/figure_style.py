"""
figure_style.py — Publication-quality figure style for K-attention paper.

Follows standards of Nature / Briefings in Bioinformatics:
  - PDF fonttype=42  →  fonts embedded as TrueType (editable in Illustrator/Inkscape)
  - Arial / Helvetica (sans-serif), clean and journal-standard
  - Font sizes calibrated for double-column layout (7–9 pt range)
  - Colorblind-friendly palette (Wong 2011, used by Nature Methods)
  - Minimal spines (top/right removed), light grid
  - 300 DPI PNG + vector PDF

Usage:
    import figure_style as fs
    fs.apply()                           # call once before any plt calls

    fig, ax = plt.subplots(figsize=fs.FIG_2COL)
    ax.plot(..., color=fs.C['blue'], linewidth=fs.LW)
    fs.save(fig, figures_dir / "my_figure")  # writes .pdf and .png
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# ── PDF / PS font embedding ────────────────────────────────────────────────
# fonttype=42 → TrueType embedded: text remains editable in Adobe Illustrator,
# Inkscape, and AI tools that ingest vector PDFs.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ── Figure dimensions (inches) ─────────────────────────────────────────────
# Nature: single col = 88 mm, double col = 180 mm, max height = 240 mm
FIG_1COL = (3.46, 2.76)    # 88 × 70 mm  — single panel
FIG_2COL = (7.09, 3.15)    # 180 × 80 mm — two panels side by side
FIG_2COL_TALL = (7.09, 4.0)
FIG_SQUARE = (3.46, 3.46)

# ── Font sizes (pt) ────────────────────────────────────────────────────────
FS_TITLE  = 9    # panel/figure title
FS_LABEL  = 8    # axis labels
FS_TICK   = 7    # tick labels
FS_LEGEND = 7    # legend text
FS_ANNOT  = 6    # heatmap annotations, p-value labels

# ── Line / marker geometry ─────────────────────────────────────────────────
LW        = 1.5   # main line width
LW_THIN   = 0.8   # spine, grid, error-bar cap
MS        = 5     # marker size
CAPSIZE   = 3     # error bar cap size
SCATTER_S = 18    # scatter dot area (for seed overlay)

# ── Wong (2011) colorblind-safe palette ────────────────────────────────────
# DOI: 10.1038/nmeth.1618  — widely used in Nature journals
C = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "sky":     "#56B4E9",
    "yellow":  "#F0E442",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "black":   "#000000",
    "gray":    "#999999",
}

# ── Per-model color assignments ────────────────────────────────────────────
# Primary model (KNET) gets blue; ablation gets red/orange; baselines spread.
MODEL_COLORS = {
    "KNET_rc":              C["blue"],
    "KNET":                 C["blue"],
    "KNET_uncons_rc":       C["red"],
    "KNET_uncons":          C["red"],
    "cnn_transformer_pm":   C["green"],
    "cnn_transformer":      C["orange"],
    "transformer_cls":      C["purple"],
    "transformer_cls_kmer": C["sky"],
    "cnn":                  C["black"],
    "mha":                  C["gray"],
}

# ── Display names ──────────────────────────────────────────────────────────
MODEL_NAMES = {
    "KNET_rc":              "KNET (RC)",
    "KNET":                 "KNET",
    "KNET_uncons_rc":       "KNET-uncons (RC)",
    "KNET_uncons":          "KNET-uncons",
    "cnn_transformer_pm":   "CNN-TF (matched)",
    "cnn_transformer":      "CNN-TF",
    "transformer_cls":      "Transformer",
    "transformer_cls_kmer": "Transformer (k-mer)",
    "mha":                  "MHA",
    "cnn":                  "CNN",
}


def apply() -> None:
    """Apply global rcParams for publication-quality figures."""
    mpl.rcParams.update({
        # ── font ──────────────────────────────────────────────────────────
        "pdf.fonttype":          42,
        "ps.fonttype":           42,
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":             FS_TICK,
        "axes.labelsize":        FS_LABEL,
        "axes.titlesize":        FS_TITLE,
        "xtick.labelsize":       FS_TICK,
        "ytick.labelsize":       FS_TICK,
        "legend.fontsize":       FS_LEGEND,
        "legend.title_fontsize": FS_LEGEND,
        # ── lines ─────────────────────────────────────────────────────────
        "lines.linewidth":       LW,
        "lines.markersize":      MS,
        "patch.linewidth":       LW_THIN,
        # ── spines / ticks ────────────────────────────────────────────────
        "axes.linewidth":        LW_THIN,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "xtick.major.width":     LW_THIN,
        "ytick.major.width":     LW_THIN,
        "xtick.major.size":      3.0,
        "ytick.major.size":      3.0,
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        # ── grid ──────────────────────────────────────────────────────────
        "axes.grid":             True,
        "grid.linewidth":        0.4,
        "grid.alpha":            0.35,
        "grid.color":            "#cccccc",
        # ── legend ────────────────────────────────────────────────────────
        "legend.frameon":        False,
        "legend.borderpad":      0.3,
        "legend.handlelength":   1.5,
        "legend.handletextpad":  0.4,
        # ── figure / saving ───────────────────────────────────────────────
        "figure.dpi":            100,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.05,
        "figure.constrained_layout.use": False,
    })


def save(fig: "plt.Figure", stem, formats=("pdf", "png")) -> None:
    """Save figure to PDF and PNG (300 dpi).

    Args:
        fig:     matplotlib Figure object
        stem:    Path without extension, e.g. figures_dir / "exp_a_heatmap"
        formats: tuple of extensions to write
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if fmt == "png":
            kwargs["dpi"] = 300
        fig.savefig(out, **kwargs)
        print(f"  Saved: {out.resolve()}")
