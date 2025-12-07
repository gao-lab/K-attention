import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Callable
from einops import rearrange
from torch.utils.data import DataLoader
import pytorch_lightning as L
from datasets import load_dataset, load_from_disk, DatasetDict
from kattn.kattention import KNET_Crispr_test,CRISPRon,KNET_Crispr_test1,CRISPRon_pt,KNET_Crispr_test2,CRISPRon_base,KNET_Crispr_test3
from kattn.transformers import (
    TransformerCLSModel,
    TransformerAttnModel,
    MHAModel,
    TransformerConfig,

)
from kattn.tokenizers import RNATokenizer, RNAKmerTokenizer
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
import pdb

# basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%m-%d %H:%M:%S")
# logger = getLogger()
import os

KATTN_SRC_DIR = Path(os.environ["KATTN_SRC_DIR"])
KATTN_RESOURCES_DIR = Path(os.environ["KATTN_RESOURCES_DIR"])

from step1_select_kernel import LightningTestModel,TestDataModule,find_best_ckpt

from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math

ID2NT = {3: "A", 4: "U", 5: "C", 6: "G"}
NT2ID = {v:k for k,v in ID2NT.items()}
VALID_IDS = set(ID2NT.keys())  # {3,4,5,6}

try:
    import logomaker as lm
    HAS_LOGOMAKER = True
except Exception:
    HAS_LOGOMAKER = False

# def _find_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
#     for n, m in model.named_modules():
#         if n == name:
#             return m
#     raise KeyError(f"Module '{name}' not found. Check model.named_modules().")

def _find_module_by_name(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    dotted = dotted.strip().lstrip(".")
    # 直接路径
    cur = root
    if dotted:
        ok = True
        for part in dotted.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, torch.nn.Module):
            return cur

    # 常见前缀自动尝试
    for prefix in ("model", "net", "backbone", "module"):
        cur = root
        ok = True
        for part in (prefix + "." + dotted).split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, torch.nn.Module):
            return cur

    # 后缀匹配（唯一即可）
    all_named = dict(root.named_modules())
    if dotted in all_named:
        return all_named[dotted]
    candidates = [n for n in all_named if n.endswith(dotted)]
    if len(candidates) == 1:
        return all_named[candidates[0]]
    if len(candidates) > 1:
        raise KeyError(f"Ambiguous module suffix '{dotted}'. Candidates:\n" + "\n".join(candidates[:20]))
    raise KeyError(f"Module '{dotted}' not found.")

# ------- 小工具：找到给定 kattn 子模块所属的 self_kattn 父模块（用于取 H、D、reverse） -------
def _find_parent_self_kattn(root: torch.nn.Module, dotted: str):
    dotted = dotted.lstrip(".")
    if dotted.startswith("model."):
        dotted = dotted[len("model."):]
    parts = dotted.split(".")
    # 逐级尝试父路径
    for cut in range(len(parts), 0, -1):
        candidate = ".".join(parts[:cut])
        try:
            mod = _find_module_by_name(root, candidate)
        except KeyError:
            continue
        if getattr(mod, "__class__", type(None)).__name__ == "self_kattn":
            return mod, candidate
    # 兜底：如果当前本身是 self_kattn
    try:
        mod = _find_module_by_name(root, dotted)
        if getattr(mod, "__class__", type(None)).__name__ == "self_kattn":
            return mod, dotted
    except KeyError:
        pass
    return None, None

@torch.no_grad()
def collect_attn_logits_for_head(
    model: torch.nn.Module,
    loader,
    DEVICE,
    layer_name: str,       # "Kattention1" 或 "Kattention1.kattn"（不用写前缀 "model."）
    head_idx: int,         # 指定的 head 索引
    max_batches: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
      logits_all: [N, Q, K]       # 只取指定 head 的 attn_logits，且已按 self_kattn.forward 进行 reverse 与 /sqrt(h_d) 处理
      labels_all: [N]
      ids_all:    [N, L]          # 原始整数 token 序列
    """
    model.eval().to(DEVICE)

    # 解析模块：既支持传父名也支持传子名
    try:
        mod = _find_module_by_name(model, layer_name)
    except KeyError:
        # 只有当名称未以 ".kattn" 结尾时，才补一次
        if not layer_name.endswith(".kattn"):
            mod = _find_module_by_name(model, layer_name + ".kattn")
        else:
            raise

    # 如果拿到的是 self_kattn，就取其 .kattn
    if getattr(mod, "__class__", type(None)).__name__ == "self_kattn":
        if not hasattr(mod, "kattn"):
            raise KeyError(f"Module '{layer_name}' is self_kattn but has no 'kattn' submodule.")
        mod_kattn = mod.kattn
        parent = mod
    else:
        # 是 KattentionV4，查找父 self_kattn 以获取 H、D、reverse
        mod_kattn = mod
        parent, _ = _find_parent_self_kattn(model, layer_name)

    if getattr(mod_kattn, "__class__", type(None)).__name__ != "KattentionV4":
        raise TypeError(f"Resolved module is not KattentionV4 (got {type(mod_kattn).__name__})")

    if parent is None or getattr(parent, "__class__", type(None)).__name__ != "self_kattn":
        raise RuntimeError(f"Cannot locate parent self_kattn for '{layer_name}' to read k_n/h_d/reverse.")

    H  = int(parent.k_n)
    D  = int(parent.h_d)
    rv = bool(getattr(parent, "reverse", False))

    # 使用 forward hook 抓原始 attn_logits
    buffer_logits: List[torch.Tensor] = []

    def _hook(_m, _in, out):
        # out 应含 "attn_logits"
        A = out["attn_logits"]  # [B,H,L,L] 或 [B,L,L]
        if A.dim() == 3:        # 单头情形
            B, Lq, Lk = A.shape
            A = A.unsqueeze(1).expand(B, H, Lq, Lk)
        elif A.dim() == 4:
            if A.size(1) != H:
                # 轻度容错：若 H 不一致，以 A.size(1) 为准（但会影响 head_idx 的范围）
                H_local = A.size(1)
                if head_idx >= H_local:
                    raise IndexError(f"head_idx {head_idx} out of range H={H_local}")
        else:
            raise RuntimeError(f"attn_logits dim={A.dim()} (expect 3 or 4)")

        # reverse 处理（与 self_kattn.forward 一致）
        if rv:
            A = A.flip([-1])

        # /sqrt(h_d) 缩放（与 self_kattn.forward 一致）
        # A = A / math.sqrt(float(D))

        if head_idx >= A.size(1):
            raise IndexError(f"head_idx {head_idx} out of range H={A.size(1)}")
        buffer_logits.append(A[:, head_idx, ...].detach().cpu())  # [B,Q,K]

    handle = mod_kattn.register_forward_hook(_hook)

    labels_all, ids_all = [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        # IDs：可能是 [B,L] 或 one-hot [B,L,C]，统一转成整数 token
        ids = batch["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids)
        if ids.dim() == 3:       # one-hot -> ids
            ids_int = ids.argmax(dim=-1)
        else:
            ids_int = ids.long()
        ids_all.append(ids_int.detach().cpu())

        # label（按你的数据字段）
        if "mean_eff" in batch:
            y = batch["mean_eff"]
        elif "me" in batch:
            y = batch["me"]
        else:
            raise KeyError("Batch needs 'mean_eff' or 'me'.")
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        labels_all.append(y.view(-1).detach().cpu())

        # 前向（触发 hook）
        y_for_loss = y.to(DEVICE).float().view(-1)
        ids_for_model = batch["input_ids"].to(DEVICE)
        crisproff = batch.get("CRISPRoff_score", None)
        if isinstance(crisproff, torch.Tensor):
            crisproff = crisproff.to(DEVICE)
        _ = model(ids_for_model, y_for_loss, crisproff)

    handle.remove()

    if len(buffer_logits) == 0:
        raise RuntimeError(f"No logits captured for layer '{layer_name}'. "
                           f"Make sure the layer name is correct and the forward pass hits that module.")

    # 拼接
    logits_all = torch.cat(buffer_logits, dim=0)  # [N,Q,K]
    labels_all = torch.cat(labels_all, dim=0)     # [N]
    ids_all    = torch.cat(ids_all, dim=0)        # [N,L]
    return logits_all, labels_all, ids_all

def argmax_over_qk(logits_qk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    logits_qk: [N,Q,K]
    返回:
      max_vals: [N]
      max_q:    [N]   （行为索引）
      max_k:    [N]   （列为索引）
    """
    N, Q, K = logits_qk.shape
    flat = logits_qk.view(N, -1)                # [N,QK]
    idx  = flat.argmax(dim=1)                   # [N]
    max_vals = flat.max(dim=1).values           # [N]
    max_q = idx // K
    max_k = idx % K
    return max_vals, max_q, max_k

def plot_max_hist(max_vals: torch.Tensor, out_png: str | Path, bins: int = 50, title: str = ""):
    mv = max_vals.detach().cpu().numpy()
    plt.figure(figsize=(6,4), dpi=160)
    sns.histplot(mv, bins=bins, kde=True)
    plt.xlabel("max attn_logits (per sample)")
    plt.ylabel("count")
    plt.title(title or "Distribution of max attn_logits")
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.as_posix(), bbox_inches="tight"); plt.close()

def plot_corr_scatter(max_vals: torch.Tensor, labels: torch.Tensor, out_png: str | Path, title: str = ""):
    x = max_vals.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    # Pearson & Spearman
    pr, pp = stats.pearsonr(x, y) if np.std(x) > 0 else (0.0, 1.0)
    sr, sp = stats.spearmanr(x, y)
    # 回归线
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]  # y ≈ a*x + b
    yhat = a*x + b

    plt.figure(figsize=(5.5,5), dpi=160)
    plt.scatter(x, y, s=10, alpha=0.6)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = a*xs + b
    plt.plot(xs, ys, lw=2)
    plt.xlabel("max attn_logits")
    plt.ylabel("label")
    plt.title(title or f"Corr: pearson={pr:.3f} (p={pp:.1e}), spearman={sr:.3f} (p={sp:.1e})")
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.as_posix(), bbox_inches="tight"); plt.close()

    return {"pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp, "slope": a, "intercept": b}

def filter_by_threshold(
    max_vals: torch.Tensor,
    *,
    value: float | None = None,
    percentile: float | None = None,
    mode: str = "ge",   
):
    """
    根据阈值过滤样本，返回 (keep_mask, threshold)。

    参数
    ----
    max_vals : Tensor [N]
        每个样本的打分（如 max attn_logits）。
    value : float | None
        直接使用的数值阈值；与 percentile 互斥，优先使用 percentile。
    percentile : float | None
        使用分位数阈值（0~100）。若传入，将覆盖 value。
    mode : {"ge","gt","le","lt"}
        过滤模式：
          - "ge": 保留 >= 阈值 的样本（默认）
          - "gt": 保留 >  阈值 的样本
          - "le": 保留 <= 阈值 的样本
          - "lt": 保留 <  阈值 的样本
    返回
    ----
    keep_mask : np.ndarray[bool] 形状 [N]
    thr       : float            实际使用的阈值
    """
    mv = max_vals.detach().cpu().numpy()

    if percentile is not None:
        thr = float(np.percentile(mv, percentile))
    elif value is not None:
        thr = float(value)
    else:
        raise ValueError("Provide either `percentile` or `value`.")

    mode = mode.lower()
    if mode in ("ge", ">="):
        keep = mv >= thr
    elif mode in ("gt", ">"):
        keep = mv > thr
    elif mode in ("le", "<="):
        keep = mv <= thr
    elif mode in ("lt", "<"):
        keep = mv < thr
    else:
        raise ValueError("`mode` must be one of {'ge','gt','le','lt'}.")

    return keep, thr


def plot_pos_heatmap(max_q: torch.Tensor, max_k: torch.Tensor, Q: int, K: int, keep_mask: np.ndarray,
                     out_png: str | Path, title: str = ""):
    q = max_q.detach().cpu().numpy()[keep_mask]
    k = max_k.detach().cpu().numpy()[keep_mask]
    mat = np.zeros((Q, K), dtype=int)
    for qi, ki in zip(q, k):
        if 0 <= qi < Q and 0 <= ki < K:
            mat[int(qi), int(ki)] += 1

    plt.figure(figsize=(6,5), dpi=160)
    sns.heatmap(mat, cmap="Reds")
    plt.xlabel("k index"); plt.ylabel("q index")
    plt.title(title or f"Position heatmap (N={keep_mask.sum()})")
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.as_posix(), bbox_inches="tight",dpi=300,format="svg")
    plt.close()

def _slice_tokens_1d(tokens_1d: np.ndarray, center: int, win_len: int) -> np.ndarray:
    """从一维 token 序列中裁切窗口，返回实际长度可能 < win_len（边界裁剪）"""
    L = tokens_1d.shape[0]
    half = win_len // 2
    start = max(0, center - half)
    end   = min(L, center + half + (0 if win_len%2==0 else 1))
    return tokens_1d[start:end]

def tokens_to_pwm(
    seqs: list[np.ndarray],
    base_order: list[str] = ["A","C","G","U"],  # 想按 A,U,C,G 就改这里
) -> pd.DataFrame:
    """
    seqs: 列表，每个元素是一维 token 数组（长度相同或不同均可）
          - 若位置 token 不在 VALID_IDS（如 -1 或 padding），该位置不计数（0 贡献）
    base_order: PWM 的列顺序，决定 4 列分别是谁
    return: DataFrame 形状 [L, 4]，列名为 base_order
    """
    if len(seqs) == 0:
        raise ValueError("No sequences to build PWM.")

    # 对齐长度（取最短，简单稳妥；也可改为中心对齐裁剪）
    min_len = min(len(s) for s in seqs)
    clipped = [s[:min_len] for s in seqs]
    L = min_len

    # 列映射：A->0, C->1, G->2, U->3（按 base_order 决定）
    base_to_col = {b: i for i, b in enumerate(base_order)}

    M = np.zeros((L, len(base_order)), dtype=float)  # 计数矩阵

    for s in clipped:
        for i, tid in enumerate(s):
            t = int(tid)
            if t not in VALID_IDS:
                # 无效/边缘/未知 -> 不计数（0 贡献）
                continue
            base = ID2NT[t]           # "A"/"U"/"C"/"G"
            if base not in base_to_col:
                # 不在列序里（一般不会发生），跳过
                continue
            col = base_to_col[base]   # col ∈ {0,1,2,3}
            M[i, col] += 1.0

    # 归一化为概率
    denom = M.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0  # 防 0
    P = M / denom
    return pd.DataFrame(P, columns=base_order)

# def plot_pwm_logo(pwm_df: pd.DataFrame, out_png: str | Path, title: str = ""):
#     out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
#     if HAS_LOGOMAKER:
#         plt.figure(figsize=(max(4, pwm_df.shape[0]*0.2), 2.2), dpi=160)
#         logo = lm.Logo(pwm_df, color_scheme="classic")
#         plt.title(title or "Sequence logo")
#         plt.xlabel("position"); plt.ylabel("frequency")
#         plt.tight_layout()
#         plt.savefig(out_png.as_posix(), bbox_inches="tight"); plt.close()
#     else:
#         # fallback: heatmap
#         plt.figure(figsize=(max(4, pwm_df.shape[0]*0.2), 2.2), dpi=160)
#         sns.heatmap(pwm_df.T, cmap="mako", cbar=True)
#         plt.title((title or "PWM (heatmap)") + " [install logomaker for logo]")
#         plt.ylabel("base"); plt.xlabel("position")
#         plt.tight_layout()
#         plt.savefig(out_png.as_posix(), bbox_inches="tight"); plt.close()

def pwm_to_information_matrix(
    pwm_df: pd.DataFrame,
    background: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    将 PWM 频率矩阵 (L x 4) 转换为信息矩阵 (L x 4)，元素为 p*log2(p/bg)；p=0 时置 0。
    background: 背景分布，默认均匀 {"A":0.25,"C":0.25,"G":0.25,"U":0.25}
    """
    if background is None:
        # 列名可能是 ["A","C","G","U"] 或 ["A","U","C","G"]，按列名均匀赋值
        background = {col: 1.0 / pwm_df.shape[1] for col in pwm_df.columns}

    P = pwm_df.to_numpy(dtype=float)             # (L,4)
    cols = list(pwm_df.columns)
    bg = np.array([background[c] for c in cols], dtype=float)  # (4,)

    # 避免 log(0)：p=0 时令项为 0
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.where(P > 0, P * (np.log(P / bg) / np.log(2.0)), 0.0)  # bits

    info_df = pd.DataFrame(R, columns=cols)
    return info_df

def plot_pwm_logo(
    pwm_df: pd.DataFrame,
    out_png: str | Path,
    title: str = "",
    background: dict[str, float] | None = None,
):
    """
    以“相对背景 0.25 的信息量（bits）”为纵轴绘制 sequence logo。
    - 若安装了 logomaker，则画标准 logo；否则 fallback 为信息矩阵热力图。
    """
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)

    # 1) 频率 -> 信息矩阵（p*log2(p/bg)）
    info_df = pwm_to_information_matrix(pwm_df, background=background)

    # 2) y 轴上限：各位置总信息量的最大值（并留一点边距）
    total_bits_per_pos = info_df.sum(axis=1).to_numpy()   # (L,)
    ymax = float(total_bits_per_pos.max() if total_bits_per_pos.size else 0.0)
    # ylim = ymax * 1.1 if ymax > 0 else 1.0
    ylim =2
    if HAS_LOGOMAKER:
        # 使用 logomaker 直接画“信息矩阵”logo
        plt.figure(figsize=(max(4, info_df.shape[0] * 0.22), 2.4), dpi=160)
        # 经典配色（也可自定义 color_scheme）
        logo = lm.Logo(info_df, color_scheme="classic")
        plt.ylim(0, ylim)
        plt.xlabel("position")
        plt.ylabel("IC")
        if title:
            plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(out_png.as_posix(), bbox_inches="tight"); plt.close()
    else:
        # 没有 logomaker：用热力图展示信息矩阵
        plt.figure(figsize=(max(4, info_df.shape[0] * 0.22), 2.4), dpi=160)
        sns.heatmap(info_df.T, cmap="viridis", cbar=True)
        plt.xlabel("position"); plt.ylabel("base")
        if title:
            plt.title(f"{title}  [info bits vs bg=0.25]", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_png.as_posix(), bbox_inches="tight"); plt.close()

def plot_logits_per_sample(
    logits_qk: torch.Tensor,        # [N, Q, K]
    keep_mask: np.ndarray,          # bool 数组 [N]，为 True 的样本会被绘图
    out_dir: str | Path,
    max_plots: int | None = None,   # 限制最多绘多少张；None 表示全部
    dpi: int = 160,
):
    """
    对满足 keep_mask 的每个样本，分别绘制一张 [Q,K] 的原始 logits 热力图（不做任何处理）。
    输出：<out_dir>/sample_{idx}.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logits_np = logits_qk.detach().cpu().numpy()  # [N,Q,K]
    idxs = np.where(keep_mask)[0]
    if max_plots is not None:
        idxs = idxs[:max_plots]

    for idx in idxs:
        mat = logits_np[idx]  # [Q,K]
        plt.figure(figsize=(6, 5), dpi=dpi)
        sns.heatmap(mat, cmap="viridis", cbar=True)  # 原始值，无遮挡/无裁剪
        plt.title(f"sample #{int(idx)} logits [QxK]={mat.shape[0]}x{mat.shape[1]}")
        plt.xlabel("k index")
        plt.ylabel("q index")
        out_png = out_dir / f"sample_{int(idx)}.png"
        plt.tight_layout()
        plt.savefig(out_png.as_posix(), bbox_inches="tight")
        plt.close()

    print(f"[per-sample logits] saved {len(idxs)} figures to: {out_dir.resolve()}")

def plot_qk_heatmaps_and_distribution(
    T: torch.Tensor,
    out_dir: str = "./qk_plots",
    max_heatmaps: int = 10,
    dpi: int = 150
):
    """
    绘制并保存 (N, q, k) 张量的热图与整体分布图。

    Parameters
    ----------
    T : torch.Tensor
        形状为 (N, q, k) 的张量。
    out_dir : str
        图片输出目录（若不存在将自动创建）。
    max_heatmaps : int
        最多绘制多少个 N 维上的切片热图（从 0 开始）。
    dpi : int
        保存图片分辨率 DPI。

    Returns
    -------
    heatmap_paths : list[str]
        已保存的前若干个切片热图路径。
    dist_path : str
        已保存的整体扁平化分布直方图路径。
    """
    if not isinstance(T, torch.Tensor):
        raise TypeError("T must be a torch.Tensor.")
    if T.ndim != 3:
        raise ValueError(f"Expected shape (N, q, k), got {tuple(T.shape)}.")

    os.makedirs(out_dir, exist_ok=True)

    N, q, k = T.shape
    T_cpu = T.detach().cpu()

    # ---- (1) 前 max_heatmaps 个切片热图 ----
    to_plot = min(max_heatmaps, N)
    heatmap_paths = []
    for i in range(to_plot):
        slice_np = T_cpu[i].numpy()

        # 处理非有限值（inf/NaN），避免绘图报错
        finite_mask = np.isfinite(slice_np)
        if not finite_mask.all():
            finite_vals = slice_np[finite_mask]
            repl = float(finite_vals.mean()) if finite_vals.size > 0 else 0.0
            slice_np = np.where(finite_mask, slice_np, repl)

        plt.figure()
        plt.imshow(slice_np, aspect="auto")
        plt.colorbar()
        plt.title(f"Heatmap (N index {i}) — shape: {q}×{k}")
        path = os.path.join(out_dir, f"heatmap_N{i}.png")
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close()
        heatmap_paths.append(path)

    # ---- (2) 整体扁平化分布直方图 ----
    flat = T_cpu.reshape(-1).numpy()
    flat = flat[np.isfinite(flat)]  # 去除非有限值

    plt.figure()
    plt.hist(flat, bins=100)
    plt.title(f"Flattened value distribution — N={N}, q={q}, k={k}, total={flat.size}")
    dist_path = os.path.join(out_dir, "flattened_distribution.png")
    plt.savefig(dist_path, bbox_inches="tight", dpi=dpi)
    plt.close()

    return heatmap_paths, dist_path

def run_kernel_peak_and_logo(
    model: torch.nn.Module,
    loader,
    device,
    layer_name: str,       # "model.Kattention1" 或 "model.Kattention1.kattn"
    head_idx: int,         # 指定 kernel/head
    percentile: Optional[float] = 90.0,  # 过滤阈值（分位数，None 表示不用过滤）
    value_thr: Optional[float] = None,   # 或者直接给数值阈值
    win_len_q: int = 11,   # q窗口长度（奇数为中心对齐）
    win_len_k: int = 11,   # k窗口长度
    out_dir: str | Path = "./kernel_peak_logo",
    mode: str = 'ge'
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("==== modules under LightningTestModel ====")
    for n, m in model.named_modules():
        if m.__class__.__name__ in {"self_kattn", "KattentionV4"}:
            print(f"{n:50s}  ->  {type(m).__name__}")
    # 1) 收集 attn_logits（指定 head）与标签、原始 ids
    logits_qk, labels, ids = collect_attn_logits_for_head(
        model, loader, layer_name=layer_name, head_idx=head_idx, DEVICE=device
    )  # logits_qk: [N,Q,K], labels:[N], ids:[N,L]

    N, Q, K = logits_qk.shape
    # heatmaps, dist = plot_qk_heatmaps_and_distribution(logits_qk, out_dir=f"{out_dir}/qk_plots", max_heatmaps=10)

    L = ids.shape[1]
    print(f"[info] collected logits: N={N}, Q={Q}, K={K}, L={L}")

    # 2) 最大值与位置
    max_vals, max_q, max_k = argmax_over_qk(logits_qk)  # [N], [N], [N]
    # 保存基本表
    df_basic = pd.DataFrame({
        "max_val": max_vals.numpy(),
        "max_q":   max_q.numpy(),
        "max_k":   max_k.numpy(),
        "label":   labels.numpy()
    })
    df_basic.to_csv(out_dir / f"peaks_{layer_name.replace('.','_')}_h{head_idx}.csv", index=False)

    # 3) 最大值分布
    plot_max_hist(max_vals, out_dir / f"max_hist_{layer_name.replace('.','_')}_h{head_idx}.png",
                  title=f"{layer_name} head={head_idx}: max logits")

    # 4) 相关性 + 散点
    corr_info = plot_corr_scatter(max_vals, labels, out_png=out_dir / f"max_vs_label_{layer_name.replace('.','_')}_h{head_idx}.png",
                                  title=f"{layer_name} head={head_idx}  (max logits vs label)")
    pd.DataFrame([corr_info]).to_csv(out_dir / f"corr_{layer_name.replace('.','_')}_h{head_idx}.csv", index=False)

    # 5) 过滤样本 & 位置热图
    if percentile is not None or value_thr is not None:
        keep_mask, thr = filter_by_threshold(max_vals, value=value_thr, percentile=percentile, mode=mode)
        print(f"[info] filter kept {keep_mask.sum()}/{len(keep_mask)} samples, thr={thr:.4f}")
        plot_pos_heatmap(max_q, max_k, Q=Q, K=K, keep_mask=keep_mask,
                         out_png=out_dir / f"pos_heatmap_{layer_name.replace('.','_')}_h{head_idx}.svg",
                         title='')

        # plot_logits_per_sample(
        #     logits_qk=logits_qk,
        #     keep_mask=keep_mask,
        #     out_dir=out_dir / f"per_sample_logits/{layer_name.replace('.','_')}_h{head_idx}",
        #     max_plots=None,   # 或者给个整数，比如 100
        # )
        # 6) 提取两段序列 → PWM → logo
        kept_idx = np.where(keep_mask)[0]
        seqs_q, seqs_k = [], []
        for i in kept_idx:
            tokens = ids[i].numpy()   # [L] int
            qpos = int(max_q[i])-2; kpos = int(max_k[i])-2
            q_center = min(max(qpos, 0), L-1) 
            k_center = min(max(kpos, 0), L-1)
            half_q = win_len_q // 2
            half_k = win_len_k // 2

            # 有效中心：必须能完整覆盖一个 win_len 窗口，不触及原序列两端
            q_valid = (q_center >= half_q) and (q_center <= L - 1 - half_q)
            k_valid = (k_center >= half_k) and (k_center <= L - 1 - half_k)

            # 只在有效时才计入 PWM；无效样本的贡献=0（等价于跳过）
            if q_valid & k_valid:
                seq_q = _slice_tokens_1d(tokens, q_center, win_len_q)
                # 只保留 RNA 四字母（3,4,5,6），其余 token 视作空（不计数）
                seq_q = np.array([t if int(t) in VALID_IDS else -1 for t in seq_q], dtype=int)
                seqs_q.append(seq_q)

                seq_k = _slice_tokens_1d(tokens, k_center, win_len_k)
                seq_k = np.array([t if int(t) in VALID_IDS else -1 for t in seq_k], dtype=int)
                seqs_k.append(seq_k)
        pwm_q = tokens_to_pwm(seqs_q, base_order=["A","C","G","U"])
        pwm_k = tokens_to_pwm(seqs_k, base_order=["A","C","G","U"])
        pwm_q.to_csv(out_dir / f"pwm_q_{layer_name.replace('.','_')}_h{head_idx}.csv", index=False)
        pwm_k.to_csv(out_dir / f"pwm_k_{layer_name.replace('.','_')}_h{head_idx}.csv", index=False)

        plot_pwm_logo(pwm_q, out_dir / f"logo_q_{layer_name.replace('.','_')}_h{head_idx}.png",
                      title=f"{layer_name} head={head_idx}  (Q-window PWM)")
        plot_pwm_logo(pwm_k, out_dir / f"logo_k_{layer_name.replace('.','_')}_h{head_idx}.png",
                      title=f"{layer_name} head={head_idx}  (K-window PWM)")
    else:
        print("[info] no filtering performed (percentile/value_thr are both None)")


# ----------------- USAGE -----------------
if __name__ == "__main__":
    # assume you have wrapped_model and loader in scope; e.g. load checkpoint etc.
    # Example:
    # wrapped_model = LightningTestModel.load_from_checkpoint("best.ckpt").to(DEVICE).eval()
    # data_module.setup("test"); loader = data_module.test_dataloader()
    # Then:
    # results = run_full_analysis(wrapped_model, loader, out_dir="./kernel_analysis", max_batches=None)

    # nucleotide id -> char（按你的数据改）

    test_config = 'doench2014-Hs'
    model_type = 'KNET_Crispr'
    num_ds = 'set2'
    version = '3'
    RESULTS_DIR = f'../../results/Crispr/{test_config}/{model_type}/{num_ds}/{version}/'
    OUT_DIR = os.path.join(RESULTS_DIR, "kernel_analysis")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 载入最优权重（如果你已经有 wrapped_model 可直接复用）
    best_ckpt = find_best_ckpt(RESULTS_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 如果 LightningTestModel 在 ckpt里保存了超参，这里可不手动传
    wrapped_model = LightningTestModel.load_from_checkpoint(best_ckpt, map_location=device, test_config=test_config, model_type=model_type,num_ds=num_ds)
    wrapped_model.to(device).eval()

    data_module = TestDataModule(
        dst_name=KATTN_SRC_DIR / "kattn/general_crispr",
        dst_config=test_config,
        dst_dir=KATTN_RESOURCES_DIR,
        tokenizer=wrapped_model.tokenizer,
        cache_dir="_cache_dsts",
        batch_size=512,
        num_procs=4,
        num_set=num_ds,
    )

    data_module.setup(stage="train")  # 或者 "validate"，看你想用哪个 split
    loader = data_module.train_dataloader()  # 若用验证集: data_module.val_dataloader()
    
    # "ge"(>=), "gt"(>), "le"(<=), "lt"(<)
    head_idx = 0
    mode = 'ge'
    kattn = "Kattention2.kattn"

    run_kernel_peak_and_logo(
        model=wrapped_model,
        device=device,
        loader=loader,
        layer_name=kattn,   # 或 "model.Kattention1.kattn"
        head_idx=head_idx,                      # 指定要分析的 kernel/head
        percentile=50,                  # 取 max_logits 的 top 10% 样本
        value_thr=None,                   # 或直接用固定阈值
        win_len_q=5,
        win_len_k=5,
        mode=mode,
        out_dir=f"../draw/kernel_peak_logo_K1_h12/{test_config}/{kattn}/{head_idx}/{mode}"
    )

