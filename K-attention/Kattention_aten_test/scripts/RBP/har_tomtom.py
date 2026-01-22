#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import pdb
# sys.path.append("../../corecode/")
from build import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.colors as colors
plt.switch_backend('agg')
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import h5py
from typing import Dict, Tuple, Callable, Union
from einops import rearrange
from collections import defaultdict

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

try:
    import logomaker as lm
    HAS_LOGOMAKER = True
except Exception:
    HAS_LOGOMAKER = False
# -----------------------------
# 基础：sequence/structure 解码与 k-mer 抽取
# -----------------------------
BASES = ["A", "C", "G", "U"]

def decode_seq_and_struct(region: np.ndarray, struct_threshold: float = 0.233) -> Tuple[str, str]:
    """
    region: (L,5), 前4列 one-hot A/C/G/U，第5列 icSHAPE
    struct: >=阈值 -> U, <阈值 -> P
    """
    assert region.ndim == 2 and region.shape[1] == 5
    onehot = region[:, :4]
    struct_vals = region[:, 4]

    base_idx = onehot.argmax(axis=1)
    seq = "".join(BASES[i] for i in base_idx)
    struct = "".join("U" if v >= struct_threshold else "P" for v in struct_vals)
    return seq, struct

def extract_kmers_from_regions(regions: List[np.ndarray], k: int = 6, struct_threshold: float = 0.233) -> List[Dict[str, Any]]:
    """
    返回 list[dict]:
      {"seq":..., "struct":..., "region_id":..., "start":...}
    """
    kmers: List[Dict[str, Any]] = []
    for ridx, region in enumerate(regions):
        seq, struct = decode_seq_and_struct(region, struct_threshold)
        L = len(seq)
        if L < k:
            continue
        for start in range(0, L - k + 1):
            kmers.append({
                "seq": seq[start:start + k],
                "struct": struct[start:start + k],
                "region_id": ridx,
                "start": start,
            })
    return kmers

def hamming_distance(s1: str, s2: str) -> int:
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def cluster_kmers_seq(kmers: List[Dict[str, Any]], max_seq_mismatch: int = 1) -> List[List[Dict[str, Any]]]:
    """
    贪心聚簇：与簇代表(第一个 kmer)的序列 Hamming 距离 <= max_seq_mismatch
    """
    clusters: List[List[Dict[str, Any]]] = []
    for kmer in kmers:
        assigned = False
        for cl in clusters:
            rep = cl[0]["seq"]
            if hamming_distance(kmer["seq"], rep) <= max_seq_mismatch:
                cl.append(kmer)
                assigned = True
                break
        if not assigned:
            clusters.append([kmer])
    return clusters

def build_seq_pwm_for_cluster(cluster: List[Dict[str, Any]], k: int) -> np.ndarray:
    """
    cluster 内所有 k-mer 统计 seq PWM: (k,4)
    """
    counts = np.zeros((k, 4), dtype=float)
    for item in cluster:
        seq = item["seq"]
        assert len(seq) == k
        for pos, base in enumerate(seq):
            counts[pos, BASES.index(base)] += 1.0
    pwm = counts / np.clip(counts.sum(axis=1, keepdims=True), 1e-9, None)
    return pwm

# -----------------------------
# ATtRACT 读取
# -----------------------------
def load_pwms_from_file(pwm_txt_path: str) -> Dict[str, np.ndarray]:
    """
    解析 ATtRACT 样式 pwm.txt：
      >630    6
      0.1 0.2 0.3 0.4
      ...
    返回: {motif_id: (L,4)}
    """
    motifs: Dict[str, np.ndarray] = {}
    cur_id: Optional[str] = None
    rows: List[List[float]] = []

    with open(pwm_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None and rows:
                    motifs[cur_id] = np.array(rows, dtype=float)
                parts = line[1:].split()
                cur_id = parts[0]
                rows = []
            else:
                rows.append([float(x) for x in line.split()])

    if cur_id is not None and rows:
        motifs[cur_id] = np.array(rows, dtype=float)
    return motifs

def build_rbp2motif_ids(meta_path: str) -> Dict[str, List[str]]:
    """
    解析 ATtRACT_db.txt（制表符分隔）
    第一列: RBP 名
    倒数第二列: motif_id
    返回: {rbp: [motif_id1, motif_id2, ...]}
    """
    from collections import defaultdict
    rbp2motifs = defaultdict(list)

    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            rbp = cols[0]
            motif_id = cols[-2]
            if motif_id not in rbp2motifs[rbp]:
                rbp2motifs[rbp].append(motif_id)

    return dict(rbp2motifs)

# -----------------------------
# TOMTOM：PWM -> MEME 格式 + 调用与解析
# -----------------------------
def pwm_to_meme_str_dna(motif_name: str, pwm: np.ndarray) -> str:
    """
    (L,4) PWM -> MEME motif, using A C G T.
    约定：你输入列顺序仍是 A/C/G/U，但写给 tomtom 时把第4列当 T。
    """
    assert pwm.ndim == 2 and pwm.shape[1] == 4
    L = pwm.shape[0]

    eps = 1e-6
    pwm = np.clip(pwm.astype(float), eps, 1.0)
    pwm = pwm / pwm.sum(axis=1, keepdims=True)

    s = []
    s.append("MEME version 4\n\n")
    s.append("ALPHABET= ACGU\n\n")
    s.append("strands: + -\n\n")
    s.append("Background letter frequencies:\n")
    s.append("A 0.25 C 0.25 G 0.25 U 0.25\n\n")
    s.append(f"MOTIF {motif_name}\n")
    s.append(f"letter-probability matrix: alength= 4 w= {L} nsites= 20 E= 0\n")
    for i in range(L):
        s.append(" ".join(f"{x:.6f}" for x in pwm[i]) + "\n")
    s.append("\n")
    return "".join(s)

def _find_tomtom_table(outdir: Path) -> Path:
    # 不同版本可能叫 tomtom.tsv 或 tomtom.txt
    for fn in ["tomtom.tsv", "tomtom.txt", "tomtom.tab"]:
        p = outdir / fn
        if p.exists():
            return p
    raise FileNotFoundError(f"No tomtom result table found under {outdir}")

def run_tomtom_one(query_pwm: np.ndarray,
                   target_pwm: np.ndarray,
                   tomtom_bin: str = "tomtom",
                   dist: str = "pearson") -> Dict[str, Any]:
    """
    返回：
      - found=True/False
      - 若 found=True：p/e/q/overlap/offset/orientation 等
      - 若 found=False：reason + stderr（便于你定位）
    """
    if shutil.which(tomtom_bin) is None:
        return {"found": False, "reason": f"'{tomtom_bin}' not found in PATH"}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        q_path = td / "query.meme"
        t_path = td / "target.meme"
        outdir = td / "tomtom_out"

        q_path.write_text(pwm_to_meme_str_dna("QUERY", query_pwm), encoding="utf-8")
        t_path.write_text(pwm_to_meme_str_dna("TARGET", target_pwm), encoding="utf-8")

        cmd = [
            tomtom_bin,
            "-oc", str(outdir),
            "-dist", dist,
            "-thresh", "1.0",   # 输出所有匹配（即使不显著）
            str(q_path),
            str(t_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # tomtom 有时 returncode=0 但 stderr 里有 warning
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()

        if proc.returncode != 0:
            return {"found": False, "reason": "tomtom failed", "stderr": stderr, "stdout": stdout}

        # 解析表格
        try:
            table_path = _find_tomtom_table(outdir)
        except Exception as e:
            return {"found": False, "reason": f"no output table: {e}", "stderr": stderr, "stdout": stdout}

        # tomtom.tsv 是 tab 分隔，前面可能有 # 注释行
        # 用 comment="#" 跳过注释
        try:
            df = pd.read_csv(table_path, sep="\t", comment="#")
        except Exception:
            # 有些版本 tomtom.txt 可能是空格对齐
            df = pd.read_csv(table_path, sep=r"\s+", comment="#", engine="python")

        # 兼容不同列名
        # 常见列：Query ID / Target ID / p-value / E-value / q-value / overlap / offset / orientation
        if df.empty:
            return {"found": False, "reason": "tomtom table empty (no hits)", "stderr": stderr, "stdout": stdout}

        # 规范化列名
        cols = {c.lower().strip(): c for c in df.columns}
        def get_col(*cands):
            for c in cands:
                if c in cols:
                    return cols[c]
            return None

        col_q = get_col("q-value", "qvalue")
        col_p = get_col("p-value", "pvalue")
        col_e = get_col("e-value", "evalue")
        col_query = get_col("query id", "query_id", "query")
        col_target = get_col("target id", "target_id", "target")
        col_overlap = get_col("overlap")
        col_offset = get_col("offset", "optimal offset", "optimal_offset")
        col_orient = get_col("orientation")

        # 有些版本 q-value 可能全是 NaN（特别是目标很少时）
        # 策略：优先按 q，再按 p
        if col_q is not None:
            df_sorted = df.sort_values(by=[col_q, col_p] if col_p else [col_q], ascending=True, na_position="last")
        elif col_p is not None:
            df_sorted = df.sort_values(by=[col_p], ascending=True)
        else:
            # 实在没有 p/q 就取第一行
            df_sorted = df

        best = df_sorted.iloc[0]

        def safe_float(x):
            try:
                return float(x)
            except Exception:
                return float("nan")

        def safe_int(x):
            try:
                return int(float(x))
            except Exception:
                return None

        return {
            "found": True,
            "query_id": str(best[col_query]) if col_query else "QUERY",
            "target_id": str(best[col_target]) if col_target else "TARGET",
            "p_value": safe_float(best[col_p]) if col_p else float("nan"),
            "e_value": safe_float(best[col_e]) if col_e else float("nan"),
            "q_value": safe_float(best[col_q]) if col_q else float("nan"),
            "overlap": safe_int(best[col_overlap]) if col_overlap else None,
            "offset": safe_int(best[col_offset]) if col_offset else None,
            "orientation": str(best[col_orient]) if col_orient else "",
            "stderr": stderr,
        }

# -----------------------------
# fallback 相似度（你原来的 -FrobeniusDist 信息矩阵）
# -----------------------------
def pwm_to_information_matrix_kmer(pwm: np.ndarray, background: np.ndarray) -> np.ndarray:
    pwm = np.asarray(pwm, dtype=float)
    bg = np.asarray(background, dtype=float)
    P = np.clip(pwm, 1e-9, 1.0)
    B = np.clip(bg, 1e-9, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        info = P * (np.log2(P / B))
    info[np.abs(info) < 1e-9] = 0.0
    return info

def pwm_similarity_to_ref(cluster_pwm: np.ndarray, ref_pwm: np.ndarray, background: Optional[np.ndarray] = None) -> float:
    if background is None:
        background = np.ones(ref_pwm.shape[1], dtype=float)
        background /= background.sum()
    info_clu = pwm_to_information_matrix_kmer(cluster_pwm, background)
    info_ref = pwm_to_information_matrix_kmer(ref_pwm, background)
    dist = np.linalg.norm(info_clu - info_ref)
    return float(-dist)

# -----------------------------
# logo 输出（简单版：直接输出概率 logo；你也可以替换成 logomaker 信息量版）
# -----------------------------
def plot_seq_logo_from_pwm(seq_pwm: np.ndarray, out_png: str, title: str = "") -> None:
    import matplotlib.pyplot as plt
    try:
        import logomaker as lm
        has_lm = True
    except Exception:
        has_lm = False

    out_png = str(out_png)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    df = pd.DataFrame(seq_pwm, columns=BASES)

    plt.figure(figsize=(max(4, seq_pwm.shape[0] * 0.35), 2.4), dpi=300)
    ax = plt.gca()
    if has_lm:
        lm.Logo(df, ax=ax, color_scheme="classic")
        ax.set_ylabel("Probability")
    else:
        ax.imshow(df.values.T, aspect="auto")
        ax.set_yticks(range(4))
        ax.set_yticklabels(BASES)
        ax.set_ylabel("Base")
    ax.set_xlabel("Position")
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# -----------------------------
# 核心：对单个 dataset 做 motif discovery + TOMTOM 筛最显著 cluster
# -----------------------------
def run_motif_discovery_with_ref_tomtom(
    motifs,
    h_idx,
    ref_pwm: np.ndarray,
    out_dir: str,
    top_k_by_q: int = 1,
    tomtom_bin: str = "tomtom",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    返回：
      - df_clusters：每个 cluster 的统计（含 tomtom q-value 或 fallback similarity）
      - best：q-value 最小（最显著）的 cluster 记录
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    k = ref_pwm.shape[0]

    rows: List[Dict[str, Any]] = []
    # for cid, cluster in enumerate(clusters):
    #     if len(cluster) < min_cluster_size:
    #         continue
    #     seq_pwm = build_seq_pwm_for_cluster(cluster, k=k)
    for motif in motifs:
        rec: Dict[str, Any] = {
            "cluster_id": motif['cluster_id'],
            "size": motif['size'],
        }

        hit = run_tomtom_one(motif['seq_pwm'], ref_pwm, tomtom_bin=tomtom_bin)

        if hit["found"]:
            rec["method"] = "tomtom"
            rec["p_value"] = hit["p_value"]
            rec["q_value"] = hit["q_value"]
            rec["e_value"] = hit["e_value"]
            rec["overlap"] = hit["overlap"]
            rec["offset"] = hit["offset"]
            rec["orientation"] = hit["orientation"]
            rec["score_for_ranking"] = - (hit["q_value"] if np.isfinite(hit["q_value"]) else hit["p_value"])
        else:
            # 这里你可以选择：要么 fallback_similarity，要么直接标记 no_hit
            rec["method"] = "tomtom_no_hit"
            rec["tomtom_reason"] = hit.get("reason", "")
            rec["tomtom_stderr"] = hit.get("stderr", "")
            sim = pwm_similarity_to_ref(motif['seq_pwm'], ref_pwm, background=np.array([0.25,0.25,0.25,0.25]))
            rec["similarity"] = sim
            rec["score_for_ranking"] = sim

        rec["_seq_pwm"] = motif['seq_pwm']  # 临时挂着，后面画图用
        rows.append(rec)

    if not rows:
        raise ValueError("No clusters passed min_cluster_size.")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_seq_pwm"} for r in rows])
    # 选择 best：优先 tomtom 的 q-value；若全是 fallback，则按 similarity 最大
    if (df["method"] == "tomtom").any():
        df_t = df[df["method"] == "tomtom"].copy()
        best_row = df_t.sort_values("p_value", ascending=True).iloc[0].to_dict()
    else:
        best_row = df.sort_values("similarity", ascending=False).iloc[0].to_dict()

    # 同时输出 top_k_by_q（或 similarity） logo
    def get_pwm_by_cluster_id(cluster_id: int) -> np.ndarray:
        for r in rows:
            if r["cluster_id"] == cluster_id:
                return r["_seq_pwm"]
        raise KeyError(cluster_id)

    if (df["method"] == "tomtom").any():
        top = df[df["method"] == "tomtom"].sort_values("q_value", ascending=True).head(top_k_by_q)
        for rank, rr in enumerate(top.to_dict("records")):
            pwm = get_pwm_by_cluster_id(rr["cluster_id"])
            title = f"cluster={rr['cluster_id']} n={rr['size']} q={rr['q_value']:.2e}"
            # plot_seq_logo_from_pwm(pwm, os.path.join(out_dir, f"top{rank}_cluster{rr['cluster_id']}_q.png"), title=title)
            # plot_logo_ic(pwm, os.path.join(out_dir, f"top{rank}_cluster{rr['cluster_id']}_q.png"), title=title)
    else:
        top = df.sort_values("similarity", ascending=False).head(top_k_by_q)
        for rank, rr in enumerate(top.to_dict("records")):
            pwm = get_pwm_by_cluster_id(rr["cluster_id"])
            title = f"cluster={rr['cluster_id']} n={rr['size']} sim={rr['similarity']:.4f}"
            plot_seq_logo_from_pwm(pwm, os.path.join(out_dir, f"top{rank}_cluster{rr['cluster_id']}_sim.png"), title=title)

    # 输出 best 的 logo
    best_cid = int(best_row["cluster_id"])
    best_pwm = get_pwm_by_cluster_id(best_cid)
    if best_row.get("method") == "tomtom":
        if best_row['p_value']<0.05:
            best_title = f"kernel={h_idx} n={best_row['size']} P-value={best_row['p_value']:.2e} "
            best_path = os.path.join(out_dir, f"BEST_p={best_row['p_value']}_kernel={h_idx}.pdf")
            plot_logo_ic(best_pwm, best_path, title=best_title)

    else:
        best_title = f"BEST cluster={best_cid} n={best_row['size']} sim={best_row['similarity']:.4f}"
        best_path = os.path.join(out_dir, f"BEST_cluster{best_cid}_sim.png")
    # plot_seq_logo_from_pwm(best_pwm, best_path, title=best_title)

    return df, best_row

def pwm_to_ic_matrix(pwm: np.ndarray, background: np.ndarray = None, eps: float = 1e-9) -> np.ndarray:
    """
    pwm: (L,4) 概率矩阵（每行和为1）
    返回: (L,4) 的 IC logo 高度矩阵（bits）
    """
    pwm = np.asarray(pwm, dtype=float)
    pwm = np.clip(pwm, eps, 1.0)
    pwm = pwm / pwm.sum(axis=1, keepdims=True)

    # 对 4 字母，最大信息量是 2 bits
    H = -(pwm * np.log2(pwm)).sum(axis=1)          # (L,)
    R = 2.0 - H                                    # (L,)
    ic = pwm * R[:, None]                          # (L,4)
    return ic
BASES = ["A", "C", "G", "U"]  # 你的列顺序

def plot_logo_probability(pwm: np.ndarray, out_png: str, title: str = "", bases=BASES):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    import logomaker as lm

    pwm = np.asarray(pwm, dtype=float)
    pwm = pwm / np.clip(pwm.sum(axis=1, keepdims=True), 1e-12, None)

    df = pd.DataFrame(pwm, columns=bases)
    plt.figure(figsize=(max(4, pwm.shape[0]*0.35), 2.6), dpi=300)
    ax = plt.gca()
    lm.Logo(df, ax=ax)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Position")
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# def plot_logo_ic(pwm: np.ndarray, out_png: str, title: str = "", bases=BASES):
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#     import logomaker as lm

#     ic = pwm_to_ic_matrix(pwm)  # (L,4) bits
#     df = pd.DataFrame(ic, columns=bases)

#     plt.figure(figsize=(max(4, ic.shape[0]*0.35), 2.5), dpi=300)
#     ax = plt.gca()
#     lm.Logo(df, ax=ax)
#     ax.set_ylabel("Information content (bits)")
#     ax.set_xlabel("Position")
#     ax.set_ylim(0, 2)
#     if title:
#         ax.set_title(title, fontsize=10)
#     plt.tight_layout()
#     # plt.savefig(out_png, bbox_inches="tight")
#     plt.savefig(out_png, dpi=300, format='pdf', bbox_inches="tight")
#     plt.close()

def plot_logo_ic(pwm: np.ndarray, out_png: str, title: str = "", bases=BASES):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    import logomaker as lm

    ic = pwm_to_ic_matrix(pwm)  # (L,4) bits
    df = pd.DataFrame(ic, columns=bases)

    plt.figure(figsize=(max(4, ic.shape[0] * 0.35), 2.5), dpi=300)
    ax = plt.gca()
    lm.Logo(df, ax=ax)

    # 固定 y 范围（不显示任何刻度/文字）
    ax.set_ylim(0, 2)

    # 取消所有轴相关文字与刻度
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # 去掉边框线
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=300, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close()
# =========================
from build import *
# from step1_select_kernels import loadData, TorchDataset_multi, KNET_plus_ic
def loadData(path, Modeltype=None):
    """
    load data

    :param path:
    :return:
    """

    f_test = h5py.File(path + "/test.hdf5", "r")
    TestX = f_test["sequences"][()]
    TestX2 = f_test["seq_struct"][()]
    TestY = f_test["labs"][()].squeeze()
    TestY = TestY.squeeze()
    f_test.close()

    TestX2 = np.swapaxes(TestX2, 1, 2)
    TestX3 = np.concatenate([TestX, TestX2], axis=2)
    TestX = [TestX, TestX3]

    return [TestX, TestY]

class TorchDataset_multi(Dataset):
    def __init__(self, dataset):
        self.X1 = dataset[0][0].astype("float32")
        self.X2 = dataset[0][1].astype("float32")
        self.Y = dataset[1].astype("float32")

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, i):
        return self.X1[i], self.X2[i], self.Y[i]

# =========================
# 1) 可复现设置
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2) 模型 & 数据集加载（按你现有写法）
# =========================
def load_model_and_data(
    dataset_name: str,
    kernel_len: int,
    random_seed: int,
    model_ckpt_path: str,
    hdf5_dir: str,
    batch_size: int = 128,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    # ---- model ----
    model = KNET_plus_ic(5, kernel_len, 128, 1)  # 你脚本就是这么建的:contentReference[oaicite:5]{index=5}
    state_dict = torch.load(model_ckpt_path, map_location=torch.device("cpu"))
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state)
    model = model.to(device).eval()

    # ---- data ----
    data_set = loadData(hdf5_dir)  # 你脚本 loadData 读 test.hdf5 :contentReference[oaicite:6]{index=6}
    test_set = TorchDataset_multi(data_set)  # :contentReference[oaicite:7]{index=7}
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return model, test_loader


# =========================
# 3) step1_select_kernels：|w*grad| 选 topK kernels
#    输出：
#      - top_kernels: List[int]  (kernel idx in [0..kernel_len-1])
#      - best_head_per_kernel: Dict[kernel_idx -> head_idx]
# =========================
@torch.no_grad()
def _reshape_wattn_weight(model) -> torch.Tensor:
    """
    Wattn.weight: (out_channels, in_channels, 1)
    -> (k, c1, c2, h)
    其中 out_channels == k*h*c1
    """
    wattn_weight = model.Kattention.Wattn.weight.detach().squeeze(-1)  # :contentReference[oaicite:8]{index=8}
    kernel_len   = model.Kattention.kernel_size
    num_heads    = model.Kattention.num_heads
    channel_size = model.Kattention.channel_size  # 你脚本里也用它:contentReference[oaicite:9]{index=9}
    w4d = rearrange(
        wattn_weight,
        "(k h c1) c2 -> k c1 c2 h",
        k=kernel_len, h=num_heads, c1=channel_size
    )
    return w4d


def step1_select_kernels(
    model: nn.Module,
    dataloader: DataLoader,
    top_k: int = 10,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[List[int], Dict[int, int], torch.Tensor]:
    """
    计算 |w * grad|：
      - grad：在全数据上累积的 |dL/dw| 平均
      - saliency：|w| * |grad|
    然后：
      - kernel_importance: 对 (c1,c2,h) 聚合得到 (k,)
      - per_kernel_head_importance: 对 (c1,c2) 聚合得到 (k,h)
    """
    model.eval()

    wattn_weight = model.Kattention.Wattn.weight  # :contentReference[oaicite:10]{index=10}
    grad_abs_sum = torch.zeros_like(wattn_weight, device=device)

    criterion = nn.BCELoss()
    num_batches = 0

    for X1, X2, Y in tqdm(dataloader, desc="Accumulating |grad| for Wattn"):
        X1 = X1.to(device)
        X2 = X2.to(device)
        Y  = Y.to(device).float()

        model.zero_grad(set_to_none=True)
        pred = model(X1, X2)
        Y = Y.view_as(pred)
        loss = criterion(pred, Y)

        loss.backward()
        grad_abs_sum += wattn_weight.grad.detach().abs()
        num_batches += 1

    grad_abs_mean = grad_abs_sum / max(1, num_batches)

    # ---- |w * grad| ----
    weight = wattn_weight.detach().squeeze(-1)      # (out_ch, in_ch) :contentReference[oaicite:11]{index=11}
    grad   = grad_abs_mean.detach().squeeze(-1)
    saliency = weight.abs() * grad                  # :contentReference[oaicite:12]{index=12}

    # ---- reshape to (k, c1, c2, h) ----
    kernel_len   = model.Kattention.kernel_size
    channel_size = model.Kattention.channel_size
    num_heads    = model.Kattention.num_heads

    sal_4d = rearrange(
        saliency,
        "(k h c1) c2 -> k c1 c2 h",
        k=kernel_len, h=num_heads, c1=channel_size
    )  # :contentReference[oaicite:13]{index=13}

    # kernel importance: (k,)
    kernel_importance = sal_4d.mean(dim=(1, 2, 3))  # :contentReference[oaicite:14]{index=14}

    sorted_scores, sorted_indices = torch.sort(kernel_importance, descending=True)

    print("=== Kernel importance ranking (high -> low) ===")
    for rank, (idx, score) in enumerate(zip(sorted_indices.tolist(), sorted_scores.tolist()), start=1):
        print(f"Rank {rank:2d} | kernel {idx:2d} | importance = {score:.6f}")
    return sorted_indices[:top_k]


# =========================
# 4) step2_get_regions：对每个 (kernel, head) 提取 HAR regions
#    关键：hook 抓 attn_logits，然后对高置信样本找最大注意力位置 -> 切片得到片段
# =========================
def step2_get_regions_for_heads(
    model,
    dataloader: DataLoader,
    h_idx: int,
    threshold: float = 0.8,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[np.ndarray]:
    """
    保持原脚本逻辑：
    - forward hook 抓 attn_logits (B,H,Q,K)
    - 对 pred > threshold 的样本：
        attn_mat = attn_logits[b, h_idx]
        找最大值位置 -> (i, j)
        从输入 X2 切出 seq1 与 seq2 (kernel_len, 5)
    - 返回 regions = [seq1, seq2, seq1, seq2, ...]
    """

    model.eval()
    kernel_len = model.Kattention.kernel_size  # 原脚本也是用这个长度切片

    # 用 list 保存每个 batch 的 attn_logits
    attn_logits_buffer = []

    def kattn_hook(module, inputs, output):
        """
        与原脚本一致：Kattention forward 输出 dict，含 'attn_logits'
        """
        # output["attn_logits"]: (B,H,Q,K)
        attn_logits_buffer.append(output["attn_logits"].detach().cpu())

    hook_handle = model.Kattention.register_forward_hook(kattn_hook)

    regions: List[np.ndarray] = []

    with torch.no_grad():
        for X1, X2, Y in tqdm(dataloader, desc="Extract regions from attn_logits"):
            attn_logits_buffer.clear()
            X1 = X1.to(device)
            X2 = X2.to(device)

            # 预测分数，用于筛选 pred > threshold 的样本
            pred = model(X1, X2).detach().view(-1).cpu()  # (B,)
            # hook 应该捕获到本 batch 的 attn_logits
            if len(attn_logits_buffer) != 1:
                continue
            attn_logits = attn_logits_buffer[0]  # (B,H,Q,K)

            B, H, Q, K = attn_logits.shape

            # 把 X2 放到 CPU，便于切片
            X2_cpu = X2.detach().cpu()                 # (B,L,5)
            X2_cpu_rev = torch.flip(X2_cpu, dims=[1])  # (B,L,5) 反向序列（用于 seq2 逻辑）

            # 只处理 pred > threshold 的样本
            pos_idx = (pred > threshold).nonzero(as_tuple=False).squeeze(-1)
            if pos_idx.numel() == 0:
                continue

            L_seq = X2_cpu.shape[1]

            for b in pos_idx.tolist():
                if h_idx < 0 or h_idx >= H:
                    continue

                # 原逻辑：attn_mat = attn_logits[b, h_idx]
                attn_mat = attn_logits[b, h_idx]  # (Q,K)

                # 原逻辑：flip key 维，再取最大值的位置
                attn_flip = torch.flip(attn_mat, dims=[-1])  # (Q,K)

                flat_idx = int(attn_flip.reshape(-1).argmax().item())
                i = flat_idx // K   # query index
                j = flat_idx % K    # key index (on flipped)

                # 从输入中切片得到 seq1 / seq2（每个都是 kernel_len x 5）
                # 保持原脚本的边界检查
                if (i + kernel_len) <= L_seq and (j + kernel_len) <= L_seq:
                    # seq1：正向 X2 的 [i:i+kernel_len]
                    seq1 = X2_cpu[b, i:i + kernel_len, :]  # (kernel_len,5)

                    # seq2：在反向 X2_cpu_rev 上切 [j:j+kernel_len] 再翻回去
                    seq2 = torch.flip(
                        X2_cpu_rev[b, j:j + kernel_len, :],
                        dims=[0]  # 反向片段翻回正向（对应原脚本的“再flip一次”）
                    )

                    regions.append(seq1.numpy())
                    regions.append(seq2.numpy())

    hook_handle.remove()
    return regions


# =========================
# 5) step3_motif：regions -> k-mer -> cluster -> seq_pwm
# =========================
def step3_regions_to_seq_pwm(
    regions: List[np.ndarray],
    kmer_len: int = 6,
    max_seq_mismatch: int = 1,
    min_cluster_size: int = 5,
) -> List[Dict]:
    """
    返回按簇大小排序的若干 motif：
      [{"cluster_id":..., "size":..., "seq_pwm": np.ndarray(k,4)}]
    """
    # 复用你脚本的 k-mer 抽取 + 聚簇 + PWM 统计:contentReference[oaicite:23]{index=23}
    kmers = extract_kmers_from_regions(regions, k=kmer_len)
    if len(kmers) == 0:
        return []

    clusters = cluster_kmers_seq(kmers, max_seq_mismatch=max_seq_mismatch)

    motifs = []
    for cid, cl in enumerate(clusters):
        if len(cl) < min_cluster_size:
            continue
        seq_pwm = build_seq_pwm_for_cluster(cl, k=kmer_len)
        motifs.append({"cluster_id": cid, "size": len(cl), "seq_pwm": seq_pwm})

    motifs.sort(key=lambda x: x["size"], reverse=True)
    return motifs


# =========================
# 6) 一键运行：top10 kernel -> HAR -> seq_pwm
# =========================
def run_kernel_to_pwm_pipeline(
    ref_pwm,
    dataset_name: str,
    model_ckpt_path: str,
    hdf5_dir: str,
    out_dir: str,
    kernel_len: int = 12,
    random_seed: int = 666,
    top_kernels: int = 10,
    score_threshold: float = 0.8,
    kmer_len: int = 6,
    tomtom_bin: str = "tomtom"
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, test_loader = load_model_and_data(
        dataset_name=dataset_name,
        kernel_len=kernel_len,
        random_seed=random_seed,
        model_ckpt_path=model_ckpt_path,
        hdf5_dir=hdf5_dir,
        batch_size=128,
        device=device,
    )

    # ---- step1: select kernels by |w*grad| ----
    k_imp = step1_select_kernels(
        model=model, dataloader=test_loader, top_k=top_kernels, device=device
    )
    # 保存 kernel 重要性
    # np.save(out_dir / "kernel_importance.npy", k_imp.numpy())

    # ---- step2+3: for each kernel -> regions -> seq_pwm ----
    all_results = []
    for rank, h_idx in enumerate(k_imp, start=1):
        h_idx = int(h_idx)
        regions = step2_get_regions_for_heads(
            model=model,
            dataloader=test_loader,
            h_idx=h_idx,
            # kernel_len=kernel_len,
            threshold=score_threshold,
            device=device,
        )
        
        print(f"Kernel {h_idx}: extracted regions = {len(regions)}")
        if len(regions) == 0:
            continue

        motifs = step3_regions_to_seq_pwm(
            regions=regions,
            kmer_len=kmer_len,
            max_seq_mismatch=1,
            min_cluster_size=int(len(regions)/10),
        )

        df_clusters, best = run_motif_discovery_with_ref_tomtom(
                motifs=motifs,
                h_idx=h_idx,
                ref_pwm=ref_pwm,
                out_dir=out_dir.as_posix(),
                top_k_by_q=3,
                tomtom_bin=tomtom_bin,
        )
        # pdb.set_trace()
        # # 保存每个 kernel 的 top motif（例如 top3）
        # k_save = out_dir / f"rank_{rank} kernel_{h_idx}"
        # k_save.mkdir(parents=True, exist_ok=True)

        # np.save(k_save / "regions.npy", np.array(regions, dtype=np.float32))

        # # 保存 PWM
        # for rank, m in enumerate(motifs[:3], start=1):
        #     np.save(k_save / f"motif_rank{rank}_seq_pwm.npy", m["seq_pwm"])
        #     # 如果你想直接画 logo，可复用 plot_seq_logo_from_pwm / plot_pwm_logo（你脚本里已有）:contentReference[oaicite:24]{index=24}

        all_results.append({
            "kernel": int(h_idx),
            "n_regions": int(len(regions)),
            "n_motifs": int(len(motifs)),
            "top_motifs": motifs[:3],
        })

    return all_results
# -----------------------------
# 批量主入口
# -----------------------------
def batch_run(
    dataset_list: List[str],
    regions_meta: pd.DataFrame,
    pwm_txt_path: str,
    attract_meta_path: str,
    out_root: str = "./motif_batch_out",
    tomtom_bin: str = "tomtom",
    max_seq_mismatch: int = 1,
    min_cluster_size: int = 5,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    返回：
      success_list: 处理成功的 dataset
      skipped_list: (dataset, reason) 的列表
    """
    out_root = str(out_root)
    os.makedirs(out_root, exist_ok=True)

    motif_dict = load_pwms_from_file(pwm_txt_path)
    rbp2motif_ids = build_rbp2motif_ids(attract_meta_path)

    success: List[str] = []
    skipped: List[Tuple[str, str]] = []

    for dataset in dataset_list:
        print(f"=========================== processing {dataset} ===========================")
        rbp_name = dataset.split("_")[0]
        motif_ids = rbp2motif_ids.get(rbp_name, [])
        if not motif_ids:
            skipped.append((dataset, f"ATtRACT_db has no motif for RBP={rbp_name}"))
            continue

        # 取第一条 motif 作为 ref（你也可以改成遍历所有 motif_id）
        ref_id = motif_ids[0]
        if ref_id not in motif_dict:
            skipped.append((dataset, f"motif_id={ref_id} not found in pwm.txt"))
            continue
        ref_pwm = motif_dict[ref_id]
        ref_ic_png = os.path.join(f"./kernel2pwm_out/{dataset}/", "ref_pwm_IC.pdf")
        plot_logo_ic(ref_pwm, ref_ic_png, title="Reference motif (IC)")
        if len(motif_ids)>1:
            for ind,id in enumerate(motif_ids):
                ref_pwm_ = motif_dict[id]
                ref_ic_png = os.path.join(f"./kernel2pwm_out/{dataset}/", f"ref_pwm_IC_{ind}.pdf")
                plot_logo_ic(ref_pwm_, ref_ic_png, title="Reference motif (IC)")
        # 加载 regions
        try:
            # regions = load_regions_for_dataset(dataset, regions_dir=regions_dir)
            best_para = df[(df['dataset'] == dataset) &(df['model'] == 'KNET_plus_ic')].sort_values(by='AUC', ascending=False).iloc[0,:][['random_seed','kernel_len']]
            random_seed = best_para[0]
            kernel_len = int(best_para[1])
            model1 = KNET_plus_ic(5, kernel_len, 128, 1)
            path = f'/lustre/grp/gglab/liut/Kattention_aten_test/result/RBP/HDF5/{dataset}/KNET_plus_ic/'
            model_ckpt_path = path + f'model_KernelNum-128kernel_size-{kernel_len}_seed-{random_seed}_opt-adamw.checkpointer.pt'

            hdf5_dir = f'../../external/RBP/HDF5/{dataset}/'
            out_dir = f"./kernel2pwm_out/{dataset}/"

            results = run_kernel_to_pwm_pipeline(
                ref_pwm=ref_pwm,
                dataset_name=dataset,
                model_ckpt_path=model_ckpt_path,
                hdf5_dir=hdf5_dir,
                out_dir=out_dir,
                kernel_len=kernel_len,
                random_seed=random_seed,
                top_kernels=64,
                score_threshold=0.8,
                kmer_len=ref_pwm.shape[0],
                tomtom_bin=tomtom_bin,
            )

            # # 保存结果表
            # df_clusters.to_csv(out_dir / "cluster_tomtom_results.tsv", sep="\t", index=False)
            # # 记录 ref 信息
            # (out_dir / "ref_info.txt").write_text(
            #     f"dataset={dataset}\nrbp_name={rbp_name}\nref_motif_id={ref_id}\nref_len={ref_pwm.shape[0]}\n\nbest={best}\n",
            #     encoding="utf-8",
            # )
            success.append(dataset)
        except Exception as e:
            skipped.append((dataset, f"regions load failed: {e}"))
            continue

        # 跑 motif discovery + TOMTOM
        out_dir = Path(out_root) / dataset
        out_dir.mkdir(parents=True, exist_ok=True)



    # 汇总报告
    report = Path(out_root) / "batch_report.tsv"
    pd.DataFrame({
        "dataset": [*success, *(x[0] for x in skipped)],
        "status": ["success"] * len(success) + ["skipped"] * len(skipped),
        "reason": [""] * len(success) + [x[1] for x in skipped],
    }).to_csv(report, sep="\t", index=False)

    return success, skipped


if __name__ == "__main__":
    # 你改这里即可
    # dataset_list = [
    #     "RBFOX2_HepG2","UTP3_K562","CAPRIN1_HEK293","NUDT21_HEK293","WDR3_K562","SF3B1_K562","C17ORF85_HEK293","WDR43_K562","CSTF2T_HepG2","CDC40_HepG2","NKRF_HepG2","EIF3H_HepG2","SF3B4_HepG2","EWSR1_HEK293","FIP1L1_HEK293","QKI_HEK293","DDX52_K562","U2AF2_Hela","DDX21_K562","CPSF1_HEK293","HNRNPK_K562","TRA2A_K562","IGF2BP1_K562","SRSF1_HepG2"
    # ]
    # dataset_list = list(set(df['dataset'].to_list()))

    # dataset_list = [
    #     "CPEB4_K562","FMR1_K562","FUS_HEK293","HNRNPA1_K562","HNRNPC_Hela","HNRNPD_HEK293","HNRNPU_Hela","IGF2BP3_HEK293",
    #     "KHDRBS1_K562","KHSRP_K562","LIN28A_H9","NONO_K562","NUDT21_HEK293","PCBP1_K562","PCBP2_HepG2","PTBP1_Hela","PUM1_K562",
    #     "PUM2_HEK293","QKI_HEK293","QKI_HEK293","SRSF1_HepG2","SRSF7_K562","SRSF9_HepG2","TARDBP_K562","TIA1_Hela","TIAL1_Hela",
    #     "TRA2A_K562","U2AF2_Hela","ZRANB2_K562"
    # ]
    dataset_list = [
        "RBFOX2_HepG2"
    ]
    df = pd.read_csv('./log/Train_KNET_plus_ic_test.tsv', sep='\t', header=None, names=['cls','dataset','model','kernel_len','random_seed','AUC','loss'])
    # pdb.set_trace()
    df = df[['dataset','model','AUC','kernel_len','random_seed']].reset_index(drop=True)
    success, skipped = batch_run(
        dataset_list=dataset_list,
        regions_meta=df,     # 约定：里面放 {dataset}.npz，key='regions'
        pwm_txt_path="./pwm.txt",
        attract_meta_path="./ATtRACT_db.txt",
        out_root="./picture/motif_batch_out",
        tomtom_bin="tomtom",
        max_seq_mismatch=1,
        min_cluster_size=5,
    )

    print("\n===== SUCCESS =====")
    for x in success:
        print(x)

    print("\n===== SKIPPED =====")
    for d, r in skipped:
        print(f"{d}\t{r}")
