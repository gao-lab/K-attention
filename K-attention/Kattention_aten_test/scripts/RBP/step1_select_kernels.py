import pandas as pd
import pdb
# sys.path.append("../../corecode/")
from build import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.colors as colors
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import h5py
from typing import Dict, Tuple, Callable, Union
from einops import rearrange
from collections import defaultdict

try:
    import logomaker as lm
    HAS_LOGOMAKER = True
except Exception:
    HAS_LOGOMAKER = False

def pwm_to_information_matrix_kmer(pwm: np.ndarray,
                                   background: np.ndarray):
    """
    pwm: (L, K) 概率矩阵（每行一个 position，行内和≈1）
    background: (K,) 背景分布，例如 [0.25,0.25,0.25,0.25]

    返回: 信息矩阵 info(L, K)，元素 = p * log2(p/bg)，p=0 时置 0
    """
    pwm = np.asarray(pwm, dtype=float)
    bg = np.asarray(background, dtype=float)

    P = np.clip(pwm, 1e-9, 1.0)
    B = np.clip(bg, 1e-9, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        info = P * (np.log2(P / B))

    info[np.abs(info) < 1e-9] = 0.0
    return info


def decode_seq_and_struct(region: np.ndarray,
                          struct_threshold: float = 0.233):
    """
    region: shape (L,5), 前4列是 one-hot，最后一列是 icSHAPE 或结构分数
    struct_threshold: >= 阈值记为 U, < 阈值记为 P
    返回: (seq_str, struct_str)，长度均为 L
    """
    assert region.shape[1] == 5
    L = region.shape[0]
    onehot = region[:, :4]
    struct_vals = region[:, 4]

    # one-hot -> 字母序列
    base_idx = onehot.argmax(axis=1)
    seq = "".join(BASES[i] for i in base_idx)

    # icSHAPE -> U/P
    struct = "".join("U" if v >= struct_threshold else "P"
                     for v in struct_vals)

    return seq, struct

def extract_kmers_from_regions(regions,
                               k: int = 6,
                               struct_threshold: float = 0.233):
    """
    regions: list[np.ndarray]，每个 array shape (L,5)
    返回: list[dict]，每个 dict 表示一个 integrative k-mer：
        {
            "seq": "ACGU...",
            "struct": "UPUP..",
            "region_id": int,
            "start": int   # 在该 region 中的起始位置
        }
    """
    kmers = []
    for ridx, region in enumerate(regions):
        seq, struct = decode_seq_and_struct(region, struct_threshold)
        L = len(seq)
        if L < k:
            continue
        for start in range(0, L - k + 1):
            kmer_seq = seq[start:start + k]
            kmer_struct = struct[start:start + k]
            kmers.append({
                "seq": kmer_seq,
                "struct": kmer_struct,
                "region_id": ridx,
                "start": start,
            })
    return kmers


def kmer_distance(k1, k2):
    """计算两个 integrative k-mer 的序列和结构 Hamming 距离"""
    s1, s2 = k1["seq"], k2["seq"]
    t1, t2 = k1["struct"], k2["struct"]
    assert len(s1) == len(s2) == len(t1) == len(t2)
    seq_dist = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    struct_dist = sum(c1 != c2 for c1, c2 in zip(t1, t2))
    return seq_dist, struct_dist

def hamming_distance(s1: str, s2: str) -> int:
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def cluster_kmers_seq(kmers,
                      max_seq_mismatch: int = 1):
    """
    仅按序列做一个简单的贪心聚簇：
    - 相同簇中任一 k-mer 与簇代表（首个） Hamming 距离 <= max_seq_mismatch

    kmers: list[dict]，每个 dict 至少含 "seq"
    返回: list[list[dict]]，每个子 list 是一个簇
    """
    clusters = []
    for kmer in kmers:
        assigned = False
        for cluster in clusters:
            rep = cluster[0]["seq"]  # 簇代表
            if hamming_distance(kmer["seq"], rep) <= max_seq_mismatch:
                cluster.append(kmer)
                assigned = True
                break
        if not assigned:
            clusters.append([kmer])
    return clusters

def cluster_kmers(kmers,
                  max_seq_mismatch: int = 1,
                  max_struct_mismatch: int = 2):
    """
    非常简单的单链聚类：
    - 遍历所有 k-mer
    - 对每个新 k-mer，尝试丢进一个已有簇（和簇中第一个成员比较）
    - 如果所有簇都不满足距离阈值，就新开一个簇

    返回: list[list[dict]]，每个子 list 是一个簇的 k-mers
    """
    clusters = []
    for kmer in kmers:
        assigned = False
        for cluster in clusters:
            rep = cluster[0]  # 代表元（簇里的第一个 k-mer）
            ds, dt = kmer_distance(kmer, rep)
            if ds <= max_seq_mismatch and dt <= max_struct_mismatch:
                cluster.append(kmer)
                assigned = True
                break
        if not assigned:
            clusters.append([kmer])
    return clusters

def build_pwms_for_cluster(cluster, k: int):
    """
    cluster: 一个簇（list[dict]）
    返回:
        seq_pwm: shape (k, 4)，列是 A,C,G,U
        struct_pwm: shape (k, 2)，列是 U,P
    """
    seq_counts = np.zeros((k, 4), dtype=float)
    struct_counts = np.zeros((k, 2), dtype=float)

    for kmer in cluster:
        seq = kmer["seq"]
        struct = kmer["struct"]
        assert len(seq) == len(struct) == k

        # 序列计数
        for pos, base in enumerate(seq):
            b_idx = BASES.index(base)  # A/C/G/U -> 0/1/2/3
            seq_counts[pos, b_idx] += 1

        # 结构计数
        for pos, st in enumerate(struct):
            s_idx = 0 if st == "U" else 1  # U->0, P->1
            struct_counts[pos, s_idx] += 1

    # 归一化为概率（每一行一个 position）
    seq_pwm = seq_counts / np.clip(seq_counts.sum(axis=1, keepdims=True),
                                   1e-9, None)
    struct_pwm = struct_counts / np.clip(struct_counts.sum(axis=1, keepdims=True),
                                         1e-9, None)

    return seq_pwm, struct_pwm

def build_seq_pwm_for_cluster(cluster, k: int) -> np.ndarray:
    """
    从一个簇的所有 k-mer（仅序列）统计出 seq PWM。

    cluster: list[dict]，每个 dict["seq"] 为字符串，长度为 k
    返回:
        seq_pwm: shape (k,4)，行是 position，列是 A/C/G/U 的概率
    """
    seq_counts = np.zeros((k, 4), dtype=float)

    for item in cluster:
        seq = item["seq"]
        assert len(seq) == k
        for pos, base in enumerate(seq):
            b_idx = BASES.index(base)  # A/C/G/U -> 0/1/2/3
            seq_counts[pos, b_idx] += 1

    # 归一化为概率（避免除 0）
    seq_pwm = seq_counts / np.clip(seq_counts.sum(axis=1, keepdims=True),
                                   1e-9, None)
    return seq_pwm

def pwm_to_information_matrix(pwm: np.ndarray,
                              background: np.ndarray):
    """
    pwm: (L, K) 概率矩阵（每行一个 position）
    background: (K,) 背景分布，例如 [0.25,0.25,0.25,0.25]

    返回: 信息矩阵 info(L, K)，元素 = p * log2(p/bg)，p=0 时置 0
    """
    pwm = np.asarray(pwm, dtype=float)
    bg = np.asarray(background, dtype=float)

    P = np.clip(pwm, 1e-9, 1.0)
    B = np.clip(bg, 1e-9, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        info = P * (np.log2(P / B))

    info[np.abs(info) < 1e-9] = 0.0
    return info


def pwm_similarity_to_ref(cluster_pwm: np.ndarray,
                          ref_pwm: np.ndarray,
                          background: np.ndarray = None) -> float:
    """
    计算簇的 PWM 和参考 PWM 的相似度（越大越相似）。

    思路：
    - 先把两者都变成信息矩阵 (p*log2(p/bg))
    - 然后按 Frobenius 距离计算差异 D = ||I_cluster - I_ref||
    - 相似度 = -D（距离越小，相似度越大）

    返回: similarity (float)
    """
    if background is None:
        # 默认均匀背景
        background = np.ones(ref_pwm.shape[1], dtype=float)
        background /= background.sum()

    info_clu = pwm_to_information_matrix_kmer(cluster_pwm, background)
    info_ref = pwm_to_information_matrix_kmer(ref_pwm, background)

    # 确保形状一致
    assert info_clu.shape == info_ref.shape

    dist = np.linalg.norm(info_clu - info_ref)  # Frobenius norm
    similarity = -dist
    return float(similarity)


def plot_seq_logo_from_pwm(seq_pwm: np.ndarray,
                           out_png: str,
                           title: str = ""):
    """
    给定 seq PWM（概率形式），画 sequence logo 并保存。
    """
    L = seq_pwm.shape[0]
    pwm_df = pd.DataFrame(seq_pwm, columns=BASES)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    plt.figure(figsize=(max(4, L * 0.4), 2.5), dpi=300)
    logo = lm.Logo(pwm_df, color_scheme="classic")

    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.style_xticks(rotation=0, fmt='%d')
    logo.ax.set_ylabel("Probability")
    logo.ax.set_xlabel("Position")
    if title:
        logo.ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# ========== 主流程：提取 k-mer、聚簇、对比参考 PWM、保存 top-3 logo ==========

def run_motif_discovery_with_ref(
    regions,
    ref_pwm: np.ndarray,
    out_dir: str = "./motif_with_ref",
    max_seq_mismatch: int = 1,
    min_cluster_size: int = 5,
    top_k: int = 3,
):
    """
    regions: list[np.ndarray]，每个 (12,5)，前4列 one-hot A/C/G/U，第5列 icSHAPE
    ref_pwm: 参考 PWM，形状 (k,4)，例如你给的 pwm_array
    out_dir: 保存 seq logo 的目录
    """
    os.makedirs(out_dir, exist_ok=True)

    k = ref_pwm.shape[0]  # 从参考 pwm 推出 k-mer 长度
    bg_seq = np.array([0.25, 0.25, 0.25, 0.25])  # 均匀背景

    # 1) 抽取所有 k-mer
    kmers = extract_kmers_from_regions(regions, k=k)
    print("总共抽取 k-mers 数:", len(kmers))

    if not kmers:
        print("没有抽到任何 k-mer，检查 regions 长度或 k。")
        return

    # 2) 按序列聚簇
    clusters = cluster_kmers_seq(kmers, max_seq_mismatch=max_seq_mismatch)
    print("聚簇个数:", len(clusters))

    # 3) 对每个簇生成 PWM，并与 ref_pwm 对比
    cluster_infos = []
    for idx, cluster in enumerate(clusters):
        if len(cluster) < min_cluster_size:
            continue  # 簇太小，忽略

        seq_pwm = build_seq_pwm_for_cluster(cluster, k=k)
        sim = pwm_similarity_to_ref(seq_pwm, ref_pwm, background=bg_seq)

        cluster_infos.append({
            "cluster_id": idx,
            "size": len(cluster),
            "seq_pwm": seq_pwm,
            "similarity": sim,
        })

    if not cluster_infos:
        print("没有满足 min_cluster_size 的簇，或输入数据过少。")
        return

    # 4) 按相似度从大到小排序，取前 top_k 个
    cluster_infos.sort(key=lambda x: x["similarity"], reverse=True)
    top_clusters = cluster_infos[:top_k]

    for rank, info in enumerate(top_clusters):
        cid = info["cluster_id"]
        size = info["size"]
        sim = info["similarity"]
        print(f"Top {rank}: cluster {cid}, size={size}, similarity={sim:.4f}")

        out_png = os.path.join(out_dir, f"cluster_{cid}_rank_{rank}.png")
        title = f"Cluster {cid} (rank {rank}, n={size}, sim={sim:.3f})"
        plot_seq_logo_from_pwm(info["seq_pwm"], out_png, title=title)

def run_motif_discovery(regions,
                        k: int = 6,
                        struct_threshold: float = 0.0,
                        top_k_clusters: int = 3,
                        out_dir: str = "./motif_logos"):
    """
    总流程示例：
    - 从潜在 motif 区域抽 integrative k-mer
    - 聚簇
    - 对最大的若干个簇生成 PWM 并画 integrative logo
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    # 1) 抽 k-mer
    kmers = extract_kmers_from_regions(regions, k=k,
                                       struct_threshold=struct_threshold)
    print("k-mers 数:", len(kmers))

    # 2) 聚簇
    clusters = cluster_kmers(kmers,
                             max_seq_mismatch=1,
                             max_struct_mismatch=2)
    print("簇个数:", len(clusters))

    # 按簇大小排序，取前 top_k_clusters 个
    clusters_sorted = sorted(clusters, key=len, reverse=True)
    for idx, cluster in enumerate(clusters_sorted[:top_k_clusters]):
        print(f"簇 {idx}: 大小 = {len(cluster)}")

        # 3) 计算 PWM
        seq_pwm, struct_pwm = build_pwms_for_cluster(cluster, k=k)

        # 4) 画 integrative logo
        out_png = os.path.join(out_dir, f"motif_cluster_{idx}.png")
        plot_integrative_logo_kmer(seq_pwm, struct_pwm, out_png,
                              title=f"Cluster {idx} (n={len(cluster)})")

def plot_integrative_logo_kmer(seq_pwm: np.ndarray,
                               struct_pwm: np.ndarray,
                               out_png: str,
                               title: str = ""):
    """
    seq_pwm: (k,4) 概率 PWM
    struct_pwm: (k,2) 概率 PWM
    out_png: 保存路径
    """

    k = seq_pwm.shape[0]

    # === 1) 概率 PWM -> 信息矩阵 ===
    bg_seq = np.array([0.25, 0.25, 0.25, 0.25])  # A/C/G/U 背景
    bg_struct = np.array([0.5, 0.5])             # U/P 背景

    seq_info = pwm_to_information_matrix_kmer(seq_pwm, bg_seq)        # (k,4)
    struct_info = pwm_to_information_matrix_kmer(struct_pwm, bg_struct)  # (k,2)

    # === 2) 序列 logo：只保留正值，画在 0 上方 ===
    seq_info_pos = np.clip(seq_info, 0.0, None)
    seq_df = pd.DataFrame(seq_info_pos, columns=BASES)

    # === 3) 结构 logo：只保留正值，整体向下画 ===
    struct_info_pos = np.clip(struct_info, 0.0, None)
    struct_df = pd.DataFrame(-struct_info_pos, columns=["U", "P"])

    # （可选）看看是不是 seq 部分本身就是 0
    # print("seq_info_pos max:", np.max(seq_info_pos))
    # print("struct_info_pos max:", np.max(struct_info_pos))

    # === 4) 画图 ===
    plt.figure(figsize=(max(4, k * 0.4), 3.0), dpi=300)
    ax = plt.gca()

    # 先画 sequence 信息量
    lm.Logo(seq_df, ax=ax, color_scheme="classic")

    # 再画 structure 信息量（负值）
    color_map = {"U": "gray", "P": "purple"}
    lm.Logo(struct_df, ax=ax, color_scheme=color_map)

    # === 5) 手动设置 y 轴范围，覆盖两者 ===
    # 每个 position 的总信息量（上/下）
    seq_total = seq_info_pos.sum(axis=1)        # (k,)
    struct_total = struct_info_pos.sum(axis=1)  # (k,)

    ymax = float(seq_total.max() if seq_total.size else 0.0)
    ymin = -float(struct_total.max() if struct_total.size else 0.0)

    # 给一点边距，避免贴边
    if ymax <= 0:
        ymax = 0.1
    if ymin >= 0:
        ymin = -0.1

    ax.set_ylim(ymin * 1.1, ymax * 1.1)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("position")
    ax.set_ylabel("Information (bits)")
    if title:
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def load_pwms_from_file(pwm_txt_path: str) -> dict:
    """
    解析 ATtRACT 样式 pwm.txt：
    >630    6
    0.0 ... 4列
    ...
    返回: { motif_id(str): np.ndarray(L,4) }
    """
    motifs: dict[str, np.ndarray] = {}
    cur_id = None
    cur_len = None
    rows = []

    with open(pwm_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # 先把上一个 motif 收尾
                if cur_id is not None and rows:
                    arr = np.array(rows, dtype=float)
                    motifs[cur_id] = arr
                # 解析头行：">630    6"
                parts = line[1:].split()
                cur_id = parts[0]           # "630"
                cur_len = int(parts[1])     # 6
                rows = []
            else:
                # 普通行：4 个浮点数 (A C G U)
                vals = [float(x) for x in line.split()]
                rows.append(vals)

    # 最后一个 motif
    if cur_id is not None and rows:
        arr = np.array(rows, dtype=float)
        motifs[cur_id] = arr

    return motifs

def build_rbp2motif_ids(meta_path: str) -> dict:
    """
    解析你给的前置文件（制表符分隔）：
    RBFOX2  ENSG00000100320  ...  RRM  630  1.000000**
    第一列: RBP 名
    倒数第二列: motif_id (e.g. 630, 622 ...)
    返回: { "RBFOX2": ["630","622",...], ... }
    """
    rbp2motifs = defaultdict(list)

    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            rbp = cols[0]
            motif_id = cols[-2]  # 倒数第二列

            if motif_id not in rbp2motifs[rbp]:
                rbp2motifs[rbp].append(motif_id)

    return dict(rbp2motifs)

def scan_pwm_max_score_one_seq(seq_tensor, pwm_array):
    """
    seq_tensor: (Lseq, 5) torch.Tensor，前4列 A/C/G/U one-hot，第5列 icSHAPE
    pwm_array: (Lmotif, 4) np.array，列: A,C,G,U

    返回: 该序列上滑窗打分的最大值（float）
    """
    seq = seq_tensor.detach().cpu().numpy()       # (Lseq, 5)
    onehot = seq[:, :4]                           # (Lseq, 4)
    Lseq = onehot.shape[0]
    Lmotif = pwm_array.shape[0]

    if Lseq < Lmotif:
        return float("-inf")  # 太短就返回一个很小的值

    # 把 one-hot -> base index (0:A,1:C,2:G,3:U)
    base_idx = np.argmax(onehot, axis=-1)         # (Lseq,)

    scores = []
    for start in range(Lseq - Lmotif + 1):
        window_idx = base_idx[start:start + Lmotif]      # (Lmotif,)
        # motif 每一行取对应碱基的概率（或权重），并累加
        score = float(pwm_array[np.arange(Lmotif), window_idx].sum())
        scores.append(score)

    return float(max(scores)) if scores else float("-inf")

def plot_pwm_logo(
    pwm_df: pd.DataFrame,
    out_png: Union[str, Path],
    title: str = "",
    background: Optional[Dict[str, float]] = None,
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
        plt.figure(figsize=(max(4, info_df.shape[0] * 0.22), 2.4), dpi=300)
        # 经典配色（也可自定义 color_scheme）
        logo = lm.Logo(info_df, color_scheme="classic")
        plt.ylim(0, ylim)
        plt.xlabel("position")
        plt.ylabel("IC")
        if title:
            plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(out_png.as_posix(), bbox_inches="tight",dpi=300,format="svg"); plt.close()
    else:
        # 没有 logomaker：用热力图展示信息矩阵
        plt.figure(figsize=(max(4, info_df.shape[0] * 0.22), 2.4),dpi=300,format="svg")
        sns.heatmap(info_df.T, cmap="viridis", cbar=True)
        plt.xlabel("position"); plt.ylabel("base")
        if title:
            plt.title(f"{title}  [info bits vs bg=0.25]", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_png.as_posix(), bbox_inches="tight"); plt.close()

def pwm_to_information_matrix(
    pwm_df: pd.DataFrame,
    background: Optional[Dict[str, float]] = None,
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

def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)

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

def ChangePwmtoInputFormat(pwm):
    """

    """
    output = {"A":[],"C":[],"G":[],"T":[]}
    # sortlist = {"A":[],"C":[],"G":[],"T":[]}
    sortlist = ["A","C","G","T"]
    for i in range(pwm.shape[0]):
        ShanoyE = 0
        for m in range(4):
            if pwm[i,m]>0:
                ShanoyE = ShanoyE - pwm[i,m]*np.log(pwm[i,m]) / np.log(2)
        IC = np.log(4)/np.log(2) - (ShanoyE)
        for j in range(4):
            # output[i].append([sortlist[j], pwm[i,j]*IC])
            output[sortlist[j]].append(pwm[i,j]*IC)
    output = pd.DataFrame(output)
    return output

df11 = pd.read_csv('./log/Train_KNET_plus_ic_test.tsv', sep='\t', header=None, names=['cls','dataset','model','kernel_len','random_seed','AUC','loss'])

df = df11[['dataset','model','AUC','kernel_len']].reset_index(drop=True)

dataet_ = 'RBFOX2_HepG2'
# dataet_ = 'NUDT21_HEK293'
# dataet_ = 'QKI_HEK293'
# dataet_ = 'U2AF2_Hela'

GPUID = '0'
best_para = df11[(df11['dataset'] == dataet_) &(df11['model'] == 'KNET_plus_ic')].sort_values(by='AUC', ascending=False).iloc[0,:][['random_seed','kernel_len']]
random_seed = best_para[0]
kernel_len = int(best_para[1])

random_seed = 666
kernel_len = 12

model1 = KNET_plus_ic(5, kernel_len, 128, 1)
path = f'/lustre/grp/gglab/liut/Kattention_aten_test/result/RBP/HDF5/{dataet_}/KNET_plus_ic/'
modelsave_tem = path + f'model_KernelNum-128kernel_size-{kernel_len}_seed-{random_seed}_opt-adamw.checkpointer.pt'
new_state = {}
state_dict = torch.load(modelsave_tem, map_location=torch.device('cpu'))
for key, value in state_dict.items():
    new_state[key.replace('module.', '')] = value
model1.load_state_dict(new_state)

DataPath = f'../../external/RBP/HDF5/{dataet_}/'
data_set = loadData(DataPath)
# Load dataset
test_set = data_set
test_set = TorchDataset_multi(test_set)
test_dataloader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, drop_last=False)

base_dir = './picture/figure/'

# 解析 pwm.txt
pwm_txt_path = "./pwm.txt"   # 改成真实路径
motif_dict = load_pwms_from_file(pwm_txt_path)   # { "630": (L,4), ... }

# 从前置文件构建 RBP→motifID 映射
meta_path = "./ATtRACT_db.txt"          # 就是你给的那种 RBFOX2\t...\t630\t1.000000**
rbp2motif_ids = build_rbp2motif_ids(meta_path)

target_rbp = dataet_   # 比如 "RBFOX2_HepG2"

# 改成：只取真正的 RBP 名
rbp_name = dataet_.split("_")[0]    # -> "RBFOX2"

motif_ids = rbp2motif_ids.get(rbp_name, [])

print(f"{target_rbp} motifs from meta file:", motif_ids)

# 输出目录：base_dir + f'{dataet_}/'
out_dir = Path(base_dir) / dataet_
out_dir.mkdir(parents=True, exist_ok=True)

for motif_id in motif_ids:
    if motif_id not in motif_dict:
        print(f"[WARN] motif_id {motif_id} not found in pwm.txt, skip.")
        continue

    pwm_array = motif_dict[motif_id]              # (L,4)
    pwm_df = pd.DataFrame(pwm_array,
                          columns=["A", "C", "G", "U"])

    out_svg = out_dir / f"ATtRACT_{motif_id}.svg"
    plot_pwm_logo(
        pwm_df,
        out_png=out_svg,
        title=f"{target_rbp} motif {motif_id}",
    )

    print(f"Saved logo: {out_svg}")

from statannotations.Annotator import Annotator

# ----- 1. RBP -> motif id 映射，取第一个 -----
rbp_name = dataet_.split("_")[0]               # "RBFOX2_HepG2" -> "RBFOX2"
motif_ids = rbp2motif_ids.get(rbp_name, [])
if not motif_ids:
    raise ValueError(f"No motif ids found for RBP {rbp_name} in meta file.")

motif_id = motif_ids[0]                        # 如果多个，只用第一个
print(f"Using ATtRACT motif {motif_id} for RBP {rbp_name}")

if motif_id not in motif_dict:
    raise KeyError(f"Motif id {motif_id} not found in pwm.txt")

pwm_array = motif_dict[motif_id]               # (Lmotif, 4)
Lmotif = pwm_array.shape[0]

BASES = ["A", "C", "G", "U"]

def motif_information_score(seq_pwm: np.ndarray,
                            struct_pwm: np.ndarray) -> float:
    """
    对一个 cluster 的 (seq_pwm, struct_pwm) 计算总信息量分数：
    总分 = sequence 正信息量 + structure 正信息量（单位 bits）
    """
    bg_seq = np.array([0.25, 0.25, 0.25, 0.25])
    bg_struct = np.array([0.5, 0.5])

    seq_info = pwm_to_information_matrix_kmer(seq_pwm, bg_seq)
    struct_info = pwm_to_information_matrix_kmer(struct_pwm, bg_struct)

    seq_pos = np.clip(seq_info, 0.0, None).sum()      # 只算正值
    struct_pos = np.clip(struct_info, 0.0, None).sum()

    return float(seq_pos + struct_pos)

def select_top_motifs(clusters,
                      seq_pwms,
                      struct_pwms,
                      top_k: int = 3,
                      min_cluster_size: int = 10,
                      min_info: float = 3.0):
    """
    clusters: list[list[dict]]，每个元素是一个簇的 k-mers（你聚簇的结果）
    seq_pwms: list[np.ndarray]，每个簇的序列 PWM (k,4)
    struct_pwms: list[np.ndarray]，每个簇的结构 PWM (k,2)
    top_k: 选择前多少个 motif
    min_cluster_size: 簇最小大小过滤（太小的簇不考虑）
    min_info: motif 最小信息量（bits）过滤

    返回: list[dict]，按“权重 + 信息量”排序后的前 top_k 个 motif，
         每个 dict 包含 pwm、cluster_size、weight、info_score 等信息。
    """
    assert len(clusters) == len(seq_pwms) == len(struct_pwms)

    total_kmers = sum(len(c) for c in clusters)
    selected = []

    for cl, seq_pwm, struct_pwm in zip(clusters, seq_pwms, struct_pwms):
        size = len(cl)
        if size < min_cluster_size:
            continue

        info_score = motif_information_score(seq_pwm, struct_pwm)
        if info_score < min_info:
            continue

        weight = size / total_kmers

        selected.append({
            "cluster": cl,
            "seq_pwm": seq_pwm,
            "struct_pwm": struct_pwm,
            "size": size,
            "weight": weight,
            "info": info_score,
        })

    # 按簇权重优先，其次按信息量排序（从大到小）
    selected.sort(key=lambda x: (x["weight"], x["info"]), reverse=True)

    return selected[:top_k]


import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = model1.to(device)
model1.eval()  # 不训练，但要保留梯度计算

###############################################################################
# 1. 在整个数据集上累积 Wattn.weight 的梯度（取绝对值平均）
###############################################################################
# 取得注意力卷积层的权重引用
wattn_weight = model1.Kattention.Wattn.weight   # shape: (out_channels, in_channels, 1)

# 用来累计 |grad| 的张量（和权重同形状）
grad_abs_sum = torch.zeros_like(wattn_weight, device=device)

criterion = nn.BCELoss()   # 模型最后已经 sigmoid 了，用 BCE

num_batches = 0

model1.eval()
Y_test = torch.tensor([])
Y_pred = torch.tensor([])
for X1_iter, X2_iter, Y_test_iter in tqdm(test_dataloader, desc="Accumulating gradients"):
    X1_iter = X1_iter.to(device)
    X2_iter = X2_iter.to(device)
    Y_test_iter = Y_test_iter.to(device).float()

    model1.zero_grad()

    # 前向
    Y_pred_iter = model1(X1_iter, X2_iter)  # (B,) 或 (B,1)

    # 视情况调整一下 label 形状，保证和输出对齐
    Y_test_iter = Y_test_iter.view_as(Y_pred_iter)

    loss = criterion(Y_pred_iter, Y_test_iter)

    # 反向传播，得到当前 batch 对 Wattn.weight 的梯度
    loss.backward()

    # 累加本 batch 的 |grad|，避免正负抵消
    grad_abs_sum += wattn_weight.grad.detach().abs()

    num_batches += 1

    Y_test = torch.concat([Y_test, Y_test_iter.cpu().detach()])

    try:
        Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
    except:
        Y_pred_iter = Y_pred_iter.cpu().detach()
        Y_pred_iter = torch.reshape(Y_pred_iter, (1,))
        Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])

from sklearn.metrics import f1_score, precision_recall_curve
#计算精确率、召回率和阈值
precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred)
#计算 F1 分数
f1_scores =2 * (precision * recall) / (precision + recall +1e-10)
# 找到最佳阈值
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1_score = f1_scores[best_index]

# 得到在数据集上平均的 |grad|
grad_abs_mean = grad_abs_sum / num_batches      # shape: (out_channels, in_channels, 1)

###############################################################################
# 2. 计算 |w * grad| 作为每个参数的 saliency，再按 kernel 聚合
###############################################################################
# 去掉最后的 kernel_size 维（=1）
weight = wattn_weight.detach().squeeze(-1)      # (out_channels, in_channels)
grad   = grad_abs_mean.detach().squeeze(-1)     # (out_channels, in_channels)

# 组合：|w * grad|，常见的一阶重要性度量
saliency = (weight.abs() * grad)                # (out_channels, in_channels)

###############################################################################
# 3. 按 (kernel, head, channel) reshape，并对 kernel 聚合
###############################################################################
# 从 Kattention 里面直接拿到超参数
kernel_len   = model1.Kattention.kernel_size    # k
channel_size = model1.Kattention.channel_size   # c1
num_heads    = model1.Kattention.num_heads      # h

# 检查维度是否一致（可选）
out_channels, in_channels = saliency.shape
assert out_channels == kernel_len * num_heads * channel_size, \
    f"out_channels={out_channels}, but k*h*c1={kernel_len*num_heads*channel_size}"

# 参考你给的写法：
# weight1_shape = rearrange(weight1, "(k h c1) c2 -> k c1 c2 h", k=kernel_len, c1=4, h=64)
saliency_reshaped = rearrange(
    saliency,
    '(k h c1) c2 -> k c1 c2 h',
    k=kernel_len,
    h=num_heads,
    c1=channel_size
)   # shape: (k, c1, c2, h)

# 对 c1, c2, h 三个维度聚合，得到每个 kernel 的重要性
# 你也可以用 .sum(...)，看你更喜欢“总量”还是“平均”
kernel_importance = saliency_reshaped.mean(dim=(1, 2, 3))   # shape: (k,)

###############################################################################
# 4. 对 kernel 重要性排序，并输出结果
###############################################################################
sorted_scores, sorted_indices = torch.sort(kernel_importance, descending=True)

print("=== Kernel importance ranking (high -> low) ===")
for rank, (idx, score) in enumerate(zip(sorted_indices.tolist(), sorted_scores.tolist()), start=1):
    print(f"Rank {rank:2d} | kernel {idx:2d} | importance = {score:.6f}")


# ---- 取出 Wattn 的权重 ----
# Wattn.weight: (out_channels, in_channels, 1)
wattn_weight = model1.Kattention.Wattn.weight.detach().cpu().squeeze(-1)  # (out_channels, in_channels)

kernel_len   = model1.Kattention.kernel_size     # k
num_heads    = model1.Kattention.num_heads       # h
channel_size = 5  # 这里你之前是 c1=4（A,C,G,U），如果不是 4 自己改

# 从你的注释参考：
# weight1_shape = rearrange(weight1, "(k h c1) c2 -> k c1 c2 h", k=kernel_len, c1=4, h=64)
kernels_all = rearrange(
    wattn_weight,                  # (out_channels, in_channels)
    '(k h c1) c2 -> k c1 c2 h',
    k=kernel_len,
    h=num_heads,
    c1=channel_size
)   # shape: (k, c1, c2, h)

top10_heads = sorted_indices[:10]

for rank, h_idx in enumerate(top10_heads, start=1):
    h_idx = int(h_idx)

    # 取出这个 head 对应的所有 kernel，形状: (kernel_len, 5, 5)
    kernels = kernels_all[:, :, :, h_idx].detach().cpu().numpy()

    fig, axes = plt.subplots(1, kernel_len, figsize=(24, 5))
    if kernel_len == 1:
        axes = [axes]

    for i in range(kernel_len):
        ax = axes[i]
        arr = kernels[i]      # (5, 5)

        # =============================
        # 行/列重排（前 4 个是 A,C,G,U，第 5 个是 ic）
        # 原 arr 的行/列顺序假设是 [A, U, C, G, ic]
        # 旧代码的 4×4 重排是:
        #   行: [0, 2, 3, 1]  -> A, C, G, U
        #   列: [0, 2, 3, 1]  -> A, C, G, U
        # 现在扩展为 5×5，并保持 ic 在最后:
        #   行: [0, 2, 3, 1, 4]
        #   列: [0, 2, 3, 1, 4]
        # =============================
        new_arr1 = np.empty((5, 5), dtype=arr.dtype)
        # 行重排
        new_arr1[0, :] = arr[0, :]   # A
        new_arr1[1, :] = arr[2, :]   # C
        new_arr1[2, :] = arr[3, :]   # G
        new_arr1[3, :] = arr[1, :]   # U (原来的 U 换到第 4 行)
        new_arr1[4, :] = arr[4, :]   # ic 保持在最后一行

        new_arr2 = np.empty((5, 5), dtype=arr.dtype)
        # 列重排
        new_arr2[:, 0] = new_arr1[:, 0]   # A
        new_arr2[:, 1] = new_arr1[:, 2]   # C
        new_arr2[:, 2] = new_arr1[:, 3]   # G
        new_arr2[:, 3] = new_arr1[:, 1]   # U
        new_arr2[:, 4] = new_arr1[:, 4]   # ic 保持在最后一列

        # =============================
        # 以 2 为底指数 + 行归一化（现在是 5×5）
        # =============================
        # exp_arr = np.power(2, new_arr2)
        # row_sums = exp_arr.sum(axis=1, keepdims=True)
        # proportional_array = exp_arr / row_sums
        proportional_array = new_arr2
        # =============================
        # 画热图
        # =============================
        if i == 0:
            sns.heatmap(
                proportional_array,
                ax=ax,
                cbar=False,
                # annot=True,
                fmt=".2f",
                cmap='vlag',
                square=True,
                center=0.0,
                xticklabels=["A", "C", "G", "U", "ic"],
                yticklabels=["A", "C", "G", "U", "ic"],
            )
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            sns.heatmap(
                proportional_array,
                ax=ax,
                cbar=False,
                # annot=True,
                fmt=".2f",
                cmap='vlag',
                square=True,
                center=0.0,
                xticklabels=["A", "C", "G", "U", "ic"],
                yticklabels=False,
            )

        ax.xaxis.tick_top()
        ax.tick_params(bottom=False, top=False, left=False, right=False)

    # fig.suptitle(f"Head {h_idx} (rank {rank})", y=1.02)

    savepath = base_dir + f'{dataet_}/rank{rank}_head{h_idx}/'
    mkdir(savepath)
    plt.savefig(savepath+'kerenl.svg', bbox_inches='tight', dpi=300, format="svg")
    plt.clf()
    plt.close(fig)

    # ===============================



def plot_integrative_logo(
    pwm_df: pd.DataFrame,
    ic_mean: np.ndarray,
    out_png: Union[str, Path],
    title: str = "",
    background: Optional[Dict[str, float]] = None,
):
    """
    综合 sequence PWM + icSHAPE 的 integrative motif：

    - 上半部分: A/C/G/U 的信息量 logo（只在 0 以上）
    - 下半部分: 结构信息 (U/P)，
        * icSHAPE >= 0.233 记为 U（unpaired）
        * icSHAPE <  0.233 记为 P（paired）
      高度 |value| ∈ [0,1]，阈值处为 0，越靠近 0 或 1 绝对值越大，方向都向下（0 以下）。
    """

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 1) PWM → 信息矩阵 (碱基部分)
    info_df = pwm_to_information_matrix(pwm_df, background=background)  # (L,4)
    L = info_df.shape[0]

    # 不允许 PWM 有负值，防止 A/C/G/U 出现在 0 以下
    info_df = info_df.clip(lower=0.0)

    # 2) icSHAPE → 结构信息矩阵
    ic_mean = np.asarray(ic_mean, dtype=float)
    threshold = 0.233  # median of all icSHAPE scores

    struct_mat = np.zeros((L, 2), dtype=float)  # 列为 ["U","P"]

    for i, m in enumerate(ic_mean):
        # 计算归一化高度 h ∈ [0,1]
        if m >= threshold:
            # U：unpaired，ic >= 0.233
            if 1.0 > threshold:
                h = (m - threshold) / (1.0 - threshold)
            else:
                h = 0.0
            h = float(np.clip(h, 0.0, 1.0))
            struct_mat[i, 0] = -h  # U 向下
            struct_mat[i, 1] = 0.0
        else:
            # P：paired，ic < 0.233
            if threshold > 0.0:
                h = (threshold - m) / threshold
            else:
                h = 0.0
            h = float(np.clip(h, 0.0, 1.0))
            struct_mat[i, 0] = 0.0
            struct_mat[i, 1] = -h  # P 向下

    struct_df = pd.DataFrame(struct_mat, columns=["U", "P"])

    if HAS_LOGOMAKER:
        # 用同一个坐标轴叠加两个 logo
        plt.figure(figsize=(max(4, L * 0.22), 3.0), dpi=300)
        ax = plt.gca()

        # 2.1 先画碱基信息（只在 0 以上）
        _ = lm.Logo(info_df, ax=ax, color_scheme="classic")

        # 2.2 再画结构信息（U/P 都在 0 以下）
        color_map = {"U": "gray", "P": "purple"}
        _ = lm.Logo(struct_df, ax=ax, color_scheme=color_map)

        # y 轴范围：上面最多 ~2 bits，下面 -1~0
        ax.set_ylim(-1.1, 2.1)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("position")
        ax.set_ylabel("Bits / structural score")
        if title:
            ax.set_title(title, fontsize=10)

        plt.tight_layout()
        plt.savefig(out_png.as_posix(), bbox_inches="tight", dpi=300, format="svg")
        plt.close()
    else:
        # fallback：用两个热力图拼一起
        fig, axes = plt.subplots(
            2, 1, figsize=(max(4, L * 0.22), 4.8), dpi=300,
            sharex=True
        )

        sns.heatmap(info_df.T, cmap="viridis", cbar=True, ax=axes[0])
        axes[0].set_ylabel("base")
        axes[0].set_title("sequence info")

        sns.heatmap(struct_df.T, cmap="coolwarm", center=-0.5, cbar=True, ax=axes[1])
        axes[1].set_ylabel("structure (U/P)")
        axes[1].set_xlabel("position")

        if title:
            fig.suptitle(title, fontsize=10)

        plt.tight_layout()
        plt.savefig(out_png.as_posix(), bbox_inches="tight", dpi=300, format="svg")
        plt.close()


# ============================================
# 1. 注册 hook：抓 Kattention 的 attn_logits
# ============================================
attn_logits_buffer = []

def kattn_hook(module, inputs, output):
    # output 是 KattentionV4.forward 返回的 dict
    attn = output["attn_logits"].detach().cpu()  # (B, H, Q, K)
    attn_logits_buffer.append(attn)

hook_handle = model1.Kattention.register_forward_hook(kattn_hook)

# ============================================
# 2. 辅助函数：从片段列表构建 PWM（只用前 4 个碱基通道）
# ============================================
def build_pwm_from_segments(segments_list):
    """
    segments_list: list[np.ndarray]，每个元素形状 (kernel_len, 5)
    返回: DataFrame, shape (kernel_len, 4)，列为 A,C,G,U 的频率矩阵
    """
    arr = np.stack(segments_list, axis=0)   # (N, kernel_len, 5)
    onehot = arr[..., :4]                   # (N, kernel_len, 4), 只用 A,C,G,U
    N, Lk, _ = onehot.shape

    # 每个位置取 argmax 获取碱基 index (0:A,1:C,2:G,3:U)，假定数据是 one-hot
    base_idx = np.argmax(onehot, axis=-1)   # (N, Lk)

    pwm = np.zeros((Lk, 4), dtype=float)
    for pos in range(Lk):
        counts = np.bincount(base_idx[:, pos], minlength=4)
        pwm[pos, :] = counts / float(N)

    pwm_df = pd.DataFrame(pwm, columns=["A", "C", "G", "U"])
    return pwm_df

def build_pwm_and_ic_from_segments(segments_list):
    """
    segments_list: list[np.ndarray]，每个 (kernel_len, 5)
      前4列: one-hot A,C,G,U
      第5列: icSHAPE ∈ [0,1]

    返回:
      pwm_df: (L,4) 频率矩阵 DataFrame，列为 A,C,G,U
      ic_mean: (L,) 每个位置的 icSHAPE 均值
    """
    arr = np.stack(segments_list, axis=0)   # (N, L, 5)
    onehot = arr[..., :4]                   # (N, L, 4)
    ic_vals = arr[..., 4]                   # (N, L)

    N, L, _ = onehot.shape

    # one-hot → 每个位置 base index
    base_idx = np.argmax(onehot, axis=-1)   # (N, L)

    pwm = np.zeros((L, 4), dtype=float)
    for pos in range(L):
        counts = np.bincount(base_idx[:, pos], minlength=4)
        pwm[pos, :] = counts / float(N)

    pwm_df = pd.DataFrame(pwm, columns=["A", "C", "G", "U"])
    ic_mean = ic_vals.mean(axis=0)          # (L,), 0~1

    return pwm_df, ic_mean


# ============================================
# 3. 对每一个 head 单独做：统计位置 + 提取片段 + 画图
# ============================================
for rank, h_idx in enumerate(top10_heads, start=1):
    h_idx = int(h_idx)

    savepath = base_dir + f'{dataet_}/rank{rank}_head{h_idx}/'
    print(f"Processing head {h_idx} (rank {rank})")

    seg1_list = []   # 存放第一个片段 (kernel_len, 5)
    seg2_list = []   # 存放第二个片段 (kernel_len, 5)
    pos_counts = None
    Q_valid = None
    K_valid = None
    L_seq   = None

    with torch.no_grad():
        for X1_iter, X2_iter, Y_iter in tqdm(
            test_dataloader, desc=f"Head {h_idx}"
        ):
            # 触发 forward，拿 attn_logits
            attn_logits_buffer.clear()

            X1_iter = X1_iter.to(device)
            X2_iter = X2_iter.to(device)
            Y_iter  = Y_iter.to(device)

            pred_iter = model1(X1_iter, X2_iter)    # 不需要输出，只为了触发 hook
            pred_cpu = pred_iter.detach().cpu().view(-1)
            assert len(attn_logits_buffer) == 1
            attn_logits = attn_logits_buffer[0]   # (B, H, Q, K)
            B, H, Q, K = attn_logits.shape

            X2_cpu = X2_iter.detach().cpu()       # (B, L, 5)
            X2_cpu_rev = torch.flip(X2_cpu, dims=[1])   # (Q_valid, K_valid)
            Y_cpu  = Y_iter.detach().cpu().view(-1)  # (B,)

            # 初始化一些尺寸信息 & 计数矩阵
            if L_seq is None:
                L_seq = X2_cpu.shape[1]           # 例如 101
                # 这里按照你的描述，对 Q,K 都取到 q-kernel_len+1, k-kernel_len+1
                Q_valid = Q
                K_valid = K
                pos_counts = np.zeros((Q_valid, K_valid), dtype=np.int64)

            # threshold = best_threshold   # 举例，阈值你可以在外面定义好再传进来
            threshold = 0.8
            # pdb.set_trace()
            # mask = (Y_cpu == 1) & (pred_cpu > threshold)
            mask = pred_cpu > threshold

            pos_idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if pos_idx.numel() == 0:
                continue

            for b in pos_idx.tolist():
                # 取该样本在当前 head 上的注意力矩阵
                attn_mat = attn_logits[b, h_idx]      # (Q, K)
                # 只在 [0:Q_valid, 0:K_valid] 范围内找最大值
                attn_sub = attn_mat

                # 沿最后一维 flip（列翻转）
                attn_flip = torch.flip(attn_sub, dims=[-1])   # (Q_valid, K_valid)

                # 找最大值及其位置
                flat_idx = attn_flip.view(-1).argmax()
                max_val  = attn_flip.view(-1)[flat_idx]

                i = (flat_idx // K_valid).item()   # 行 (query)
                j = (flat_idx %  K_valid).item()   # 列 (key)

                # 累计这个位置的最大值出现次数
                pos_counts[i, j] += 1

                # 从原始输入 X2 中提取两个长度为 kernel_len 的片段
                # 注意边界
                if i + kernel_len <= L_seq and j + kernel_len <= L_seq:
                    seg1 = X2_cpu[b, i:i+kernel_len, :]   # (kernel_len, 5)
                    seg2 = torch.flip(X2_cpu_rev[b, j:j+kernel_len, :], dims=[1])

                    seg1_list.append(seg1.numpy())
                    seg2_list.append(seg2.numpy())
                # 越界的话跳过

    # ==============================
    # 4. 保存位置次数热图 + 两个 PWM logo
    # ==============================
    # save_prefix = os.path.join(base_dir, f"{dataet_}", f"rank{rank}_head{h_idx}")
    # os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    # 4.1 绘制位置计数 heatmap
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(pos_counts, cmap="Reds", cbar=True)
    plt.xlabel("j (key index)")
    plt.ylabel("i (query index)")
    plt.title(f"Max-attn position counts\nhead {h_idx} (rank {rank})")
    plt.tight_layout()
    plt.savefig(savepath + "maxpos_heatmap.svg", bbox_inches="tight",dpi=300,format="svg")
    plt.close()

    # # 4.2 seg1 的 PWM + logo
    # if len(seg1_list) > 0:
    #     pwm1_df = build_pwm_from_segments(seg1_list)
    #     plot_pwm_logo(
    #         pwm1_df,
    #         out_png=save_prefix + "_seg1_logo.png",
    #         title=f"Head {h_idx} rank{rank} seg1",
    #     )

    # # 4.3 seg2 的 PWM + logo
    # if len(seg2_list) > 0:
    #     pwm2_df = build_pwm_from_segments(seg2_list)
    #     plot_pwm_logo(
    #         pwm2_df,
    #         out_png=save_prefix + "_seg2_logo.png",
    #         title=f"Head {h_idx} rank{rank} seg2",
    #     )
    # seg1

    BASES = ["A", "C", "G", "U"]   # 序列
    STRUCT_STATES = ["U", "P"]     # 结构

    # run_motif_discovery(seg1_list,
    #                     k=6,
    #                     struct_threshold=0.0,   # 根据你的 icSHAPE 缩放调整
    #                     top_k_clusters=3,
    #                     out_dir= savepath+ f"Head {h_idx} rank{rank}/integrative_motifs")

    run_motif_discovery_with_ref(
        regions=seg1_list,
        ref_pwm=pwm_array,
        out_dir=savepath+ f"Head {h_idx} rank{rank}/seq1",
        max_seq_mismatch=1,     # k-mer 序列允许的 mismatch 数
        min_cluster_size=5,     # 簇至少有多少个 k-mer 才参与比较
        top_k=3,                # 保存相似度最高的 3 个 seq logo
    )

    run_motif_discovery_with_ref(
        regions=seg2_list,
        ref_pwm=pwm_array,
        out_dir=savepath+ f"Head {h_idx} rank{rank}/seq2",
        max_seq_mismatch=1,     # k-mer 序列允许的 mismatch 数
        min_cluster_size=5,     # 簇至少有多少个 k-mer 才参与比较
        top_k=3,                # 保存相似度最高的 3 个 seq logo
    )

    if len(seg1_list) > 0:
        pwm1_df, ic1_mean = build_pwm_and_ic_from_segments(seg1_list)
        plot_integrative_logo(
            pwm1_df,
            ic_mean=ic1_mean,
            out_png=savepath + "seq1_integrative_logo.svg",
            title=f"Head {h_idx} rank{rank} seq1",
        )

    # seg2
    if len(seg2_list) > 0:
        pwm2_df, ic2_mean = build_pwm_and_ic_from_segments(seg2_list)
        plot_integrative_logo(
            pwm2_df,
            ic_mean=ic2_mean,
            out_png=savepath + "seq2_integrative_logo.svg",
            title=f"Head {h_idx} rank{rank} seq2",
        )
# 用完清理 hook
hook_handle.remove()
print("任务二完成：所有 head 的位置热图 & seg1/seg2 PWM logo 已保存。")


# ----- hook: 抓 attn_logits -----
attn_logits_buffer = []

def kattn_hook(module, inputs, output):
    attn = output["attn_logits"].detach().cpu()  # (B, H, Q, K)
    attn_logits_buffer.append(attn)

hook_handle = model1.Kattention.register_forward_hook(kattn_hook)


for rank, h_idx in enumerate(top10_heads, start=1):
    h_idx = int(h_idx)

    ATtRACT_positive_list = []
    ATtRACT_negative_list = []
    kattn_positive_list   = []
    kattn_negative_list   = []

    with torch.no_grad():
        for X1_iter, X2_iter, Y_iter in test_dataloader:
            attn_logits_buffer.clear()

            X1_iter = X1_iter.to(device)
            X2_iter = X2_iter.to(device)
            Y_iter  = Y_iter.to(device)

            # 前向，触发 Kattention hook
            _ = model1(X1_iter, X2_iter)
            assert len(attn_logits_buffer) == 1
            attn_logits = attn_logits_buffer[0]         # (B, H, Q, K)

            B, H, Q, K = attn_logits.shape
            X2_cpu = X2_iter.detach().cpu()             # (B, Lseq, 5)
            Y_cpu  = Y_iter.detach().cpu().view(-1)     # (B,)

            for b in range(B):
                y = int(Y_cpu[b].item())
                seq_b = X2_cpu[b]                       # (Lseq, 5)

                # 1) ATtRACT 最大 PWM 打分
                pwm_score = scan_pwm_max_score_one_seq(seq_b, pwm_array)

                # 2) Kattention 当前 head 的最大 attn_logits 值
                attn_mat = attn_logits[b, h_idx]        # (Q, K)
                # 你之前有裁剪 Q_valid / K_valid 的话，可以这里加；否则就全局最大
                kattn_score = float(attn_mat.max().item())

                if y == 1:
                    ATtRACT_positive_list.append(pwm_score)
                    kattn_positive_list.append(kattn_score)
                else:
                    ATtRACT_negative_list.append(pwm_score)
                    kattn_negative_list.append(kattn_score)

    print(
        f"Head {h_idx} rank {rank}: "
        f"ATtRACT pos={len(ATtRACT_positive_list)}, neg={len(ATtRACT_negative_list)}, "
        f"kattn pos={len(kattn_positive_list)}, neg={len(kattn_negative_list)}"
    )

    # ===== 组装 DataFrame ===== 
    data_rows = []

    for v in ATtRACT_positive_list:
        data_rows.append({"model": "ATtRACT_pos", "score": v})
    for v in ATtRACT_negative_list:
        data_rows.append({"model": "ATtRACT_neg", "score": v})
    for v in kattn_positive_list:
        data_rows.append({"model": "KATTN_pos", "score": v})
    for v in kattn_negative_list:
        data_rows.append({"model": "KATTN_neg", "score": v})

    df_all = pd.DataFrame(data_rows)

    # ===== 绘图保存路径 =====
    save_dir = Path(base_dir) / f"{dataet_}" / f"rank{rank}_head{h_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 图1：ATtRACT_pos vs ATtRACT_neg
    # ============================================================
    df_attr = df_all[df_all["model"].str.startswith("ATtRACT")].copy()
    # 保证顺序：neg 在左，pos 在右（可按需调整）
    order_attr = ["ATtRACT_neg", "ATtRACT_pos"]

    out_attr = save_dir / "ATtRACT_violin.svg"

    plt.figure(figsize=(4, 3))
    palette_attr = ['#B0B1B6', '#B0B1B6']  # 统一灰色系

    ax1 = sns.violinplot(
        x="model",
        y="score",
        data=df_attr,
        order=order_attr,
        linewidth=2,
        edgecolor="black",
        palette=palette_attr,
        cut=0
    )

    pairs_attr = [("ATtRACT_pos", "ATtRACT_neg")]

    annotator_attr = Annotator(
        ax1,
        pairs_attr,
        x="model",
        y="score",
        data=df_attr,
        order=order_attr,
        plot="violinplot",
    )

    annotator_attr.configure(
        test="Mann-Whitney",   # 或 "Wilcoxon"
        text_format="star",
        loc="inside",
        verbose=0
    )
    annotator_attr.apply_and_annotate()

    ax1.set_xlabel("")
    ax1.set_ylabel("Max score", fontsize=12)
    ax1.set_title("ATtRACT", fontsize=13, pad=8)

    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)

    sns.despine(ax=ax1)
    plt.tight_layout()
    plt.savefig(out_attr.as_posix(), dpi=300, format="svg")
    plt.close()
    print(f"Saved ATtRACT comparison plot to: {out_attr}")

    # ============================================================
    # 图2：KATTN_pos vs KATTN_neg
    # ============================================================
    df_kattn = df_all[df_all["model"].str.startswith("KATTN")].copy()
    order_kattn = ["KATTN_neg", "KATTN_pos"]

    out_kattn = save_dir / "KATTN_violin.svg"

    plt.figure(figsize=(4, 3))
    palette_kattn = ['#5DA0B6', '#5DA0B6']  # 统一蓝色系

    ax2 = sns.violinplot(
        x="model",
        y="score",
        data=df_kattn,
        order=order_kattn,
        linewidth=2,
        edgecolor="black",
        palette=palette_kattn,
        cut=0
    )

    pairs_kattn = [("KATTN_pos", "KATTN_neg")]

    annotator_kattn = Annotator(
        ax2,
        pairs_kattn,
        x="model",
        y="score",
        data=df_kattn,
        order=order_kattn,
        plot="violinplot",
    )

    annotator_kattn.configure(
        test="Mann-Whitney",
        text_format="star",
        loc="inside",
        verbose=0
    )
    annotator_kattn.apply_and_annotate()

    ax2.set_xlabel("")
    ax2.set_ylabel("Max score", fontsize=12)
    ax2.set_title("K-NET kernel", fontsize=13, pad=8)

    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)

    sns.despine(ax=ax2)
    plt.tight_layout()
    plt.savefig(out_kattn.as_posix(), dpi=300, format="svg")
    plt.close()
    print(f"Saved KATTN comparison plot to: {out_kattn}")

# 用完记得把 hook 卸掉（如果后面不用了）
hook_handle.remove()
