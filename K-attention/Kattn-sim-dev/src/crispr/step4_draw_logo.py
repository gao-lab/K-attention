import pandas as pd
import numpy as np
import torch
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import logomaker
import string
import pdb
# ===== 1) 读取并拆分 seq1;seq2 =====
csv_path = "../../results/Crispr/doench2014-Hs/KNET_Crispr/attn_logits/Kattention1.kattn/1.csv"  # 换成你的路径
save_ = csv_path.split("/")
save_name = save_[-2].split(".")[0] + '_' + save_[-1].split(".")[0]

df = pd.read_csv(csv_path)

# 列名就是 'seq1;seq2'，形如 "GGTCG;CGTCT"
pairs = df['seq1;seq2'].astype(str).str.split(';', expand=True)
pairs.columns = ['seq1', 'seq2']

# 可选：统一大写并去除空白
pairs['seq1'] = pairs['seq1'].str.upper().str.strip()
pairs['seq2'] = pairs['seq2'].str.upper().str.strip()

# ===== 2) one-hot 编码 =====
BASES = ['A', 'C', 'G', 'T']
base_to_idx = {b:i for i,b in enumerate(BASES)}

def one_hot_encode(seq: str, pad_to: int|None=None, pad='N'):
    """返回 [L,4] 的 one-hot；非 ACGT 置 0（或按需要在这里处理）。"""
    if pad_to is not None:
        seq = (seq[:pad_to]).ljust(pad_to, pad)
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for pos, ch in enumerate(seq):
        i = base_to_idx.get(ch, None)
        if i is not None:
            arr[pos, i] = 1.0
        # 其他字符（如 N/U ）保持全 0
    return torch.from_numpy(arr)

# —— 统一长度策略：截断到所有序列的最短长度（也可以改为填充到最长）
lengths = pairs.applymap(len)
L = int(min(lengths.min()))   # 公共长度
N = len(pairs)

# 生成 [N, L, 4] 的张量
stacked_tensors1 = torch.stack([one_hot_encode(s, pad_to=L) for s in pairs['seq1'].tolist()], dim=0)
stacked_tensors2 = torch.stack([one_hot_encode(s, pad_to=L) for s in pairs['seq2'].tolist()], dim=0)
print("stacked_tensors1:", stacked_tensors1.shape, " stacked_tensors2:", stacked_tensors2.shape)
# 例如：torch.Size([N, L, 4])

# ===== 3) 把两个 one-hot 在每个位置组合成二核苷酸，并统计 =====
def convert_one_hot_to_atcg(two_onehots: torch.Tensor):
    """
    输入：two_onehots 形状 [N, 2, 4]，表示同一位置（seq1与seq2）的 one-hot。
    输出：长度为 N 的字符串列表，如 'AA','AC',...,'TT'
    """
    idx = two_onehots.argmax(dim=-1)      # [N, 2]
    letters = torch.tensor(list(range(4)))  # 0,1,2,3
    # 映射到字符
    map_idx_to_base = np.array(BASES)
    s1 = map_idx_to_base[idx[:,0].cpu().numpy()]
    s2 = map_idx_to_base[idx[:,1].cpu().numpy()]
    return [a + b for a, b in zip(s1, s2)]

kernel_len = L
non_pwm_list = []
for frag in range(kernel_len):
    # [N, 2, 4]：取两条序列在同一位置的 one-hot
    test = torch.stack((stacked_tensors1[:, frag, :],
                        stacked_tensors2[:, frag, :]), dim=1)
    test_seq = convert_one_hot_to_atcg(test)

    # 计数 16 种二核苷酸
    counted_elements = Counter(test_seq)
    keys = ['AA','AC','AG','AT','CA','CC','CG','CT',
            'GA','GC','GG','GT','TA','TC','TG','TT']
    counts = [counted_elements.get(k, 0) for k in keys]
    total = sum(counts)
    if total == 0:
        proportions = torch.zeros(16, dtype=torch.float32)
    else:
        proportions = torch.tensor([c/total for c in counts], dtype=torch.float32)

    # 信息量（IC）
    H  = -torch.sum(proportions * torch.log2(proportions + 1e-10))
    IC = 4 - H
    non_pwm_list.append(proportions * IC)

non_pwm_array = torch.stack(non_pwm_list)          # [L, 16]
clamped_array = torch.clamp(non_pwm_array, min=0)  # <0 归零

# # 列名 A..P（16列）
cols = [chr(i) for i in range(ord('A'), ord('P') + 1)]
non_pwm_df = pd.DataFrame(clamped_array.numpy(), columns=cols)
# keys = ['AA','AC','AG','AT',
#         'CA','CC','CG','CT',
#         'GA','GC','GG','GT',
#         'TA','TC','TG','TT']

# non_pwm_df = pd.DataFrame(clamped_array.numpy(), columns=keys)

# ===== 4) 绘制 motif logo =====
plt.figure(figsize=(8, 4))
logo = logomaker.Logo(non_pwm_df)
logo.style_spines(visible=False)
logo.style_spines(spines=['left', 'bottom'], visible=True)
logo.style_xticks(rotation=0, fmt='%d')
logo.ax.set_ylim([0, 4])
logo.ax.set_ylabel('Bits', labelpad=-1)
plt.title("Non-PWM Motif Logo", fontsize=14)

# 保存
dataet_ = "my_dataset"  # 可改
save_dir = Path(f"../draw/logo/{save_name}/")
save_dir.mkdir(parents=True, exist_ok=True)
out_path = save_dir / "logo.svg"
plt.savefig(out_path, bbox_inches='tight', dpi=300, format="svg")
plt.close()
print(f"saved to: {out_path}")

# letter_ = [chr(i) for i in range(ord('A'), ord('P') + 1)]
# for i in range(17):
#     print(f'{letter_[i]}:{keys[i]}')
    

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# 可作为备用的热力图/散点
import seaborn as sns

# ====== logomaker 可选依赖 ======
try:
    import logomaker as lm
    HAS_LOGOMAKER = True
except Exception:
    HAS_LOGOMAKER = False
    print("[warn] logomaker not found. Will fallback to heatmap.")

# =========================================================
# 你提供的函数（略作整理，保持原逻辑）
# =========================================================
def pwm_to_information_matrix(
    pwm_df: pd.DataFrame,
    background: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    将 PWM 频率矩阵 (L x 4) 转换为信息矩阵 (L x 4)，元素为 p*log2(p/bg)；p=0 时置 0。
    background: 背景分布，默认按列名均匀，如 {"A":0.25,"C":0.25,"G":0.25,"U":0.25}
    """
    if background is None:
        background = {col: 1.0 / pwm_df.shape[1] for col in pwm_df.columns}

    P = pwm_df.to_numpy(dtype=float)             # (L,4)
    cols = list(pwm_df.columns)
    bg = np.array([background[c] for c in cols], dtype=float)  # (4,)

    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.where(P > 0, P * (np.log(P / bg) / np.log(2.0)), 0.0)  # bits

    info_df = pd.DataFrame(R, columns=cols)
    return info_df

def pwm_to_classic_logo_matrix(pwm_df: pd.DataFrame) -> pd.DataFrame:
    """
    经典 sequence logo 高度矩阵：
      H(pos)  = -sum_b p_b log2(p_b)
      IC(pos) = log2(|alphabet|) - H(pos)  (DNA/RNA 4 -> 2 bits)
      height  = p_b * IC(pos)
    输入 pwm_df: (L x 4) 频率矩阵，行和应为 1（或接近 1）。
    输出: (L x 4) 每个碱基在该位点的“高度”(bits)，非负。
    """
    P = pwm_df.to_numpy(dtype=float)          # (L,4)
    cols = list(pwm_df.columns)
    A = P.shape[1]                            # alphabet size, 4

    with np.errstate(divide='ignore', invalid='ignore'):
        logP = np.where(P > 0, np.log2(P), 0.0)
        H = -np.sum(P * logP, axis=1, keepdims=True)   # (L,1)
    IC = np.log2(A) - H                                 # (L,1)

    R = P * IC                                          # (L,4)
    R = np.clip(R, 0.0, None)                            # 数值稳定：去掉极小负数

    return pd.DataFrame(R, columns=cols)


def plot_pwm_logo(
    pwm_df: pd.DataFrame,
    out_png: str | Path,
    title: str = "",
    background: dict[str, float] | None = None,
):
    """
    以“相对背景的比特信息量（bits）”绘制 sequence logo。
    - 若安装了 logomaker，则画标准 logo；否则退化为信息矩阵热力图。
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 1) 频率 -> 信息矩阵（p*log2(p/bg)）
    # info_df = pwm_to_information_matrix(pwm_df, background=background)
    info_df = pwm_to_classic_logo_matrix(pwm_df)

    # 2) y 轴上限（可按需调节）
    total_bits_per_pos = info_df.sum(axis=1).to_numpy()
    ymax = float(total_bits_per_pos.max() if total_bits_per_pos.size else 0.0)
    ylim = max(1.0, min(2.0, ymax * 1.1))  # 简单限制到 [1,2] 之间；你也可固定成 2

    if HAS_LOGOMAKER:
        plt.figure(figsize=(max(4, info_df.shape[0] * 0.22), 2.4), dpi=160)
        logo = lm.Logo(info_df, color_scheme="classic")
        plt.ylim(0, 2)   
        plt.xlabel("position")
        plt.ylabel("IC (bits)")
        if title:
            plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(out_png.as_posix(), bbox_inches="tight")
        plt.close()
    else:
        plt.figure(figsize=(max(4, info_df.shape[0] * 0.22), 2.4), dpi=160)
        sns.heatmap(info_df.T, cmap="viridis", cbar=True)
        plt.xlabel("position")
        plt.ylabel("base")
        if title:
            plt.title(f"{title}  [info bits vs bg]", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_png.as_posix(), bbox_inches="tight")
        plt.close()


def plot_logits_per_sample(
    logits_qk: torch.Tensor,        # [N, Q, K]
    keep_mask: np.ndarray,          # bool 数组 [N]
    out_dir: str | Path,
    max_plots: int | None = None,
    dpi: int = 160,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logits_np = logits_qk.detach().cpu().numpy()  # [N,Q,K]
    idxs = np.where(keep_mask)[0]
    if max_plots is not None:
        idxs = idxs[:max_plots]

    for idx in idxs:
        mat = logits_np[idx]  # [Q,K]
        plt.figure(figsize=(6, 5), dpi=dpi)
        sns.heatmap(mat, cmap="viridis", cbar=True)
        plt.title(f"sample #{int(idx)} logits [QxK]={mat.shape[0]}x{mat.shape[1]}")
        plt.xlabel("k index")
        plt.ylabel("q index")
        out_png = Path(out_dir) / f"sample_{int(idx)}.png"
        plt.tight_layout()
        plt.savefig(out_png.as_posix(), bbox_inches="tight")
        plt.close()

    print(f"[per-sample logits] saved {len(idxs)} figures to: {Path(out_dir).resolve()}")

# =========================================================
# 构建 PWM 的辅助函数
# =========================================================
ALPHABET = ["A", "C", "G", "U"]  # 用 U 表示 RNA；DNA 会把 T 统一映射到 U

def clean_and_map_to_acgu(seq: str) -> str:
    """统一大写，去空白；把 T->U；保留 A/C/G/U，其它字符（N等）原样保留以便后续跳过计数。"""
    s = (seq or "").upper().strip()
    s = s.replace("T", "U")
    return s

def build_pwm_from_sequences(
    sequences: list[str],
    mode: str = "truncate",       # 'truncate'：截断到最短；'pad'：右侧填充到最长
    pad_char: str = "N",
    pseudocount: float = 0.0,     # 拉普拉斯平滑；如需避免 0，可设为 0.5 或 1.0
) -> pd.DataFrame:
    """
    根据一组同向序列构建 (L x 4) 的 PWM 频率矩阵（列顺序 A,C,G,U）。
    - 非 A/C/G/U 的碱基会被跳过，不计入该位的分母。
    """
    seqs = [clean_and_map_to_acgu(s) for s in sequences]
    lengths = [len(s) for s in seqs if s is not None]
    if not lengths:
        raise ValueError("No sequences found.")
    if mode == "truncate":
        L = min(lengths)
        seqs = [s[:L] for s in seqs]
    elif mode == "pad":
        L = max(lengths)
        seqs = [s.ljust(L, pad_char) for s in seqs]
    else:
        raise ValueError("mode must be 'truncate' or 'pad'.")

    # 计数矩阵 (L x 4)
    counts = np.zeros((L, 4), dtype=float) + pseudocount
    for s in seqs:
        for pos, ch in enumerate(s):
            if ch in ALPHABET:
                j = ALPHABET.index(ch)
                counts[pos, j] += 1.0
            else:
                # 跳过未知字符（N 等），不加计数；若使用 pad 模式，可考虑计入分母或不计入
                pass

    # 每列按位归一化 -> 频率
    # 分母为每个位的有效碱基总数（加了 pseudocount）
    denom = counts.sum(axis=1, keepdims=True)  # (L,1)
    with np.errstate(divide='ignore', invalid='ignore'):
        pwm = np.divide(counts, denom, out=np.zeros_like(counts), where=(denom > 0))

    pwm_df = pd.DataFrame(pwm, columns=ALPHABET)
    return pwm_df

# =========================================================
# 主流程：读取 CSV，拆分 seq1;seq2，分别构建 PWM 并绘图
# =========================================================
def main(
    csv_path: str | Path,
    out_dir: str | Path = "./pwm_logo_out",
    truncate_or_pad: str = "truncate",  # 'truncate' or 'pad'
    pseudocount: float = 0.0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "seq1;seq2" not in df.columns:
        raise KeyError("CSV 必须包含列名 'seq1;seq2'（例如 'GGTCG;CGTCT'）。")

    # 拆分
    pair_df = df["seq1;seq2"].astype(str).str.split(";", expand=True)
    pair_df.columns = ["seq1", "seq2"]

    # 分别收集序列
    seq1_list = pair_df["seq1"].dropna().astype(str).tolist()
    seq2_list = pair_df["seq2"].dropna().astype(str).tolist()

    # 分别构建 PWM
    pwm1 = build_pwm_from_sequences(seq1_list, mode=truncate_or_pad, pseudocount=pseudocount)
    pwm2 = build_pwm_from_sequences(seq2_list, mode=truncate_or_pad, pseudocount=pseudocount)

    # 保存 PWM 表（可选）
    pwm1.to_csv(out_dir / "pwm_seq1.csv", index=False)
    pwm2.to_csv(out_dir / "pwm_seq2.csv", index=False)

    # 分别绘制 logo
    plot_pwm_logo(pwm1, out_dir / "logo_seq1.png", title="PWM Logo (seq1)")
    plot_pwm_logo(pwm2, out_dir / "logo_seq2.png", title="PWM Logo (seq2)")

    print(f"[done] PWM & logos saved to: {out_dir.resolve()}")

# ===== 执行示例 =====
# main("../../results/Crispr/doench2014-Hs/KNET_Crispr/attn_logits/Kattention2.kattn/0.csv", out_dir="../draw/logo/", truncate_or_pad="truncate", pseudocount=0.0)
main(csv_path, out_dir=f"../draw/logo/{save_name}/", truncate_or_pad="truncate", pseudocount=0.0)
