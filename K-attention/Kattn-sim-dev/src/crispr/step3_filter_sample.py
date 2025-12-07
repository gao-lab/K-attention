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

from step2_analysist_kernel import _find_module_by_name

def _find_parent_self_kattn(root: torch.nn.Module, dotted: str):
    dotted = dotted.strip().lstrip(".")
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        cand = ".".join(parts[:cut])
        try:
            mod = _find_module_by_name(root, cand)
        except KeyError:
            continue
        if getattr(mod, "__class__", type(None)).__name__ == "self_kattn":
            return mod
    # dotted 本身是 self_kattn 也返回它
    try:
        mod = _find_module_by_name(root, dotted)
        if getattr(mod, "__class__", type(None)).__name__ == "self_kattn":
            return mod
    except KeyError:
        pass
    return None

def complementary_score(seq1: str, seq2: str) -> int:
    """
    计算两条等长DNA序列的互补得分：
    同一位置碱基互补(A-T, T-A, C-G, G-C)则+1，返回总分。
    """
    if len(seq1) != len(seq2):
        raise ValueError("两序列长度必须相同。")
    
    # 统一大写并去空白
    s1 = seq1.replace(" ", "").upper()
    s2 = seq2.replace(" ", "").upper()
    if len(s1) != len(s2):
        raise ValueError("去除空白后长度不一致。")

    valid = set("ATCG")
    if not set(s1).issubset(valid) or not set(s2).issubset(valid):
        raise ValueError("序列只能包含 A/T/C/G。")

    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    score = sum(1 for a, b in zip(s1, s2) if comp[a] == b)
    return score

@torch.no_grad()
def export_max_attn_logits_to_csv(
    model: torch.nn.Module,
    loader,
    DEVICE,
    layer_name: str,         # 如 "model.Kattention1" 或 "model.Kattention1.kattn"
    head_idx: int,           # 指定 head（0-based）
    output_csv: str,
    label_keys: Tuple[str, ...] = ("mean_eff","me"),
):
    """
    遍历 loader，收集指定层/指定 head 的 attn_logits 最大值与位置，并导出 CSV。
    CSV 列：idx, head, max_logit, i, j, i_minus_j, label, input_ids
    """
    model.eval().to(DEVICE)

    # 解析模块与父 self_kattn（读 h_d 和 reverse）
    try:
        mod = _find_module_by_name(model, layer_name)
    except KeyError:
        mod = _find_module_by_name(model, layer_name + ".kattn")

    if getattr(mod, "__class__", type(None)).__name__ == "self_kattn":
        if not hasattr(mod, "kattn"):
            raise KeyError(f"'{layer_name}' 是 self_kattn 但没有子模块 'kattn'")
        parent = mod
        mod_kattn = mod.kattn
    else:
        mod_kattn = mod
        parent = _find_parent_self_kattn(model, layer_name)
        if parent is None:
            raise RuntimeError(f"找不到 '{layer_name}' 的父 self_kattn（用于读取 k_n/h_d/reverse）")

    if getattr(mod_kattn, "__class__", type(None)).__name__ != "KattentionV4":
        raise TypeError(f"目标模块不是 KattentionV4（got {type(mod_kattn).__name__}）")

    H  = int(parent.k_n)
    D  = int(parent.h_d)
    rv = bool(getattr(parent, "reverse", False))

    # hook 捕获 attn_logits（原始输出）；稍后按 self_kattn 规则做 reverse/scale
    buffer_logits: List[torch.Tensor] = []
    def _hook(_m, _in, out):
        A = out["attn_logits"]            # [B,H,L,L] 或 [B,L,L]
        if A.dim() == 3:
            B, Lq, Lk = A.shape
            A = A.unsqueeze(1).expand(B, H, Lq, Lk)
        # reverse（如需要）
        if rv:
            A = A.flip([-1])
        # 按 forward 规则缩放
        # A = A / math.sqrt(float(D))
        # 取指定 head
        if head_idx >= A.size(1):
            raise IndexError(f"head_idx {head_idx} 超界，H={A.size(1)}")
        buffer_logits.append(A[:, head_idx, ...].detach().cpu())  # [B,L,L]

    handle = mod_kattn.register_forward_hook(_hook)

    rows = []
    sample_offset = 0
    for bi, batch in enumerate(loader):
        # label
        y = None
        for k in label_keys:
            if k in batch:
                y = batch[k]; break
        if y is None:
            raise KeyError(f"batch 里找不到 label（候选 keys={label_keys}）")
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        y = y.view(-1)

        # ids：保存**整数 token**用于 CSV
        ids = batch["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids)
        if ids.dim() == 3:  # 若为 one-hot，取 argmax 还原
            ids_int = ids.argmax(dim=-1)
        else:
            ids_int = ids.long()

        # 触发 forward（hook 会抓 logits）
        y_for_loss = y.to(DEVICE).float()
        ids_for_model = batch["input_ids"].to(DEVICE)
        crisproff = batch.get("CRISPRoff_score", None)
        if isinstance(crisproff, torch.Tensor):
            crisproff = crisproff.to(DEVICE)
        _ = model(ids_for_model, y_for_loss, crisproff)

        # 取出本批 logits
        if len(buffer_logits) == 0:
            raise RuntimeError("未捕获到 attn_logits，检查 layer_name 是否正确")
        A = buffer_logits.pop(0)            # [B,L,L]，已做 reverse/scale
        B, Lq, Lk = A.shape
        A_flat = A.view(B, -1)
        flat_idx = torch.argmax(A_flat, dim=1)       # [B]
        i = (flat_idx // Lk).to(torch.long)          # 行（query）
        j = (flat_idx %  Lk).to(torch.long)          # 列（key）
        vmax = A[torch.arange(B), i, j]              # [B]

        # 写行
        y_np = y.detach().cpu().numpy()
        ids_np = ids_int.detach().cpu().numpy()
        for b in range(B):
            if (3<i[b]<24-5) & (4<j[b]<24-5):
                dna = ids_to_dna(ids_np[b])
                rna = extract_guide_rna(dna, start=5, length=20)
                seq1 = dna[i[b]-2:i[b]-2+5]
                seq2 = dna[j[b]-2:j[b]-2+5][::-1]
                rows.append({
                    "idx": sample_offset + b,
                    "head": head_idx,
                    "max_logit": float(vmax[b].item()),
                    "i": int(i[b].item()),
                    "j": int(j[b].item()),
                    "i_minus_j": int(i[b].item() - j[b].item()),
                    "label": float(y_np[b]),
                    "dna_seq": ids_to_dna(ids_np[b]),  # ← 新增
                    "guide_RNA": rna,
                    "seq1;seq2": f"{seq1};{seq2}",
                    "score": f"{complementary_score(seq1,seq2)}"
                })
        sample_offset += B

    handle.remove()

    df = pd.DataFrame(
        rows,
        columns=["idx","head","max_logit","i","j","i_minus_j","label","dna_seq","guide_RNA","seq1;seq2","score"]
    )
    df.to_csv(output_csv, index=False)
    print(f"[done] saved: {output_csv}  (rows={len(df)})")

ID2DNA = {3: "A", 4: "C", 5: "G", 6: "T"}  # 其余 -> 'N'

def ids_to_dna(ids_1d: np.ndarray) -> str:
    """ids_1d: [L] 的整数 token 序列 -> DNA 字符串"""
    return "".join(ID2DNA.get(int(t), "N") for t in ids_1d.tolist())

def extract_guide_rna(dna_seq: str, start: int = 5, length: int = 20, reverse_complement: bool = False) -> str:
    """
    从第5个碱基起取20个（1-based），对该 DNA 片段做互补并输出 RNA。
    映射：A->U, T->A, C->G, G->C（大小写均支持）。reverse_complement=True 时再反向。
    """
    i0 = max(0, start - 1)                 # 1-based -> 0-based
    seg = dna_seq[i0:i0 + length]

    # DNA 到 互补 RNA 的映射
    trans_tbl = str.maketrans({
        "A": "U", "a": "u",
        "T": "A", "t": "a",
        "C": "G", "c": "g",
        "G": "C", "g": "c",
        "N": "N", "n": "n",
    })
    rna_comp = seg.translate(trans_tbl)
    if reverse_complement:
        rna_comp = rna_comp[::-1]
    return rna_comp.upper()

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
    
    OUT_DIR = f'../../results/Crispr/{test_config}/{model_type}/attn_logits'
    # RESULTS_DIR = f'../../results/Crispr/{test_config}/{model_type}/{num_ds}/{version}/'
    head_idx = 0
    kattn = "Kattention2.kattn"
    os.makedirs(f"{OUT_DIR}/{kattn}", exist_ok=True)
    # "ge"(>=), "gt"(>), "le"(<=), "lt"(<)

    export_max_attn_logits_to_csv(
    model=wrapped_model,                 # 或直接传 backbone
    loader=loader,
    DEVICE=device,
    layer_name=kattn,  # 或 "model.Kattention1.kattn"
    head_idx=head_idx,                      # 要分析的 head
    output_csv=f"{OUT_DIR}/{kattn}/{head_idx}.csv"
    )