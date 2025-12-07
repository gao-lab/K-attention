# analysis_kernels.py
import torch
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from typing import Dict, Tuple, Callable
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange

#!/usr/bin/env python

# %%
from einops.layers.torch import Rearrange
from einops import rearrange
import os
import sys
from pathlib import Path
from logging import getLogger, basicConfig
from typing import Literal, Optional
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import get_scheduler, BertConfig

from kattn.kattention import KNET_Crispr_test,CRISPRon,KNET_Crispr_test1,CRISPRon_pt,KNET_Crispr_test2,CRISPRon_base,KNET_Crispr_test3,KNET_Crispr_test4,KNET_Crispr_test6,KNET_Crispr
from kattn.transformers import (
    TransformerCLSModel,
    TransformerAttnModel,
    MHAModel,
    TransformerConfig,

)
from kattn.cnns import CNNModel,CNNModel_1,CNNModel_6,CNNModel_9
from kattn.modules import CNNMixerConfig

from kattn.tokenizers import RNATokenizer, RNAKmerTokenizer
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
import pdb
import pytorch_lightning.callbacks as callbacks
import math
from scipy.stats import spearmanr
import csv

from typing import Literal, Optional
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, ElasticNetCV

basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%m-%d %H:%M:%S")
logger = getLogger()

KATTN_SRC_DIR = Path(os.environ["KATTN_SRC_DIR"])
KATTN_RESOURCES_DIR = Path(os.environ["KATTN_RESOURCES_DIR"])
# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import glob
import pandas as pd
logging.getLogger("kattn.tokenizers").setLevel(logging.ERROR)  # 或 CRITICAL
from sklearn.metrics import make_scorer

class LightningTestModel(L.LightningModule):
    def __init__(
        self,
        model_type: str,  # Literal["rifle", "rnabert", "cnn", "kattn_v3", "kattn_v4", "kattn_lt"],
        test_config: str,
        num_ds: str,
        version: int = 1,
        optimizer_type: Literal["adamw", "sgd"] = "adamw",
        epochs: int = 200,
        weight_decay: float = 0.01,
        max_lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999
    ):
        super().__init__()
        self.auroc_list = []
        self.model_type = model_type
        self.test_config = test_config
        self.num_ds = num_ds
        self.version = version
        self.optimizer_type = optimizer_type
        self.tokenizer = RNATokenizer(
            code_mode="base", T2U=True, special_token_mode="none"
        )
        self.need_amp = False

        if model_type == "cnn3":
            self.model = CNNModel(
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test":
            self.model = KNET_Crispr_test(
                # kernel_size=12,
                number_of_kernel=64,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "CRISPRon":
            self.model = CRISPRon(
                # k_list=(3,5,7),
                # c_list=(64,64,64),
                # fc_collect=256,
                # fc_hidden=128,
                dropout=0.2,
                # seq_len=30,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test1":
            self.model = KNET_Crispr_test(
                # kernel_size=12,
                number_of_kernel=64,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "CRISPRon_pt":
            self.model = CRISPRon_pt(
                k_list=(3,5,7),
                c_list=(64,64,64),
                fc_collect=256,
                fc_hidden=128,
                dropout=0.2,
                seq_len=30,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test2":
            self.model = KNET_Crispr_test2(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test3":
            self.model = KNET_Crispr_test3(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test4":
            self.model = KNET_Crispr_test4(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test6":
            self.model = KNET_Crispr_test6(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr":
            self.model = KNET_Crispr(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "CRISPRon_base":
            self.model = CRISPRon_base(
                dropout=0.2,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "transformer_cls":
            self.need_amp = True
            self.tokenizer = RNATokenizer(
                code_mode="base", T2U=True, special_token_mode="default"
            )
            transformer_config = TransformerConfig(
                vocab_size=len(self.tokenizer),
                position_emb_type="relative_RoPE",
                # attn_method="homemade",
                attn_method="flash-attn",
            )
            # self.need_amp = True
            self.model = TransformerCLSModel(transformer_config,
                regression=True
            )
        elif model_type == "transformer_cls_kmer":
            self.need_amp = True
            self.tokenizer = RNAKmerTokenizer(
                k=5, code_mode="base", T2U=True, special_token_mode="default"
            )
            transformer_config = TransformerConfig(
                num_hidden_layers=1,
                vocab_size=len(self.tokenizer),
                position_emb_type="relative_RoPE",
                # attn_method="homemade",
                attn_method="flash-attn",
            )
            self.model = TransformerCLSModel(transformer_config,regression=True)
        else:
            raise ValueError(f"model_type {model_type} not supported")

        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.max_lr = max_lr

        self.train_preds, self.train_labels = [], []
        self.val_preds,   self.val_labels   = [], []
        self.test_preds,   self.test_labels   = [], []
        self.test_preds, self.test_labels, self.test_losses = [], [], []
        self.lambda_l1 = getattr(self.hparams, "lambda_wattn_l1", 1e-3)
        self.lambda_group = getattr(self.hparams, "lambda_wattn_group", 1e-2)
        self.lambda_ortho = getattr(self.hparams, "lambda_wattn_ortho", 0.0)
    # def forward(self, **kwargs):
    #     if self.need_amp:
    #         with torch.autocast(device_type="cuda", dtype=torch.float16):
    #             return self.model(**kwargs)
    #     else:
    #         return self.model(**kwargs)
    # def forward(
    #     self,
    #     input_ids,
    #     mean_eff,
    #     CRISPRoff_score,         # 注意大小写，需与 batch 的键一致
    #     **kwargs,                 # 兜底不需要的额外字段
    # ):
    #     return self.model(input_ids,mean_eff,CRISPRoff_score)

    def forward(
        self,
        input_ids,
        mean_eff,
        CRISPRoff_score,         # 注意大小写，需与 batch 的键一致
        **kwargs,                 # 兜底不需要的额外字段
    ):
        if self.need_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.model(input_ids,mean_eff,CRISPRoff_score)
        else:
            return self.model(input_ids,mean_eff,CRISPRoff_score)

    def training_step(self, batch, batch_idx):
        ids, me, off = (batch[k] for k in ["input_ids", "mean_eff","CRISPRoff_score"])
        outputs = self(ids, me, off)
        base_loss = outputs["loss"]

        reg_loss, comps = compute_wattn_regularizers(self.model,
                                                lambda_l1=self.lambda_l1,
                                                lambda_group=self.lambda_group,
                                                lambda_ortho=self.lambda_ortho)

        loss = base_loss + reg_loss
        self.log("train_loss", loss, prog_bar=True)

        label = batch.get("me", batch.get("mean_eff"))
        self.train_preds.append(outputs["pred"].detach().float().cpu())
        self.train_labels.append(label.detach().float().cpu())


        return loss

    def on_train_epoch_end(self):
        if len(self.train_preds) == 0: 
            return
        preds  = torch.cat(self.train_preds).numpy()
        labels = torch.cat(self.train_labels).numpy()

        # Spearman（丢弃 NaN）
        rho, _ = spearmanr(preds, labels, nan_policy="omit")
        # 记录到 Logger
        self.log("train_spearman", float(rho), prog_bar=True, on_epoch=True)

        # 保存到 CSV（追加）
        # self._append_spearman_csv(epoch=self.current_epoch, split="train", rho=float(rho))

        # 清空缓存
        self.train_preds.clear()
        self.train_labels.clear()

    def validation_step(self, batch, batch_idx):
        ids, me, off = (batch[k] for k in ["input_ids", "mean_eff","CRISPRoff_score"])
        outputs = self(ids, me, off)
        # outputs = self(**batch)
        base_loss = outputs["loss"]

        reg_loss, comps = compute_wattn_regularizers(self.model,
                                                lambda_l1=self.lambda_l1,
                                                lambda_group=self.lambda_group,
                                                lambda_ortho=self.lambda_ortho)

        loss = base_loss + reg_loss

        label = batch.get("me", batch.get("mean_eff"))
        self.val_preds.append(outputs["pred"].detach().float().cpu())
        self.val_labels.append(label.detach().float().cpu())

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0: 
            return
        preds  = torch.cat(self.val_preds).numpy()
        labels = torch.cat(self.val_labels).numpy()

        rho, _ = spearmanr(preds, labels, nan_policy="omit")
        self.log("val_spearman", float(rho), prog_bar=True, on_epoch=True)
        # self._append_spearman_csv(epoch=self.current_epoch, split="val", rho=float(rho))

        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        """
        与 validation_step 对齐：收集预测与标签，用于整 epoch 计算 Spearman。
        """
        ids, me, off = (batch[k] for k in ["input_ids", "mean_eff", "CRISPRoff_score"])
        outputs = self(ids, me, off)
        base_loss = outputs["loss"]

        reg_loss, comps = compute_wattn_regularizers(self.model,
                                                lambda_l1=self.lambda_l1,
                                                lambda_group=self.lambda_group,
                                                lambda_ortpoho=self.lambda_ortho)

        loss = base_loss + reg_loss

        y = batch.get("me", batch.get("mean_eff")).float()
        loss = F.mse_loss(outputs["pred"], y)

        # 累积到 CPU，epoch 末统一计算
        self.test_preds.append(outputs["pred"].detach().float().cpu())
        self.test_labels.append(y.detach().float().cpu())
        self.test_losses.append(loss.detach().float().cpu())

        # 注意：Lightning 会对同名 metric 在 epoch 维度上聚合
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self):
        """
        整个测试集一次性计算 Spearman（避免逐 batch 求相关再平均的偏差）。
        """
        if len(self.test_preds) == 0:
            return
        preds  = torch.cat(self.test_preds).numpy()
        labels = torch.cat(self.test_labels).numpy()
        mean_loss = torch.stack(self.test_losses).mean().item()
        rho, _ = spearmanr(preds, labels, nan_policy="omit")
        self.log("test_spearman", float(rho), prog_bar=True, on_epoch=True)

        self.test_preds.clear()
        self.test_labels.clear()

        save_dir = f'../../results/Crispr/{self.test_config}/{self.model_type}/{self.num_ds}/{self.version}'
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "test_metrics.csv"), "w", newline="") as f:
            csv.writer(f).writerow(["test_loss", "test_spearman"]); csv.writer(f).writerow([mean_loss, float(rho)])

    def configure_optimizers(self):
        if self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.max_lr,
                                        betas=(self.beta1, self.beta2))
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.max_lr,
                                        weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_scheduler(
                    name="cosine",
                    optimizer=optimizer,
                    num_warmup_steps=10,
                    num_training_steps=self.epochs
                ),
                "interval": "epoch",
            }
        }

def extract_scores(batch):
    offs, means = [], []
    for s in batch["description"]:
        co = math.nan
        me = math.nan
        if isinstance(s, str):
            parts = [p.strip() for p in s.split("|")]
            kv = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k.strip()] = v.strip()
            try:
                co = float(kv.get("CRISPRoff_score", "nan"))
            except ValueError:
                pass
            try:
                me = float(kv.get("mean_eff", "nan"))
            except ValueError:
                pass
        offs.append(co)
        means.append(me)
    return {"CRISPRoff_score": offs, "mean_eff": means}

class TestDataModule(L.LightningDataModule):
    def __init__(self, dst_name, dst_config, dst_dir, tokenizer,
                 val_split: float = 0.1, seed: int = 11,
                 max_seqlen: int = 512, batch_size: int = 128,
                 num_procs: int = 4, num_set: str = 'set0',cache_dir: str | Path = None):
        super().__init__()
        self.dst_name = dst_name
        self.dst_config = dst_config
        self.dst_dir = dst_dir
        self.tokenizer = tokenizer
        self.val_split = val_split
        self.seed = seed
        self.max_seqlen = max_seqlen
        self.batch_size = batch_size
        self.num_procs = num_procs
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(os.getcwd())
        self.num_set = num_set
        # in a closure to get tokenizer
        def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | list]:
            batch = {k: [s[k] for s in batch] for k in batch[0].keys()}
            batch.update(
                self.tokenizer(
                    batch["sequence"], truncation=True, max_length=self.max_seqlen,
                    padding=True, return_special_tokens_mask=False,
                    tensorize=True
                )
            )
            del batch["sequence"]
            batch = self.tokenizer.tensorize(batch)
            return batch

        self.collate_fn = collate_fn

    def prepare_data(self):
        if os.path.exists(os.path.join(self.cache_dir, self.dst_config, self.num_set)):
            logger.info(f"Found tokenized dataset in {os.path.join(self.cache_dir, self.dst_config, self.num_set)}")
            return

        # preprocess only on master rank
        dataset1 = load_dataset(
            str(self.dst_name), self.dst_config, split=f"{self.num_set}_train_crispr_on",
            trust_remote_code=True, data_dir=str(self.dst_dir),
            num_proc=self.num_procs
        )
        dataset1 = dataset1.map(
            extract_scores,
            batched=True,
            remove_columns=["description"]  # 同时可以加上 name/sequence 等不需要的列
        )

        dataset2 = load_dataset(
            str(self.dst_name), self.dst_config, split=f"{self.num_set}_valid_crispr_on",
            trust_remote_code=True, data_dir=str(self.dst_dir),
            num_proc=self.num_procs
        )
        dataset2 = dataset2.map(
            extract_scores,
            batched=True,
            remove_columns=["description"]  # 同时可以加上 name/sequence 等不需要的列
        )

        dataset3 = load_dataset(
            str(self.dst_name), self.dst_config, split=f"{self.num_set}_test_crispr_on",
            trust_remote_code=True, data_dir=str(self.dst_dir),
            num_proc=self.num_procs
        )
        dataset3 = dataset3.map(
            extract_scores,
            batched=True,
            remove_columns=["description"]  # 同时可以加上 name/sequence 等不需要的列
        )

        datasets = DatasetDict({
                "train": dataset1,
                "valid": dataset2,
                "test": dataset3,
        })
        datasets.save_to_disk(str(self.cache_dir / self.dst_config / self.num_set))

    def setup(self, stage):
        tokenized_datasets = load_from_disk(str(self.cache_dir / self.dst_config / self.num_set))
        self.dsts = tokenized_datasets
        
    def train_dataloader(self):
        return DataLoader(
            self.dsts["train"],
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_procs,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dsts["valid"],
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_procs,
            pin_memory=True,
            shuffle=False
        )
    def test_dataloader(self):
        return DataLoader(
            self.dsts["test"],
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_procs,
            pin_memory=True,
            shuffle=False
        )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- helpers -----------------

# ---------- 1) 列出所有 self_kattn 模块 ----------
def list_self_kattn_modules(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """
    返回 {模块名: 模块实例}，筛选出所有 self_kattn 实例。
    同时兼容某些环境中类名字符串判断；也可以换成 isinstance(m, self_kattn)。
    """
    modules = {}
    for n, m in model.named_modules():
        if m.__class__.__name__ == "self_kattn":
            modules[n] = m
        else:
            # 兜底：具备 self.kattn 且其类名为 KattentionV4，也视作目标模块
            if hasattr(m, "kattn") and getattr(m.kattn, "__class__", type(None)).__name__ == "KattentionV4":
                if hasattr(m, "v") and hasattr(m, "k_n") and hasattr(m, "h_d") and hasattr(m, "k_l"):
                    modules[n] = m
    return modules

def get_label_from_batch(batch):
    if "mean_eff" in batch:
        y = batch["mean_eff"]
    elif "me" in batch:
        y = batch["me"]
    else:
        raise KeyError("Batch must contain 'mean_eff' or 'me'.")
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    return y.float().view(-1)

# ---------- 2) 收集 bhqd ----------
@torch.no_grad()
def collect_bhqd_for_all_layers(model: torch.nn.Module,
                                loader,
                                max_batches: int | None = None,
                                model_type: str = 'KNET_Crispr_test6'
                                ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    对每个 self_kattn 模块，复现 forward 里的 v/pos/attn 逻辑，输出 bhqd：
      - 返回 outs: {layer_name: Tensor[N, H, Q, D]}
      - 返回 labels: Tensor[N]
    与 self_kattn 保持一致：
      * v = mod.v(x.transpose(1,2))，若有位置则 v += pos.transpose(1,2)
      * attn_logits 若 reverse 则在最后一维 flip
      * softmax 缩放：/ sqrt(h_d)
      * einsum: "bhqk,bhkd->bhqd"
    """
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    modules = list_self_kattn_modules(model)

    outs = {name: [] for name in modules}
    labels = []
    # —— 准备 tokenizer 类似逻辑（与你现有训练管线一致）——
    # 若外部已有 RNATokenizer，可直接用；这里假设 vocab_size=10，与模型一致。
    vocab_size = 10

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        # ids: [B, L] -> one-hot -> [B, L, vocab_size] -> 取最后4通道
        ids_1d = batch["input_ids"].to(device)
        y = get_label_from_batch(batch).to(device)

        ids = F.one_hot(ids_1d, num_classes=vocab_size).float()[:, :, -4:]   # [B, L, 4]
        B, Len, C = ids.shape

        # 针对每个 self_kattn 模块，逐一复算 v_hd、attn，并得到 out_bhqd
        for name, mod in modules.items():
            H = int(mod.k_n)
            D = int(mod.h_d)

            # 1) v: Conv1d 接受 (B, C, L)
            v = mod.v(ids.transpose(1, 2))              # [B, H*D, L]

            # 2) 位置编码（注意：在 self_kattn 中是加在 v 上的）
            if hasattr(mod, "position_embeddings") and mod.position_embeddings is not None:
                position_ids = torch.arange(Len, dtype=torch.long, device=ids.device).unsqueeze(0).expand(B, Len)
                pos = mod.position_embeddings(position_ids)     # [B, L, H*D] （你的实现里 emb_dim=self.hidden_dim）
                v = v + pos.transpose(1, 2)                     # [B, H*D, L]

            # 3) v -> v_hd: [B, H, L, D]
            #    （若 hidden_dim 非 H*D，会在这里报错，确保 self.hidden_dim == k_n*h_d）
            v_hd = rearrange(v, "b (h d) l -> b h l d", h=H)

            # 4) 计算 attn_logits
            pad = int((mod.k_l - 1) / 2)
            x_ch_first = ids.transpose(1, 2)                               # [B, C, L]
            x_padded = F.pad(x_ch_first, (pad, pad), mode="constant", value=0.25)
            A_logits = mod.kattn(x_padded.transpose(1, 2))["attn_logits"]  # [B,H,L,L] 或 [B,L,L]

            # 4.1) reverse 处理（与 forward 一致）
            if getattr(mod, "reverse", False):
                A_logits = A_logits.flip([-1])

            # 4.2) 单头 -> 多头扩展
            if A_logits.dim() == 3:  # [B,L,L]
                A_logits = A_logits.unsqueeze(1).expand(B, H, Len, Len)     # [B,H,L,L]

            # 4.3) 缩放 + softmax
            A_logits = A_logits / math.sqrt(float(D))
            attn = F.softmax(A_logits, dim=-1)                              # [B,H,L,L]，q=L,k=L

            # 5) out: einsum "bhqk,bhkd->bhqd" => [B,H,L,D]
            out_bhqd = torch.einsum("bhqk,bhkd->bhqd", attn, v_hd)          # [B,H,Q,D]，Q=L
            outs[name].append(out_bhqd.detach().cpu())

        labels.append(y.detach().cpu())

    # 拼接
    outs_cat = {n: torch.cat(v_list, dim=0) for n, v_list in outs.items()}  # 各层 [N,H,Q,D]
    labels_cat = torch.cat(labels, dim=0)  # [N]
    return outs_cat, labels_cat

# ----------------- 1) compute per-head correlation -----------------
def per_head_mean_corr(outs_bhqd: torch.Tensor, labels: torch.Tensor, method="pearson"):
    """
    outs_bhqd: Tensor [N, H, Q, D]
    labels: Tensor [N]
    Returns:
      mean_corrs: np.array shape [H] — mean(abs(corr)) across all (q,d)
      raw_corrs: np.array shape [H, Q, D] — corr values per (q,d)
    """
    N, H, Q, D = outs_bhqd.shape
    y_np = labels.cpu().numpy()
    mean_corrs = np.zeros(H, dtype=float)
    raw = np.zeros((H, Q, D), dtype=float)

    for h in range(H):
        # for speed, vectorize: reshape (N, Q*D)
        mat = outs_bhqd[:, h, :, :].reshape(N, Q * D).cpu().numpy()  # [N, Q*D]
        # if a column is constant, pearsonr undefined; handle
        corr_vals = []
        for col in range(mat.shape[1]):
            x = mat[:, col]
            if np.std(x) == 0:
                corr = 0.0
            else:
                if method == "pearson":
                    corr, _ = stats.pearsonr(x, y_np)
                else:
                    corr, _ = stats.spearmanr(x, y_np)
            corr_vals.append(corr)
            # write to raw array
            q = col // D
            d = col % D
            raw[h, q, d] = corr
        # take mean of absolute correlations (or mean of signed corr if preferred)
        mean_corrs[h] = np.mean(np.abs(np.array(corr_vals)))
    return mean_corrs, raw

def _pearson_r(y_true, y_pred) -> float:
    # 常数方差的边界情况：返回 0（无法定义相关）
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.std() == 0 or yp.std() == 0:
        return 0.0
    # np.corrcoef 返回 2x2 协方差矩阵，取 [0,1]
    return float(np.corrcoef(yt, yp)[0, 1])

_PEARSON_SCORER = make_scorer(_pearson_r, greater_is_better=True)

def per_head_linear_probe_r(
    bhqd: torch.Tensor,   # [N,H,Q,D]
    y: torch.Tensor,      # [N]
    method: Literal["ridge","elasticnet"] = "ridge",
    cv: int = 5,
    alphas: Optional[np.ndarray] = None,
    l1_ratios: Optional[list[float]] = None,
    standardize: bool = True,
    plot: bool = True,
    save_path: Optional[str] = None,
    show: bool = False,  # 在服务器默认不展示，只保存
    dpi: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个 head，使用线性 probe（Ridge/ElasticNet），以 CV 平均 Pearson R 评估。
    返回:
        r_scores: [H] 每个 head 的平均 R
        order:    [H] R 从大到小的 head 索引
    若 plot=True 则绘制条形图；若给出 save_path 会自动创建目录并保存。
    """
    assert bhqd.dim() == 4, f"bhqd should be [N,H,Q,D], got {tuple(bhqd.shape)}"
    N, H, Q, D = bhqd.shape

    X_all = bhqd.detach().cpu().numpy().reshape(N, H, Q * D).astype(np.float32)  # [N,H,features]
    y_np  = y.detach().cpu().numpy().astype(np.float32)

    if alphas is None:
        alphas = np.logspace(-4, 3, 20)
    if l1_ratios is None:
        l1_ratios = [0.5, 0.7, 0.9]

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r_scores = np.zeros(H, dtype=np.float32)

    for h in range(H):
        Xh = X_all[:, h, :]  # [N, Q*D]

        if method == "ridge":
            base = RidgeCV(alphas=alphas, cv=cv)
        else:  # "elasticnet"
            base = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, n_jobs=-1)

        model = make_pipeline(StandardScaler(with_mean=True, with_std=True), base) if standardize else base
        r_cv = cross_val_score(model, Xh, y_np, scoring=_PEARSON_SCORER, cv=kf, n_jobs=-1)
        r_scores[h] = float(np.mean(r_cv))

    # 降序排序
    order = np.argsort(-r_scores)

    if plot:
        fig = plt.figure()
        plt.bar(range(H), r_scores[order])
        plt.xticks(range(H), [f"h{idx}" for idx in order], rotation=45)
        plt.ylabel("CV mean Pearson R")
        plt.title("Per-head linear probe (sorted by R)")
        plt.tight_layout()

        if save_path is not None:
            dir_ = os.path.dirname(save_path)
            if dir_ and not os.path.exists(dir_):
                os.makedirs(dir_, exist_ok=True)  # 关键：自动创建多级目录
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)

    return r_scores, order

# ----------------- 2) intervention: set attn to uniform or diag -----------------
def make_attn_modifier(mode: str, target_head: int):
    """
    returns a callable modify(attn_logits) -> attn_logits_modified
    mode: "uniform" or "diag"
    target_head: index h to modify (0-based)
    """
    assert mode in ("uniform", "diag")
    def modifier(attn_logits: torch.Tensor):
        # attn_logits: [B, H, Q, K]
        B, H, Q, K = attn_logits.shape
        new = attn_logits.clone()
        if mode == "uniform":
            # set logits=0 for head, so softmax -> uniform
            new[:, target_head, :, :] = 0.0
        else:  # diag
            # set very negative everywhere then 0 on diagonal
            huge_neg = -1e9
            mask = torch.full((B, Q, K), huge_neg, device=attn_logits.device, dtype=attn_logits.dtype)
            # ensure square: if Q != K slice min
            Lmin = min(Q, K)
            # set diag positions to 0
            idx = torch.arange(Lmin, device=attn_logits.device)
            mask[:, idx, idx] = 0.0
            # if Q!=K, we broadcast mask appropriately; but we require Q==K normally
            new[:, target_head, :mask.shape[1], :mask.shape[2]] = mask
        return new
    return modifier

def eval_model_with_attn_modification(model: torch.nn.Module, loader, modifier_factory: Callable, target_head: int, 
                                      use_loss=True, max_batches: int | None = None, model_type: str='KNET_Crispr_test4'):
    """
    Temporarily wrap KattentionV4 inside kattn_with_V by monkeypatching its forward:
      - call original forward to get attn_logits
      - replace attn_logits with modified one for target_head via modifier_factory
    Compute average MSE on loader.
    """
    model.eval().to(DEVICE)
    # collect all kattn_with_V modules
    modules = list_self_kattn_modules(model)

    # save originals
    originals = {}
    for name, mod in modules.items():
        originals[name] = mod.kattn.forward

    # define wrapper
    def make_wrapper(orig_forward, modifier):
        def wrapped(x, *args, **kwargs):
            out = orig_forward(x, *args, **kwargs)
            logits = out["attn_logits"]
            logits_mod = modifier(logits)
            out["attn_logits"] = logits_mod
            return out
        return wrapped

    # replace forward for all kattn_with_V.kattn
    modifier = modifier_factory(target_head)
    for name, mod in modules.items():
        mod.kattn.forward = make_wrapper(originals[name], modifier)

    # eval loop
    total_loss = 0.0
    total_n = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        ids = batch["input_ids"].to(DEVICE)
        y = get_label_from_batch(batch).to(DEVICE)
        with torch.no_grad():
            out = model(ids, y, batch.get("CRISPRoff_score", None))
            if use_loss:
                total_loss += out["loss"].item() * ids.size(0)
            else:
                # compute mse manually from pred
                pred = out["pred"]
                total_loss += F.mse_loss(pred, y, reduction="sum").item()
            total_n += ids.size(0)

    # restore original forwards
    for name, mod in modules.items():
        mod.kattn.forward = originals[name]

    return total_loss / max(total_n, 1)

# ----------------- 3) extract kernel and plot -----------------
def extract_kernel_for_head(kattn_module, head_idx: int) -> np.ndarray:
    """
    Try to extract the Wattn.weight and reshape to a visualizable kernel for head head_idx.
    This function is "best-effort" — Wattn weight shapes may vary; we try to map head -> kernel block.
    Returns: numpy array shape (c1, something) or (k, c1) depending on shape.
    """
    W = kattn_module.kattn.Wattn.weight.detach().cpu()  # shape depends on impl
    # Try common shape: [out_c, in_c, 1] or [out_c, in_c, h]
    Wt = W
    if Wt.ndim == 4 and Wt.shape[-1] == 1:
        Wt = Wt.squeeze(-1)  # [out_c, in_c, kh]
    if Wt.ndim == 3:
        out_c, in_c, hW = Wt.shape
        # According to earlier design out_c = k * h * c1; and head_idx corresponds to h index (num_kernels)
        # We'll try to split out_c into (k, h, c1) where h = num_kernels
        k = kattn_module.kattn.kernel_size
        num_kernels = kattn_module.kattn.num_kernels
        c1 = kattn_module.kattn.channel_size
        expected_out = k * num_kernels * c1
        if out_c == expected_out:
            # rearrange to [k, h, c1, in_c? , hW] but simplest: take block for head_idx
            W_resh = rearrange(Wt, "(k h c1) c2 hW -> k h c1 c2 hW", k=k, h=num_kernels, c1=c1)
            # take head slice: shape [k, c1, c2, hW]
            block = W_resh[:, head_idx, :, :, :]  # [k, c1, c2, hW]
            # if c2==1 and hW==1, simplify to [k, c1]
            block = block.squeeze(-1)
            # collapse c2 if >1 by mean
            if block.ndim == 3:
                # [k, c1, c2] -> average over c2
                block = block.mean(axis=-1)
            # final: [k, c1] — rows are k positions, cols are input channels
            return block.numpy()
        else:
            # fallback: take chunk of out_c corresponding to head (rough)
            # each head block size:
            block_size = out_c // num_kernels
            start = head_idx * block_size
            block = Wt[start:start+block_size, :, :]  # [block_size, in_c, hW]
            # reduce last dim
            block = block.mean(axis=-1)  # [block_size, in_c]
            return block.numpy()
    else:
        # 2D fallback
        return Wt.numpy()

def plot_kernel_matrix(mat: np.ndarray, savepath: str | Path, title: str = ""):
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(mat.shape[1], mat.shape[0]))
    sns.heatmap(mat, annot=False, cmap="vlag")
    plt.title(title)
    plt.xlabel("input-channel")
    plt.ylabel("kernel-position")
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def _get_wattn_weight(kattn_module) -> torch.Tensor:
    Wattn = getattr(kattn_module, "Wattn")
    W = Wattn.weight.detach().cpu()
    if W.ndim == 4:
        o, i, kh, kw = W.shape
        if kh == 1 and kw == 1:
            W = W.view(o, i, 1)
        elif kh == 1 and kw > 1:
            W = W.squeeze(-2)      # [o,i,kw]
        elif kw == 1 and kh > 1:
            W = W.squeeze(-1)      # [o,i,kh]
        else:
            W = W.reshape(o, i, kh * kw)
    elif W.ndim == 3:
        pass                       # [o,i,h]
    elif W.ndim == 2:
        o, i = W.shape
        W = W.view(o, i, 1)
    else:
        raise ValueError(f"Unexpected Wattn.weight ndim={W.ndim}")
    return W  # [out_c, in_c, h]

def reshape_to_k_c1_c2_h(kattn_module, W: torch.Tensor) -> np.ndarray:
    assert W.ndim == 3, f"Expect [out_c,in_c,hW], got {tuple(W.shape)}"
    out_c, in_c, hW = W.shape
    k   = int(getattr(kattn_module, "kernel_size"))
    c1  = int(getattr(kattn_module, "channel_size"))
    c2  = c1
    h   = int(getattr(kattn_module, "num_kernels"))
    expected_out_c = k * h * c1
    if (out_c != expected_out_c) or (in_c != c2):
        print(f"[fallback] out_c={out_c} vs expected={expected_out_c}, in_c={in_c} vs c2={c2}.")
        if hW != h:
            if hW > h:
                W = W[..., :h]
            else:
                pad = torch.zeros(out_c, in_c, h - hW, dtype=W.dtype, device=W.device)
                W = torch.cat([W, pad], dim=-1)
        kernels = rearrange(W, "o i h1 -> o i 1 h1")
        return kernels.cpu().numpy()
    if hW == h:
        kernels = rearrange(W, "(k h c1) c2 h -> k c1 c2 h", k=k, h=h, c1=c1)
        return kernels.cpu().numpy()
    W2 = W.squeeze(-1)  # [out_c,in_c]
    kernels = rearrange(W2, "(k h c1) c2 -> k c1 c2 h", k=k, h=h, c1=c1)
    return kernels.cpu().numpy()

def find_modules_by_classname(model: torch.nn.Module, class_name: str) -> dict[str, torch.nn.Module]:
    return {n: m for n, m in model.named_modules() if m.__class__.__name__ == class_name}

# def draw_topk_kernels(
#     model: torch.nn.Module,
#     scores: dict[str, torch.Tensor],   # {layer_name: Tensor[H]}  每个 head 的打分（如 CV R² 或 ΔMSE 等）
#     topn: int = 5,
#     save_dir: str | Path = "./kernel_figs",
#     cmap: str | None = None,
# ):
#     """
#     对每个 KattentionV4 层：
#       1) 读取 Wattn 权重 -> reshape 到 [k, c1, c2, h]
#       2) 按 scores[lname]（长度 H）排序，取 Top-N 的 head 索引
#       3) 对每个 head 横向绘制 k 张 c1×c2 的热力图

#     导出：
#       - <save_dir>/<layer>_head_rank.csv   （head 与分数排序）
#       - <save_dir>/<layer>-h{hid}-rank{r}.png  （单个 head 的合成图）
#     """
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     cmap = cmap or sns.color_palette("vlag", as_cmap=True)

#     # 拿到所有 KattentionV4
#     kattn_layers = find_modules_by_classname(model, "KattentionV4")
#     if not kattn_layers:
#         raise RuntimeError("No KattentionV4 modules found in model.")

#     summary_rows = []

#     for lname, kattn in kattn_layers.items():
#         if lname not in scores:
#             print(f"[warn] scores missing for layer '{lname}', skip.")
#             continue

#         # 取权重并 reshape 到 [k, c1, c2, h]
#         W = _get_wattn_weight(kattn)                  # [out_c, in_c, hW]
#         kernels = reshape_to_k_c1_c2_h(kattn, W)      # [k, c1, c2, h]
#         K, C1, C2, H = kernels.shape

#         # head 分数
#         score_t = scores[lname]
#         score_np = score_t.detach().cpu().numpy().reshape(-1)
#         if score_np.shape[0] != H:
#             print(f"[warn] score length ({score_np.shape[0]}) != num_kernels H ({H}) for layer '{lname}'. Truncate/pad.")
#             # 与 H 对齐（截断或 0 填充）
#             if score_np.shape[0] > H:
#                 score_np = score_np[:H]
#             else:
#                 pad = np.zeros(H - score_np.shape[0], dtype=score_np.dtype)
#                 score_np = np.concatenate([score_np, pad], axis=0)

#         # 排序（按 head 分数）
#         order = np.argsort(-score_np)[: min(topn, H)]

#         # 导出 head 排序 CSV
#         df_rank = pd.DataFrame({
#             "head": np.arange(H, dtype=int),
#             "score": score_np,
#         }).sort_values("score", ascending=False).reset_index(drop=True)
#         df_rank.to_csv(save_dir / f"{lname.replace('.', '_')}_head_rank.csv", index=False)

#         # === 新增：根据 C1/C2 自动设置 y/x 轴刻度标签 ===
#         xticklabels = ["A", "C", "G", "U"] if C2 == 4 else [str(i) for i in range(C2)]
#         yticklabels = ["A", "C", "G", "U"] if C1 == 4 else [str(i) for i in range(C1)]

#         # 逐 head 绘图：横向 k 张子图（每张是 c1×c2 的热力图）
#         for r, h_idx in enumerate(order, start=1):
#             kernel_len = K  # 为了与需求中的变量名一致
#             # === 修改：调整图像大小与 squeeze=False（axes 始终二维） ===
#             fig, axes = plt.subplots(
#                 1, kernel_len,
#                 figsize=(max(kernel_len, 1) * 2.0, 2.6),
#                 dpi=160,
#                 squeeze=False
#             )
#             ax_row = axes[0]

#             for kpos in range(kernel_len):
#                 mat = kernels[kpos, :, :, h_idx]  # [c1, c2]
#                 ax = ax_row[kpos if kernel_len > 1 else 0]
#                 sns.heatmap(
#                     mat, ax=ax,
#                     cbar=(kpos == kernel_len - 1),
#                     cmap=cmap,
#                     annot=True,
#                     xticklabels=xticklabels,
#                     yticklabels=yticklabels,
#                 )
#                 ax.set_title(f"k={kpos}", fontsize=9)
#                 # 美化刻度：字号稍小、去掉多余刻度线
#                 ax.tick_params(axis="both", labelsize=8, length=0)

#             fig.suptitle(
#                 f"{lname}\nhead={h_idx} (rank {r})  score={score_np[h_idx]:.4f}",
#                 fontsize=10
#             )
#             out_png = save_dir / f"{lname.replace('.', '_')}-h{int(h_idx):03d}-rank{r:02d}.png"
#             fig.tight_layout()
#             fig.savefig(out_png.as_posix(), bbox_inches="tight")
#             plt.close(fig)

#             summary_rows.append({
#                 "layer": lname,
#                 "head": int(h_idx),
#                 "rank": int(r),
#                 "score": float(score_np[h_idx]),
#                 "k": int(K),
#                 "c1": int(C1),
#                 "c2": int(C2)
#             })

#     pd.DataFrame(summary_rows).to_csv(save_dir / "drawn_topN_summary.csv", index=False)
#     print(f"[draw] figures saved under: {save_dir.resolve()}")

def draw_topk_kernels( 
    model: torch.nn.Module,
    scores: dict[str, torch.Tensor],   # {layer_name: Tensor[H]}
    topn: int = 5,
    save_dir: str | Path = "./kernel_figs",
    cmap: str | None = None,
):
    """
    对每个 KattentionV4 层：
      1) 读取 Wattn 权重 -> reshape 到 [K, C1, C2, H]
      2) 先对每个 (k, h) 的 C1×C2 小核做 L2/Fro 归一化
      3) 按 scores[lname] 排序取 Top-N head
      4) 绘制热图（右侧 colorbar 无刻度；annot 保留两位小数）
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cmap = cmap or sns.color_palette("bwr", as_cmap=True)

    kattn_layers = find_modules_by_classname(model, "KattentionV4")
    if not kattn_layers:
        raise RuntimeError("No KattentionV4 modules found in model.")

    summary_rows = []

    for lname, kattn in kattn_layers.items():
        if lname not in scores:
            print(f"[warn] scores missing for layer '{lname}', skip.")
            continue

        # 取权重并 reshape 到 [K, C1, C2, H]
        W = _get_wattn_weight(kattn)                  # 可能是 torch.Tensor 或 np.ndarray
        kernels = reshape_to_k_c1_c2_h(kattn, W)      # [K, C1, C2, H]
        K, C1, C2, H = kernels.shape

        # # -------- 归一化（兼容 torch / numpy）---------
        # if isinstance(kernels, torch.Tensor):
        #     # 保持在 CPU 上方便后续与 numpy 混用；也可留在原 device
        #     k_flat = kernels.reshape(K, C1 * C2, H)
        #     norms = torch.norm(k_flat, p=2, dim=1, keepdim=True).clamp_min(1e-8)  # (K,1,H)
        #     kernels = (k_flat / norms).reshape(K, C1, C2, H)
        #     # 绘图时转 numpy
        #     kernels_np = kernels.detach().cpu().numpy()
        # else:
        #     # numpy 分支
        #     k_flat = kernels.reshape(K, C1 * C2, H)          # (K, C1*C2, H)
        #     norms = np.linalg.norm(k_flat, ord=2, axis=1, keepdims=True)
        #     norms = np.clip(norms, 1e-8, None)
        #     kernels_np = (k_flat / norms).reshape(K, C1, C2, H)

        # head 分数对齐
        score_t = scores[lname]
        score_np = score_t.detach().cpu().numpy().reshape(-1)
        if score_np.shape[0] != H:
            print(f"[warn] score length ({score_np.shape[0]}) != num_kernels H ({H}) for layer '{lname}'. Truncate/pad.")
            score_np = score_np[:H] if score_np.shape[0] > H else np.pad(score_np, (0, H - score_np.shape[0]))

        order = np.argsort(-score_np)[: min(topn, H)]

        # 导出 head 排序 CSV
        pd.DataFrame({"head": np.arange(H, dtype=int), "score": score_np}) \
          .sort_values("score", ascending=False) \
          .reset_index(drop=True) \
          .to_csv(save_dir / f"{lname.replace('.', '_')}_head_rank.csv", index=False)

        # 轴刻度标签
        xticklabels = ["A", "C", "G", "U"] if C2 == 4 else [str(i) for i in range(C2)]
        yticklabels = ["A", "C", "G", "U"] if C1 == 4 else [str(i) for i in range(C1)]

        # 逐 head 绘图
        for r, h_idx in enumerate(order, start=1):
            kernel_len = K
            fig, axes = plt.subplots(
                1, kernel_len,
                figsize=(max(kernel_len, 1) * 2.0, 2.0),
                dpi=160,
                squeeze=False
            )
            ax_row = axes[0]

            for kpos in range(kernel_len):
                mat = kernels[kpos, :, :, h_idx]  # numpy [C1, C2]
                ax = ax_row[kpos if kernel_len > 1 else 0]
                # 仅最后一幅显示 colorbar，但不显示刻度
                show_cbar = (kpos == kernel_len - 1)
                hm = sns.heatmap(
                    mat, ax=ax,
                    cbar=False,
                    # cbar_kws={"ticks": []} if show_cbar else None,
                    cmap=cmap,
                    annot=True, fmt=".2f",   # ← annot 两位小数
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    center=0.0,
                )
                if show_cbar and hm.collections and hm.collections[0].colorbar is not None:
                    cb = hm.collections[0].colorbar
                    cb.set_ticks([])
                    cb.set_ticklabels([])

                ax.set_title(f"k={kpos}", fontsize=9)
                ax.tick_params(axis="both", labelsize=8, length=0)

            # fig.suptitle(
            #     f"{lname}\nhead={h_idx} (rank {r})  score={score_np[h_idx]:.4f}",
            #     fontsize=10
            # )
            out_png = save_dir / f"{lname.replace('.', '_')}-h{int(h_idx):03d}-rank{r:02d}.svg"
            fig.tight_layout()
            fig.savefig(out_png.as_posix(), bbox_inches="tight",dpi=300,format="svg")
            plt.close(fig)

            summary_rows.append({
                "layer": lname,
                "head": int(h_idx),
                "rank": int(r),
                "score": float(score_np[h_idx]),
                "k": int(K),
                "c1": int(C1),
                "c2": int(C2)
            })

    pd.DataFrame(summary_rows).to_csv(save_dir / "drawn_topN_summary.csv", index=False)
    print(f"[draw] figures saved under: {save_dir.resolve()}")

# ----------------- high-level orchestration -----------------
def run_full_analysis(wrapped_model, loader, out_dir="./kernel_analysis", max_batches=None, model_type='KNET_Crispr_test3'):
    """
    1) compute per-head mean correlation
    2) compute baseline MSE, then for each head compute MSE with uniform and diag modifications -> ΔMSE
    3) plot kernels for top heads
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wrapped_model.eval().to(DEVICE)

    # collect bhqd outputs and labels
    print("[1/3] collecting bhqd outputs ...")
    outs, labels = collect_bhqd_for_all_layers(wrapped_model, loader, max_batches=max_batches,model_type=model_type)
    # iterate layers
    results = []
    for lname, bhqd in outs.items():
        print(f"Processing layer {lname} ...")
        # bhqd: [N,H,Q,D]
        # mean_corrs, raw_corrs = per_head_mean_corr(bhqd, labels, method="pearson")
        # # sort by mean_corrs descending
        # order_corr = np.argsort(-mean_corrs)
        # r2_scores = per_head_linear_probe_r2(
        #     bhqd=bhqd,         # [N,H,Q,D]
        #     y=labels,          # [N]
        #     method="ridge",    # or "elasticnet"
        #     cv=5,
        #     alphas=None,
        #     l1_ratios=None,
        #     standardize=True,
        # )
        r_scores, _ = per_head_linear_probe_r(
            bhqd, labels, method="ridge", cv=5, plot=True, save_path=f"{out_dir}/{lname.replace(".","_")}/per_head_R_barplot.png"
        )
        order_r = np.argsort(-r_scores)

        import pandas as pd
        layer_out = Path(out_dir) / lname.replace(".", "_")
        layer_out.mkdir(parents=True, exist_ok=True)

        df_head_r = pd.DataFrame({
            "head": np.arange(len(order_r)),
            "cv_r": order_r,
        }).sort_values("cv_r", ascending=False).reset_index(drop=True)
        df_head_r.to_csv(layer_out / "head_rank_by_cv_r.csv", index=False)

        # baseline mse
        print("Computing baseline mse ...")
        baseline_mse = eval_model_baseline_mse(wrapped_model, loader, max_batches=max_batches)
        print(f"Baseline MSE: {baseline_mse:.6f}")

        # for each head compute ΔMSE for uniform and diag
        H = bhqd.shape[1]
        delta_uniform = np.zeros(H, dtype=float)
        delta_diag = np.zeros(H, dtype=float)
        for h in range(H):
            print(f"  Evaluating head {h} intervention ...")
            mse_uniform = eval_model_with_attn_modification(wrapped_model, loader, 
                                                           modifier_factory=lambda hh: make_attn_modifier("uniform", hh),
                                                           target_head=h, use_loss=True, max_batches=max_batches,model_type=model_type)
            mse_diag = eval_model_with_attn_modification(wrapped_model, loader, 
                                                         modifier_factory=lambda hh: make_attn_modifier("diag", hh),
                                                         target_head=h, use_loss=True, max_batches=max_batches,model_type=model_type)
            delta_uniform[h] = mse_uniform - baseline_mse
            delta_diag[h] = mse_diag - baseline_mse

        # gather and save per-layer info
        layer_out = Path(out_dir) / lname.replace(".", "_")
        layer_out.mkdir(parents=True, exist_ok=True)

        # # save correlation ranking
        # import pandas as pd
        # df_corr = pd.DataFrame({
        #     "head": np.arange(len(mean_corrs)),
        #     "mean_abs_pearson": mean_corrs,
        # }).sort_values("mean_abs_pearson", ascending=False).reset_index(drop=True)
        # df_corr.to_csv(layer_out / "head_corr_rank.csv", index=False)

        # save delta mse ranking
        df_delta = pd.DataFrame({
            "head": np.arange(H),
            "delta_mse_uniform": delta_uniform,
            "delta_mse_diag": delta_diag
        }).sort_values("delta_mse_uniform", ascending=False).reset_index(drop=True)
        df_delta.to_csv(layer_out / "head_delta_mse.csv", index=False)

        # plot top5 by corr and top5 by delta_mse_uniform
        topk = 48
        top_corr = df_head_r.head(topk)["head"].values
        top_delta = df_delta.sort_values("delta_mse_uniform", ascending=False).head(topk)["head"].values
        top_union = sorted(set(top_corr.tolist() + top_delta.tolist()))

        # for h in top_union:
            # find module object
            # mod = list_kattn_with_V_modules(wrapped_model)[lname]
            # kern = extract_kernel_for_head(mod, int(h))
            # save_png = layer_out / f"head{h}_kernel.png"
            # plot_kernel_matrix(kern, save_png, title=f"{lname} head {h}")
        scores_for_layer = {f"{lname}.kattn": torch.tensor(r_scores)}  # r2_scores: np.ndarray[H]

        # 指定该层的输出目录作为 save_dir（也可以给一个总目录）
        draw_topk_kernels(
            model=wrapped_model,
            scores=scores_for_layer,
            topn=topk,                               # 与你的 topk 保持一致
            save_dir=layer_out,                      # 每层各自目录
            # cmap=sns.light_palette("#2ecc71", as_cmap=True),
        )

        results.append({
            "layer": lname,
            "df_head_r": df_head_r,
            "df_delta": df_delta,
            "top_union": top_union
        })

    print("ALL DONE. Outputs written under:", Path(out_dir).resolve())
    return results

# small helper to compute baseline mse
def eval_model_baseline_mse(model, loader, max_batches: int | None = None):
    model.eval().to(DEVICE)
    total_loss = 0.0
    total_n = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        ids = batch["input_ids"].to(DEVICE)
        y = get_label_from_batch(batch).to(DEVICE)
        with torch.no_grad():
            out = model(ids, y, batch.get("CRISPRoff_score", None))
            total_loss += out["loss"].item() * ids.size(0)
            total_n += ids.size(0)
    return total_loss / max(total_n, 1)

# 从目录里找最佳 ckpt（你用的 ModelCheckpoint 命名规则）
def find_best_ckpt(results_dir: str) -> str:
    # 你训练时也可以用 ckpt_path="best"，这里我们找真实文件
    cands = glob.glob(os.path.join(results_dir, "best-*.ckpt"))
    if not cands:
        raise FileNotFoundError(f"No ckpt found under {results_dir}")
    # 若只有一个，直接用；若多个，取 val_loss 最小（从文件名解析）
    def parse_val_loss(p):
        # 文件名样例: best-epoch=148-val_loss=0.050-version1-lr0.0001.ckpt
        base = os.path.basename(p)
        try:
            s = base.split("-val_loss=")[1]
            val = float(s.split("-")[0])
        except Exception:
            val = math.inf
        return val
    return sorted(cands, key=parse_val_loss)[0]

# ----------------- USAGE -----------------
if __name__ == "__main__":
    # assume you have wrapped_model and loader in scope; e.g. load checkpoint etc.
    # Example:
    # wrapped_model = LightningTestModel.load_from_checkpoint("best.ckpt").to(DEVICE).eval()
    # data_module.setup("test"); loader = data_module.test_dataloader()
    # Then:
    # results = run_full_analysis(wrapped_model, loader, out_dir="./kernel_analysis", max_batches=None)
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

    data_module.setup(stage="test")  # 或者 "validate"，看你想用哪个 split
    loader = data_module.train_dataloader()  # 若用验证集: data_module.val_dataloader()
    results = run_full_analysis(wrapped_model, loader, out_dir=OUT_DIR, max_batches=None, model_type=model_type)
