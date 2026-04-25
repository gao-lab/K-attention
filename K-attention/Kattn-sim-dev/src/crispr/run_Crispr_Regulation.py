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

from kattn.kattention import KNET_Crispr_test,CRISPRon,KNET_Crispr_test1,CRISPRon_pt,KNET_Crispr_test2,CRISPRon_base, KNET_Crispr_test3,KNET_Crispr_test4,KNET_Crispr_test5,KNET_Crispr_test6, KNET_Crispr_test7, KNET_Crispr_test8,KNET_Crispr_test9,KNET_Crispr_test11
from kattn.transformers import (
    TransformerCLSModel,
    TransformerAttnModel,
    MHAModel,
    TransformerConfig,

)
from kattn.kattention import KNET_Crispr_test12, KNET_Crispr_test13, KNET_Crispr_test14, KNET_Crispr_test16, KNET_Crispr_test17, KNET_Crispr
from kattn.cnns import CNNModel, CNNTransformerCrispr, CNNTransformerCrisprMatched
from kattn.modules import CNNMixerConfig

from kattn.tokenizers import RNATokenizer, RNAKmerTokenizer
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
import pdb
import pytorch_lightning.callbacks as callbacks
import math
from scipy.stats import spearmanr
import csv

basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%m-%d %H:%M:%S")
logger = getLogger()

KATTN_SRC_DIR = Path(os.environ["KATTN_SRC_DIR"])
KATTN_RESOURCES_DIR = Path(os.environ["KATTN_RESOURCES_DIR"])
# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logging.getLogger("kattn.tokenizers").setLevel(logging.ERROR)  # 或 CRITICAL


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
            self.model = KNET_Crispr_test1(
                # kernel_size=12,
                number_of_kernel=48,
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
        elif model_type == "KNET_Crispr_test5":
            self.model = KNET_Crispr_test5(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test6":
            self.model = KNET_Crispr_test6(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test7":
            self.model = KNET_Crispr_test7(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test8":
            self.model = KNET_Crispr_test8(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test9":
            self.model = KNET_Crispr_test9(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test11":
            self.model = KNET_Crispr_test11(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test12":
            self.model = KNET_Crispr_test12(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test13":
            self.model = KNET_Crispr_test13(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test14":
            self.model = KNET_Crispr_test14(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test16":
            self.model = KNET_Crispr_test16(
                number_of_kernel=48,
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "KNET_Crispr_test17":
            self.model = KNET_Crispr_test17(
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
        elif model_type == "cnn_transformer":
            self.model = CNNTransformerCrispr(
                vocab_size=len(self.tokenizer),
            )
        elif model_type == "cnn_transformer_pm":
            self.model = CNNTransformerCrisprMatched(
                vocab_size=len(self.tokenizer),
            )
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
        self.lambda_l1 = getattr(self.hparams, "lambda_wattn_l1", 1e-5)
        self.lambda_group = getattr(self.hparams, "lambda_wattn_group", 1e-4)
        self.lambda_ortho = getattr(self.hparams, "lambda_wattn_ortho", 0.0)


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
                                                lambda_ortho=self.lambda_ortho)

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

# ========== 工具：从模型收集 Wattn 权重并展平为 per-kernel 向量 ==========
def collect_wattn_kernel_vectors(model: torch.nn.Module):
    """
    返回 dict: layer_name -> Tensor shape [K, feat_dim]
    feat_dim = number of parameters per kernel after flatten

    """
    layers = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ == "KattentionV4":
            W = getattr(module, "Wattn").weight  # Tensor, device may be CUDA
            # W(h k c) c_in(4) kernel_size(1)
            k = int(model.Kattention1.k_l)
            c2 = c1 = 4
            h = int(model.Kattention1.k_n)    # h=num_kernels
            kernels = rearrange(W, "(k h c1) c2 s ->(h s) k c1 c2", k=k, h=h, c1=c1,s=1)
            layers[name] = kernels  # shape [K, feat_dim]
    return layers

# ========== 正则项计算 ==========
def compute_wattn_regularizers(model: torch.nn.Module, 
                               lambda_l1: float = 0.0,     # 元素级 L1
                               lambda_group: float = 0.0,  # 按 head 的去相似（cos 相似度的 off-diag 惩罚）
                               lambda_ortho: float = 0.0   # 额外（可选）的正交约束
                               ):
    """
    返回:
        reg_loss: 标量正则损失
        comps:    各项分量的浮点值（便于日志）
    期望 collect_wattn_kernel_vectors(model) 返回:
        dict[name] -> Tensor of shape [H, K, C1, C2]  (或 [H_like, K, C1, C2])
    """
    layers = collect_wattn_kernel_vectors(model)
    device = next(model.parameters()).device

    total_l1 = torch.tensor(0., device=device)
    total_group = torch.tensor(0., device=device)
    total_ortho = torch.tensor(0., device=device)

    for lname, Wk in layers.items():
        # 迁移到同一 device
        Wk = Wk.to(device)

        # ---- 1) L1：逐元素绝对值之和（不变） ----
        if lambda_l1 > 0:
            total_l1 = total_l1 + Wk.abs().sum()

        # 期望形状: [H_like, K, C1, C2]；若与你实际不一致，请在 collect 函数里规范化
        # 这里将每个 "head" 展平成一个向量
        #   heads_matrix: [H_like, K*C1*C2]
        if Wk.dim() == 4:
            H_like, K, C1, C2 = Wk.shape
            heads_matrix = Wk.reshape(H_like, -1)
        elif Wk.dim() == 3:
            # 兼容: [H_like, K, C] 的情况
            H_like, K, C = Wk.shape
            heads_matrix = Wk.reshape(H_like, -1)
        elif Wk.dim() == 2:
            # 兼容: 已经是 [H_like, feat_dim]
            heads_matrix = Wk
            H_like = heads_matrix.shape[0]
        else:
            raise ValueError(f"[{lname}] Unexpected kernel tensor shape: {tuple(Wk.shape)}")

        # ---- 2) group：按 head 去相似（减少 head 之间的相似性）----
        # 做法：把每个 head 向量 L2 归一化，构造 Gram 矩阵，惩罚 off-diagonal
        if lambda_group > 0:
            eps = 1e-8
            norms = torch.norm(heads_matrix, p=2, dim=1, keepdim=True).clamp_min(eps)
            Hn = heads_matrix / norms                                # [H_like, D]
            G = Hn @ Hn.t()                                           # [H_like, H_like]，近似 cosine 相似度
            I = torch.eye(H_like, device=G.device, dtype=G.dtype)
            offdiag = G - I                                           # 去掉对角（自身与自身）
            # 惩罚项：非对角元素的平方和（越相似惩罚越大）
            total_group = total_group + (offdiag ** 2).sum()

        # ---- 3) 额外（可选）正交惩罚：若你仍希望保留 per-layer 的 kernel 级别正交，可在此实现 ----
        # 这里给出一个示例：把 K 维与通道维合并，对 "kernel 切片" 做正交
        if lambda_ortho > 0:
            # 按 head 内部的 K 个切片来做 Gram（可按需调整）
            # reshape 为 [H_like*K, C1*C2]，每个切片是一个向量
            if Wk.dim() == 4:
                HK = H_like * K
                slices = Wk.reshape(HK, -1)
                snorm = torch.norm(slices, p=2, dim=1, keepdim=True).clamp_min(1e-8)
                Sn = slices / snorm
                Gs = Sn @ Sn.t()                                      # [HK, HK]
                Is = torch.eye(HK, device=Gs.device, dtype=Gs.dtype)
                total_ortho = total_ortho + ((Gs - Is) ** 2).sum()
            else:
                # 若不是 4 维，就退化为对 heads_matrix 做一遍（与 group 相同，不建议重复）
                pass

    reg_loss = lambda_l1 * total_l1 + lambda_group * total_group + lambda_ortho * total_ortho
    comps = {
        "l1":    float(total_l1.detach().cpu()) if total_l1.numel() else 0.0,
        "group": float(total_group.detach().cpu()) if total_group.numel() else 0.0,
        "ortho": float(total_ortho.detach().cpu()) if total_ortho.numel() else 0.0,
    }
    return reg_loss, comps

# %%
def main(
    model_type: str = "cnn", test_config: str = "abs-ran_pwm", num_ds: str = 'set0', num_kernels: int = 32,
    optimizer_type: str = "adamw", max_epochs: int = 200, patience: int = 20,
    max_lr: float = 1e-3, version: int = 0, cache_run: bool = False
):
    wrapped_model = LightningTestModel(
        model_type=model_type,
        test_config=test_config,
        num_ds=num_ds,
        optimizer_type=optimizer_type,
        epochs=max_epochs,
        max_lr=max_lr,
        version=version
    )
    data_module = TestDataModule(
        dst_name=KATTN_SRC_DIR / "kattn/general_crispr",
        dst_config=test_config,
        dst_dir=KATTN_RESOURCES_DIR,
        tokenizer=wrapped_model.tokenizer,
        cache_dir="_cache_dsts",
        batch_size=512,
        num_procs=1,
        num_set=num_ds,
    )
    if cache_run:
        data_module.prepare_data()
        return
    
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')
    # Early_stopping
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss',
                                                  # min_delta=0.00,
                                                #   stopping_threshold=0.99,
                                                  patience=patience,
                                                  verbose=False,
                                                  mode="min"
                                                  )

    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'../../results/Crispr/{test_config}/{model_type}/{num_ds}/{version}',
                                                    filename='best-{epoch:02d}-{val_loss:.3f}'+f'-lr{max_lr}',
                                                    save_top_k=1,
                                                    monitor='val_loss',
                                                    mode="min")

    tb_logger = TensorBoardLogger(f"tb_logs/{test_config}/{model_type}/{num_ds}/{version}",
                                  version=version)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=tb_logger,
        callbacks=[early_stop_callback,checkpoint_callback,lr_monitor],
        precision="16-mixed",
    )

    trainer.fit(wrapped_model, datamodule=data_module)

    trainer.test(wrapped_model, datamodule=data_module, ckpt_path="best")   # or ckpt_path=PATH_TO_CKPT

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-type", type=str, default="KNET_Crispr_test")
    parser.add_argument("-k", "--num-kernels", type=int, default=64)
    parser.add_argument("-c", "--test-config", type=str, default="xu2015TrainKbm7")
    parser.add_argument("-s", "--num-ds", type=str, default="set0")
    parser.add_argument("--optimizer-type", type=str, default="adamw")
    parser.add_argument("-e", "--max-epochs", type=int, default=2000)
    parser.add_argument("--max-lr", type=float, default=1e-4)

    parser.add_argument("-v", "--version", type=int, default=1)

    parser.add_argument("--cache-run", action="store_true")
    parser.add_argument('--patience', default=200, type=int, help='Epoches before early stopping')
    return parser.parse_args()

# %%
if __name__ == "__main__":

    args = parse_args()

    main(
        args.model_type, args.test_config,
        num_ds=args.num_ds,
        num_kernels=args.num_kernels,
        optimizer_type=args.optimizer_type,
        max_epochs=args.max_epochs,
        max_lr=args.max_lr,
        version=args.version,
        cache_run=args.cache_run,
        patience=args.patience
    )
