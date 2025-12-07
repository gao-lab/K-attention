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
from datasets import load_dataset, load_from_disk
from transformers import get_scheduler, BertConfig

from kattn.kattention import KattentionModel,KattentionModel_mask,KattentionModel_pos,KNET
from kattn.transformers import (
    TransformerCLSModel,
    TransformerAttnModel,
    MHAModel,
    TransformerConfig,

)
from kattn.cnns import CNNModel
from kattn.modules import CNNMixerConfig

from kattn.tokenizers import RNATokenizer, RNAKmerTokenizer
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
import pdb
import pytorch_lightning.callbacks as callbacks

basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%m-%d %H:%M:%S")
logger = getLogger()

KATTN_SRC_DIR = Path(os.environ["KATTN_SRC_DIR"])
KATTN_RESOURCES_DIR = Path(os.environ["KATTN_RESOURCES_DIR"])
# %%
class LightningTestModel(L.LightningModule):
    def __init__(
        self,
        model_type: str,  # Literal["rifle", "rnabert", "cnn", "kattn_v3", "kattn_v4", "kattn_lt"],
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
        self.optimizer_type = optimizer_type
        self.tokenizer = RNATokenizer(
            code_mode="base", T2U=True, special_token_mode="none"
        )
        self.need_amp = False

        if model_type == "transformer_cls":
            self.tokenizer = RNATokenizer(
                code_mode="base", T2U=True, special_token_mode="default"
            )
            transformer_config = TransformerConfig(
                vocab_size=len(self.tokenizer),
                position_emb_type="relative_RoPE",
                attn_method="homemade",
                # attn_method="flash-attn",
            )
            # self.need_amp = True
            self.model = TransformerCLSModel(transformer_config)
        elif model_type == "transformer_cls_kmer":
            self.tokenizer = RNAKmerTokenizer(
                k=7, code_mode="base", T2U=True, special_token_mode="default"
            )
            transformer_config = TransformerConfig(
                vocab_size=len(self.tokenizer),
                position_emb_type="relative_RoPE",
                attn_method="homemade",
            )
            self.model = TransformerCLSModel(transformer_config)
        elif model_type.startswith("transformer_attn"):
            transformer_config = TransformerConfig(
                vocab_size=len(self.tokenizer),
                position_emb_type="relative_RoPE",
                attn_method="homemade"
            )
            if model_type == "transformer_attn_cnn":
                cnn_mixer_config = CNNMixerConfig(num_layers=2)
            else:
                cnn_mixer_config = None
            self.model = TransformerAttnModel(
                transformer_config, 
                cnn_config=cnn_mixer_config,
            )
        elif model_type.startswith("mha"):
            transformer_config = TransformerConfig(
                num_hidden_layers=1,
                vocab_size=len(self.tokenizer),
                position_emb_type="relative_RoPE",
                attn_method="homemade"
            )
            if model_type == "mha_cnn":
                cnn_mixer_config = CNNMixerConfig(num_layers=2)
            else:
                cnn_mixer_config = None
            self.model = MHAModel(
                transformer_config,
                cnn_config=cnn_mixer_config,
            )
        elif model_type == "cnn":
            self.model = CNNModel(
                vocab_size=len(self.tokenizer),
            )
        # elif model_type.startswith("kattn"):
        #     self.model = KattentionModel(
        #         embedding_method="onehot",
        #         kattn_version=model_type.split("_", 1)[-1],
        #         vocab_size=len(self.tokenizer),
        #         kernel_size=12,
        #         num_kernels=128,
        #     )
        elif model_type == "kattn_v4_mask":
            self.model = KattentionModel_mask(
                embedding_method="onehot",
                kattn_version=model_type.split("_", 1)[-1],
                vocab_size=len(self.tokenizer),
                kernel_size=12,
                num_kernels=128,
            )
        elif model_type == "KNET":
            self.model = KNET(
                kernel_size=12,
                num_kernels=64
            )
        # elif model_type == "kattn_v4_pos":
        #     self.model = KattentionModel_pos(
        #         embedding_method="onehot",
        #         kattn_version=model_type.split("_", 1)[-1],
        #         vocab_size=len(self.tokenizer),
        #         kernel_size=12,
        #         num_kernels=128,
        #         cnn_config = CNNMixerConfig(
        #                         num_layers=3,
        #                         in_channels=128,
        #                         conv_kernel_sizes=[3, 5, 7],  # 查看与层数一致
        #                         conv_mid_channels=[32, 64, 128]
        #                     )
        #     )
        else:
            raise ValueError(f"model_type {model_type} not supported")

        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.max_lr = max_lr

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()

    def forward(self, **kwargs):
        if self.need_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.model(**kwargs)
        else:
            return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]
        self.log("train/loss", loss, prog_bar=True)

        self.train_accuracy(outputs["cls_logits"], batch["cls_labels"])
        self.log("train/accuracy", self.train_accuracy, prog_bar=True)

        # weight1 = self.model.kattn.Wattn.weight.squeeze(-1)
        # weight1_init = rearrange(weight1, "(k h c1) c2 -> k c1 c2 h", k=12, c1=7, h=128).detach()
        # print(weight1_init[:,:,:,0])
        return loss

    def on_validation_start(self) -> None:
        self.auroc_list_ = []
        # self.val_auroc.reset()
        # self.val_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        self.val_accuracy(outputs["cls_logits"], batch["cls_labels"])
        self.val_auroc(outputs["cls_logits"], batch["cls_labels"])
        self.log("val/accuracy", self.val_accuracy, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_epoch=True, sync_dist=True, prog_bar=True)

        current_auroc = self.val_auroc.compute()
        self.auroc_list_.append(current_auroc.item())

    def on_validation_epoch_end(self) -> None:
        self.auroc_list.append(sum(self.auroc_list_)/len(self.auroc_list_))

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

class TestDataModule(L.LightningDataModule):
    def __init__(self, dst_name, dst_config, dst_dir, tokenizer,
                 val_split: float = 0.1, seed: int = 11,
                 max_seqlen: int = 512, batch_size: int = 128,
                 num_procs: int = 4, cache_dir: str | Path = None):
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
        if os.path.exists(os.path.join(self.cache_dir, self.dst_config)):
            logger.info(f"Found tokenized dataset in {os.path.join(self.cache_dir, self.dst_config)}")
            return

        # preprocess only on master rank
        dataset = load_dataset(
            str(self.dst_name), self.dst_config, split="train",
            trust_remote_code=True, data_dir=str(self.dst_dir),
            num_proc=self.num_procs
        )
        dataset = dataset.map(
            lambda batch: {"cls_labels": [1 if s.split(" ")[0] == "positive" else 0 for s in batch["description"]]},
            batched=True, remove_columns=["description", "name"])
        datasets = dataset.train_test_split(
            test_size=self.val_split, shuffle=True, seed=self.seed
        )

        datasets.save_to_disk(str(self.cache_dir / self.dst_config))

    def setup(self, stage):
        tokenized_datasets = load_from_disk(str(self.cache_dir / self.dst_config))
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
            self.dsts["test"],
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_procs,
            pin_memory=True,
            shuffle=False
        )

# %%
def main(
    model_type: str = "cnn", test_config: str = "abs-ran_pwm", num_kernels: int = 32,
    optimizer_type: str = "adamw", max_epochs: int = 200, patience: int = 20,
    max_lr: float = 1e-3, version: int = 0, cache_run: bool = False
):
    wrapped_model = LightningTestModel(
        model_type=model_type,
        optimizer_type=optimizer_type,
        epochs=max_epochs,
        max_lr=max_lr
    )
    data_module = TestDataModule(
        dst_name=KATTN_SRC_DIR / "kattn/general_fasta",
        dst_config=test_config,
        dst_dir=KATTN_RESOURCES_DIR,
        tokenizer=wrapped_model.tokenizer,
        cache_dir="_cache_dsts",
        batch_size=512,
        num_procs=4
    )
    if cache_run:
        data_module.prepare_data()
        return
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')
    # Early_stopping
    early_stop_callback = callbacks.EarlyStopping(monitor='val_auroc',
                                                  min_delta=0.00,
                                                  patience=patience,
                                                  verbose=False,
                                                  mode="max")

    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'../../results/checkpoint/{test_config}/{model_type}',
                                                    filename='best-{epoch:02d}-{val_auroc:.3f}',
                                                    save_top_k=1,
                                                    monitor='val_auroc',
                                                    mode="max")

    tb_logger = TensorBoardLogger(f"tb_logs/{test_config}/{model_type}",
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
    )

    trainer.fit(wrapped_model, datamodule=data_module)
    val_auroc = max(wrapped_model.auroc_list)
    print(f"Best_val_auroc {val_auroc} {args.model_type} {args.test_config}")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-type", type=str, default="kattn_v4_mask")
    parser.add_argument("-k", "--num-kernels", type=int, default=32)
    parser.add_argument("-c", "--test-config", type=str, default="abs-ran_pwm")
    parser.add_argument("--optimizer-type", type=str, default="adamw")
    parser.add_argument("-e", "--max-epochs", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=1e-4)

    parser.add_argument("-v", "--version", type=int, default=0)

    parser.add_argument("--cache-run", action="store_true")
    parser.add_argument('--patience', default=20, type=int, help='Epoches before early stopping')
    return parser.parse_args()

# %%
if __name__ == "__main__":
    args = parse_args()

    main(
        args.model_type, args.test_config,
        num_kernels=args.num_kernels,
        optimizer_type=args.optimizer_type,
        max_epochs=args.max_epochs,
        max_lr=args.max_lr,
        version=args.version,
        cache_run=args.cache_run,
        patience=args.patience
    )
