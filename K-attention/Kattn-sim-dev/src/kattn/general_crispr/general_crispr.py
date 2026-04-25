import os
from pathlib import Path
import gzip
from typing import List

import datasets
from Bio.SeqIO.FastaIO import SimpleFastaParser

_CITATION = """\
@InProceedings{crispr_fasta_ds,
  title   = {CRISPR 30mer FASTA dataset loader (per dataset/set/split)},
  author  = {lt},
  year    = {2025}
}
"""

_DESCRIPTION = """\
Loads FASTA files laid out as <root>/<dataset>/set0..set4/{train,valid,test}_crispr_on.fa(.gz).
Each FASTA record is exposed as (name, description, sequence, length) and carries its
origin metadata (dataset, set_name, split_name). Split granularity is per file:
e.g., split 'set0/train_crispr_on' corresponds to <dataset>/set0/train_crispr_on.fa.
"""

LEGAL_FILE_SUFFIXES = (".fa", ".fasta", ".fa.gz", ".fasta.gz")

# ---- 按需填写：把你要支持的 dataset 名放在这里 ----
DATASET_NAMES = ['chari2015Train293T',
    'doench2014-Hs',
    'doench2014-Mm',
    'doench2016_hg19',
    # 'doench2016plx_hg19',
    'hart2016-Hct1162lib1Avg',
    'hart2016-HelaLib1Avg',
    'hart2016-HelaLib2Avg',
    'hart2016-Rpe1Avg',
    'morenoMateos2015',
    'xu2015TrainHl60',
    'xu2015TrainKbm7',
    'crispron',
    'N2000',
    'N1000',
    'N500',
    'N3075']


class CRISPRFastaDataset(datasets.GeneratorBasedBuilder):
    """Single-fasta dataset loader preserving <dataset>/set?/split structure."""

    VERSION = datasets.Version("1.0.0")

    def _config_helper(v):
        return [
            datasets.BuilderConfig(name=ds_name, version=v, description=f"Config for {ds_name}")
            for ds_name in DATASET_NAMES
        ]

    BUILDER_CONFIGS = _config_helper(VERSION)
    DEFAULT_CONFIG_NAME = DATASET_NAMES[0] if DATASET_NAMES else "demo"

    def _info(self):
        features = datasets.Features(
            {
                # FASTA 内容
                "name": datasets.Value("string"),
                "description": datasets.Value("string"),
                "sequence": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        将每个存在的 <dataset>/<set>/<base>.fa(.gz) 文件映射为一个 split。
        split 名字采用 'setX/<base>'，例如 'set0/train_crispr_on'
        """
        dataset_dir = Path(self.config.data_dir) / self.config.name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"{dataset_dir} not found. Pass data_dir=<root_dir> where <root_dir> contains '{self.config.name}' folder.")

        set_names = [f"set{i}" for i in range(5)]
        bases = ["train_crispr_on", "valid_crispr_on", "test_crispr_on"]

        split_gens = []
        for set_name in set_names:
            set_dir = dataset_dir / set_name
            if not set_dir.exists():
                continue
            for base in bases:
                # 允许 .fa / .fa.gz / .fasta / .fasta.gz
                file_found = None
                for suf in LEGAL_FILE_SUFFIXES:
                    cand = set_dir / f"{base}{suf}"
                    if cand.exists():
                        file_found = cand
                        break
                if not file_found:
                    continue

                split_gens.append(
                    datasets.SplitGenerator(
                        # name=f"{set_name}/{base}",
                        name=f"{set_name}_{base}",
                        gen_kwargs={
                            "dataset_name": self.config.name,
                            "set_name": set_name,
                            "split_name": base,
                            "filepath": file_found,
                        },
                    )
                )

        if not split_gens:
            raise FileNotFoundError(f"No FASTA files found under {dataset_dir} with expected names.")

        return split_gens

    # gen_kwargs 同上
    def _generate_examples(self, dataset_name: str, set_name: str, split_name: str, filepath: Path):
        """
        逐条读出 FASTA 记录：
        - name = 头行空格前
        - description = 头行空格后（若无空格则空串）
        - sequence = 拼接后的序列
        """
        idx = 0
        open_fn = gzip.open if filepath.suffix == ".gz" or filepath.name.endswith(".gz") else open
        mode = "rt"

        with open_fn(filepath, mode, encoding="utf-8") as fh:
            for header, seq in SimpleFastaParser(fh):
                # header 是不带 '>' 的整行
                if " " in header:
                    name, desc = header.split(" ", 1)
                else:
                    name, desc = header, ""
                yield idx, {
                    "name": name,
                    "description": desc,
                    "sequence": seq.replace("\n", "").strip()
                }
                idx += 1
