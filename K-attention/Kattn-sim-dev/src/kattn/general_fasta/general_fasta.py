# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""General purpose fasta file dataset constructer, constructed dataset contains
three fields: name, description, and sequence."""


import os
from pathlib import Path
from typing import List
import json
import logging

import datasets
import pyfastx


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{fasta,
title = {A general purpose fasta file dataset},
author={Ziyu Chen
},
year={2024}
}
"""

# You can copy an official description
_DESCRIPTION = """\
A general purpose fasta file dataset, constructed dataset contains three fields: id, label, and sequence.
If config name starts with mask, will generate another field indicating manual mask
indices according to the corresponding .json file
"""
LEGAL_FILE_SUFFIXES = [
    ".fa",
    ".fasta",
    ".fa.gz",
    ".fasta.gz",
    ".fna.gz",
]

FASTA_FILES_DIRS = {
    "abs-ran_fix1": "abs-ran_fix1.fa",
    "abs-ran_fix2": "abs-ran_fix2.fa",
    "abs-ran_pwm": "abs-ran_pwm.fa",
    "abs-ran_rand": "abs-ran_rand.fa",
    "absolute_fix1": "absolute_fix1.fa",
    "absolute_fix2": "absolute_fix2.fa",
    "absolute_pwm": "absolute_pwm.fa",
    "absolute_rand": "absolute_rand.fa",
    "random_fix1": "random_fix1.fa",
    "random_fix2": "random_fix2.fa",
    "random_pwm": "random_pwm.fa",
    "random_rand": "random_rand.fa",
    "relative_fix1": "relative_fix1.fa",
    "relative_fix2": "relative_fix2.fa",
    "relative_pwm": "relative_pwm.fa",
    "relative_rand": "relative_rand.fa",
    "F_Markov": "F_Markov.fa",
    "S_Markov": "S_Markov.fa",
    "test":"test.fa",
    "markov_test":"markov_test.fa",
    "markov_first": "markov_first.fa",
    "markov_first_1": "markov_first_1.fa",
    "markov_second_1": "markov_second_1.fa",
    "markov_first_2": "markov_first_2.fa",
    "markov_first_3": "markov_first_3.fa",
    "markov_second_2": "markov_second_2.fa",
    "markov_first_4": "markov_first_4.fa",
    "markov_0_25": "markov_0_25.fa",
    "markov_0_5": "markov_0_5.fa",
    "markov_0_75": "markov_0_75.fa",
    "markov_1_0": "markov_1_0.fa",
    "markov_1_25": "markov_1_25.fa",
    "markov_1_5": "markov_1_5.fa",
    "markov_1_75": "markov_1_75.fa",
    "markov_2_0": "markov_2_0.fa",
    "markov_1_0_5000": "markov_1_0_5000.fa",
    "markov_1_0_10000": "markov_1_0_10000.fa",
    "markov_1_0_20000": "markov_1_0_20000.fa",
    "markov_1_0_50000": "markov_1_0_50000.fa",
    "markov_1_0_100000": "markov_1_0_100000.fa",
    "markov_1_25_5000": "markov_1_25_5000.fa",
    "markov_1_25_10000": "markov_1_25_10000.fa",
    "markov_1_25_20000": "markov_1_25_20000.fa",
    "markov_1_25_50000": "markov_1_25_50000.fa",
    "markov_1_25_100000": "markov_1_25_100000.fa",
    "markov_1_5_5000": "markov_1_5_5000.fa",
    "markov_1_5_10000": "markov_1_5_10000.fa",
    "markov_1_5_20000": "markov_1_5_20000.fa",
    "markov_1_5_50000": "markov_1_5_50000.fa",
    "markov_1_5_100000": "markov_1_5_100000.fa",
    "markov_1_75_5000": "markov_1_75_5000.fa",
    "markov_1_75_10000": "markov_1_75_10000.fa",
    "markov_1_75_20000": "markov_1_75_20000.fa",
    "markov_1_75_50000": "markov_1_75_50000.fa",
    "markov_1_75_100000": "markov_1_75_100000.fa",
    "markov_0_75_5000": "markov_0_75_5000.fa",
    "markov_0_75_20000": "markov_0_75_20000.fa",
    "markov_0_75_50000": "markov_0_75_50000.fa",
    "markov_0_75_100000": "markov_0_75_100000.fa",
}

# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class FastaDataset(datasets.GeneratorBasedBuilder):
    """Fasta file based dataset."""

    VERSION = datasets.Version("1.0.0")

    def _config_helper(v):    # workaround for class/list comprehension scope issue
        return  [
            datasets.BuilderConfig(name=k, version=v, description=f"Config for {k}")
            for k in FASTA_FILES_DIRS.keys()
        ]
    BUILDER_CONFIGS = _config_helper(VERSION)

    DEFAULT_CONFIG_NAME = "demo"

    def _info(self):
        if self.config.name.startswith("mask"):
            features = datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "description": datasets.Value("string"),
                    "sequence": datasets.Value("string"),
                    "manual_mask_indices": {
                        "replace": [datasets.Value("int32")],
                        "random": [datasets.Value("int32")],
                        "keep": [datasets.Value("int32")]
                    }
                }
            )
        else:
            features = datasets.Features(
                {
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
        fa_files = []
        filepath = os.path.join(self.config.data_dir, FASTA_FILES_DIRS[self.config.name])
        # a special case for directory filepath, and we want each file in the
        # directory to be a split
        if os.path.isdir(filepath) and self.config.name.startswith("split-"):
            return [
                datasets.SplitGenerator(
                    name=os.path.basename(f).rsplit(".", 1)[0],
                    gen_kwargs={
                        "filepaths": [os.path.join(filepath, f)],
                    }
                )
                for f in os.listdir(filepath)
                if any(f.endswith(suffix) for suffix in LEGAL_FILE_SUFFIXES)
            ]

        # default case, single fasta file or a directory of fasta files, use
        # all files as one train split
        if os.path.isdir(filepath):
            for f in os.listdir(filepath):
                if any(f.endswith(suffix) for suffix in LEGAL_FILE_SUFFIXES):
                    fa_files.append(os.path.join(filepath, f))
        else:
            fa_files.append(filepath)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": fa_files,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepaths: List[str]):
        idx = 0
        if self.config.name.startswith("mask"):
            used_seqs = set()
            for filepath in filepaths:
                mask_file = filepath.rsplit(".", 1)[0] + ".json"
                with open(mask_file, "r", encoding="utf-8") as fp:
                    all_masks = json.load(fp)
                fa = pyfastx.Fasta(filepath)

                for k, masks_k in all_masks.items():
                    if k in fa:
                        if k in used_seqs:
                            logging.warning(f"Sequence with name {k} exists in multiple fasta files.")
                        else:
                            used_seqs.add(k)
                        seq = fa[k]
                        for m in masks_k:
                            yield idx, {
                                "name": seq.name,
                                "description": seq.description.split(" ", 1)[1] if " " in seq.description else "",
                                "sequence": seq.seq,
                                "manual_mask_indices": m if isinstance(m, dict) else {"replace": m},
                            }
                            idx += 1
        else:
            for filepath in filepaths:
                for seq in pyfastx.Fasta(filepath):
                    yield idx, {
                        "name": seq.name,
                        "description": seq.description.split(" ", 1)[1] if " " in seq.description else "",
                        "sequence": seq.seq
                    }
                    idx += 1
