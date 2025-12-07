from typing import Optional, Any, Literal
from itertools import product
from logging import getLogger
import random
import math

import torch
import numpy as np


logger = getLogger(__name__)

class SpecialTokensMixin():
    r"""
    A mixin to handle special tokens,
    """
    SPECIAL_TOKENS_ATTRIBUTES = [
        "pad_token",
        "cls_token",
        "mask_token",
        "sep_token",
        "bos_token",
        "eos_token",
        "unk_token",
    ]

    def __init__(self, **kwargs):
        self._cls_token = kwargs.get("cls_token", None)
        self._pad_token = kwargs.get("pad_token", None)
        self._mask_token = kwargs.get("mask_token", None)
        self._sep_token = kwargs.get("sep_token", None)
        self._bos_token = kwargs.get("bos_token", None)
        self._eos_token = kwargs.get("eos_token", None)
        self._unk_token = kwargs.get("unk_token", None)

    @property
    def special_tokens(self) -> list[str]:
        return [getattr(self, attr) for attr in self.SPECIAL_TOKENS_ATTRIBUTES
                if getattr(self, "_" + attr) is not None]

    @property
    def cls_token(self):
        assert self._cls_token is not None, "cls_token is not set"
        return self._cls_token

    @property
    def pad_token(self):
        assert self._pad_token is not None, "pad_token is not set"
        return self._pad_token

    @property
    def mask_token(self):
        assert self._mask_token is not None, "mask_token is not set"
        return self._mask_token

    @property
    def sep_token(self):
        assert self._sep_token is not None, "sep_token is not set"
        return self._sep_token

    @property
    def bos_token(self):
        assert self._bos_token is not None, "bos_token is not set"
        return self._bos_token

    @property
    def eos_token(self):
        assert self._eos_token is not None, "eos_token is not set"
        return self._eos_token

    @property
    def unk_token(self):
        assert self._unk_token is not None, "unk_token is not set"
        return self._unk_token

    @cls_token.setter
    def cls_token(self, value: str):
        assert isinstance(value, str), "cls_token must be a string"
        self._cls_token = value

    @pad_token.setter
    def pad_token(self, value: str):
        assert isinstance(value, str), "pad_token must be a string"
        self._pad_token = value

    @mask_token.setter
    def mask_token(self, value: str):
        assert isinstance(value, str), "mask_token must be a string"
        self._mask_token = value

    @sep_token.setter
    def sep_token(self, value: str):
        assert isinstance(value, str), "sep_token must be a string"
        self._sep_token = value

    @bos_token.setter
    def bos_token(self, value: str):
        assert isinstance(value, str), "bos_token must be a string"
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value: str):
        assert isinstance(value, str), "eos_token must be a string"
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value: str):
        assert isinstance(value, str), "unk_token must be a string"
        self._unk_token = value

    @property
    def pad_token_id(self):
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def unk_token_id(self):
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def mask_token_id(self):
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def cls_token_id(self):
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def bos_token_id(self):
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self):
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def sep_token_id(self):
        return self.convert_tokens_to_ids(self.sep_token)


class BaseTokenizer(SpecialTokensMixin):
    """
    - Define shared truncation/split, special tokens, padding, tensorization logics
    - A specific way of including special tokens ("default", "end_aware", "none")
      - default: add [CLS] and [EOS] at the beginning and end of the sequence, no
        matter whether the sequence is truncated or not.
      - end_aware: always add [CLS] to the beginning. Only add [BOS] when the truncated
        sequence actually have the start, and add [EOS] when the truncated sequence
        actually have the end.
      - none: no special tokens added. (except [UNK] for unknown tokens)
    """
    INPUT_IDS_KEY = "input_ids"
    KEY_PADDING_MASK_KEY = "key_padding_mask"
    SPECIAL_TOKENS_MASK_KEY = "special_tokens_mask"     # Used for MLM, masking won't be applied to special tokens

    def __init__(
        self,
        padding_side: Literal["right", "left"] = "right",
        special_token_mode: Literal["default", "end_aware", "none"] = "default",
    ):
        self.special_token_mode = special_token_mode
        special_tokens = {
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "unk_token": "[UNK]",
        }
        if self.special_token_mode == "default":
            special_tokens.update({
                "cls_token": "[CLS]",
                "eos_token": "[EOS]"
            })
        elif self.special_token_mode == "end_aware":
            special_tokens.update({
                "cls_token": "[CLS]",
                "eos_token": "[EOS]",
                "bos_token": "[BOS]"
            })
        elif self.special_token_mode != "none":
            raise ValueError(f"Unsupported special token mode {self.special_token_mode}")

        super().__init__(**special_tokens)

        # initialize the special tokens
        self._token2id = {st: id for id, st in enumerate(self.special_tokens)}
        self._id2token = {id: st for st, id in self._token2id.items()}

        self.padding_side = padding_side

    def __call__(
        self,
        batch: list[str] | str,
        truncation: bool = False,
        truncation_mode: Literal["start", "end", "random"] = "start",
        split: bool = False,    # TODO, not for pretraining, only for downstream tasks
        max_length: Optional[int] = None,
        padding: bool = False,
        pad_to_max_length: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_special_tokens_mask: bool = False,
        tensorize: bool = False,
        tensorize_ignore_keys: Optional[list[str]] = None
    ) -> dict[str, list[list[int]] | Any]:
        r"""
        Main method to tokenize and prepare for the model, one or multiple sequences.

        Note:
        HuggingFace datasets' map method will lose tensor structure after
        dataset.map(tokenizer), so the tensorization happens in the collate_fn
        of DataLoader. (Although also support here for other use cases, like
        homemade Dataset)

        Parameters
        -------------------
        batch: str, list[str]

        Returns
        -------------------
        encoded_batch: dict[str, Any]
            Encoded batch, including input_ids, attention_mask, special_tokens_masks, etc.
        """
        # input single sequence, for compatability with dataset.map
        if isinstance(batch, str): # single sequence
            # batch_ids: list[int]
            results = self._tokenize_single_sequence(batch, return_special_tokens_mask)
            if return_special_tokens_mask:
                batch_ids, special_tokens_mask = results
                encoded_batch = {self.INPUT_IDS_KEY: batch_ids, self.SPECIAL_TOKENS_MASK_KEY: special_tokens_mask}
            else:
                encoded_batch = {self.INPUT_IDS_KEY: results}
        else:  # batch
            # batch_ids: list[list[int]]
            results = [self._tokenize_single_sequence(seq, return_special_tokens_mask) for seq in batch]
            if return_special_tokens_mask:
                batch_ids, special_tokens_mask = list(zip(*results))
                encoded_batch = {self.INPUT_IDS_KEY: batch_ids, self.SPECIAL_TOKENS_MASK_KEY: special_tokens_mask}
            else:
                encoded_batch = {self.INPUT_IDS_KEY: results}

        encoded_batch = self._post_process(
            encoded_batch,
            truncation=truncation,
            truncation_mode=truncation_mode,
            split=split,
            max_length=max_length,
            padding=padding,
            pad_to_max_length=pad_to_max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            tensorize=tensorize,
            tensorize_ignore_keys=tensorize_ignore_keys
        )

        return encoded_batch

    def _post_process(
        self,
        encoded_batch: dict[str, list[Any] | Any],
        truncation: bool = True,
        truncation_mode: Literal["start", "end", "random"] = "start",
        split: bool = False,    # TODO, not for pretraining, only for downstream tasks
        max_length: Optional[int] = None,
        padding: bool = True,
        pad_to_max_length: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        tensorize: bool = False,
        tensorize_ignore_keys: Optional[list[str]] = None
    ) -> dict[str, list[Any] | Any]:
        r"""
        Post process the tokenized batch or single sequence, including padding, truncation, etc.
        Note: only padded input_ids support tensorization.

        Parameters
        -------------------
        batch_ids: list[list[int]] | list[int]
            Tokenized batch.

        Returns
        -------------------
        encoded_batch: dict[str, list[Any]] | dict[str, Any]
            Encoded batch, including input_ids, attention_mask, special_tokens_masks, etc.
            Already tensorized.
        """
        if truncation:
            assert max_length is not None, "max_length must be specified when perform truncation"

        if self.special_token_mode == "default":
            # first truncate/split then add special tokens
            if truncation:
                encoded_batch = self.truncate(encoded_batch, max_length - 2,
                                              truncation_mode)
            encoded_batch = self._add_special_tokens(encoded_batch)
        elif self.special_token_mode == "end_aware":
            # first add special tokens then truncate/split
            # Note [CLS] token is added after truncation
            encoded_batch = self._add_special_tokens(encoded_batch)
            if truncation:
                encoded_batch = self.truncate(encoded_batch, max_length - 1,
                                              truncation_mode)
            encoded_batch = self._add_cls_token(encoded_batch)
        elif self.special_token_mode == "none":
            if truncation:
                encoded_batch = self.truncate(encoded_batch, max_length,
                                              truncation_mode)

        if padding:
            encoded_batch = self.pad(
                encoded_batch, max_length,
                pad_to_max_length=pad_to_max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                tensorize=tensorize,
                tensorize_ignore_keys=tensorize_ignore_keys
            )
        else:
            # no padding, only tensorize when single sequence is given
            if tensorize:
                assert not isinstance(encoded_batch[self.INPUT_IDS_KEY][0], list),\
                    "Tensorization of batch inputs without padding is not supported"
                encoded_batch = self.tensorize(
                    encoded_batch,
                    tensorize_ignore_keys=tensorize_ignore_keys
                )

        return encoded_batch

    def _add_special_tokens(self, encoded_batch: dict[str, list[Any] | Any]):
        r"""
        Add special tokens to input_ids or batched input_ids.
        Note: inplace operation on batch_ids dict.

        A specific way of including special tokens ("default", "end_aware", "none")
        - default: add [CLS] and [EOS] at the beginning and end of the sequence, no
            matter whether the sequence is truncated or not.
        - end_aware: always add [CLS] to the beginning. Only add [BOS] when the truncated
            sequence actually have the start, and add [EOS] when the truncated sequence
            actually have the end.
        - none: no special tokens added. (except [UNK] for unknown tokens)
        """
        if self.special_token_mode == "none":
            return encoded_batch

        batch_ids = encoded_batch[self.INPUT_IDS_KEY]
        special_tokens_mask = encoded_batch.get(self.SPECIAL_TOKENS_MASK_KEY, None)

        if self.special_token_mode == "default":
            front_token = self.cls_token_id
            end_token = self.eos_token_id
        elif self.special_token_mode == "end_aware":
            front_token = self.bos_token_id
            end_token = self.eos_token_id

        if isinstance(batch_ids[0], list):
            if special_tokens_mask is not None:
                special_tokens_mask = [[True] + m + [True] for m in special_tokens_mask]
            batch_ids = [[front_token] + ids + [end_token] for ids in batch_ids]
        else:
            if special_tokens_mask is not None:
                special_tokens_mask = [True] + special_tokens_mask + [True]
            batch_ids = [front_token] + batch_ids + [end_token]

        encoded_batch[self.INPUT_IDS_KEY] = batch_ids
        if special_tokens_mask is not None:
            encoded_batch[self.SPECIAL_TOKENS_MASK_KEY] = special_tokens_mask
        return encoded_batch

    def _add_cls_token(self, encoded_batch: dict[str, list[Any] | Any]):
        r"""
        Add [CLS] token to the beginning of the sequence. Only used in end_aware
        mode, as we need to add [CLS] token after truncation.
        """
        if self.special_token_mode != "end_aware":
            return encoded_batch

        batch_ids = encoded_batch[self.INPUT_IDS_KEY]
        special_tokens_mask = encoded_batch.get(self.SPECIAL_TOKENS_MASK_KEY, None)

        if isinstance(batch_ids[0], list):
            if special_tokens_mask is not None:
                special_tokens_mask = [[True] + m for m in special_tokens_mask]
            batch_ids = [[self.cls_token_id] + ids for ids in batch_ids]
        else:
            if special_tokens_mask is not None:
                special_tokens_mask = [True] + special_tokens_mask
            batch_ids = [self.cls_token_id] + batch_ids

        encoded_batch[self.INPUT_IDS_KEY] = batch_ids
        if special_tokens_mask is not None:
            encoded_batch[self.SPECIAL_TOKENS_MASK_KEY] = special_tokens_mask
        return encoded_batch

    def truncate(
        self,
        encoded_batch: dict[str, list[Any] | Any],
        max_length: int,
        truncation_mode: Literal["start", "end", "random"] = "start"
    ):
        r"""
        Truncate input_ids or batched input_ids to max given length.
        
        Parameters
        -------------------
        encoded_batch: dict containing tokenized sequences
        max_length: maximum length to truncate to
        truncation_mode: 
            - "start": keep the beginning part, truncate from the end
            - "end": keep the end part, truncate from the beginning
            - "random": randomly truncate the sequence maintaining the length
        """
        assert isinstance(max_length, int) and max_length >= 0
        batch_ids = encoded_batch[self.INPUT_IDS_KEY]
        special_tokens_mask = encoded_batch.get(self.SPECIAL_TOKENS_MASK_KEY, None)
        
        if isinstance(batch_ids[0], list):   # batch of sequences
            if truncation_mode == "start":
                batch_ids = [ids[:max_length] for ids in batch_ids]
                if special_tokens_mask is not None:
                    special_tokens_mask = [mask[:max_length] for mask in special_tokens_mask]
            elif truncation_mode == "end":
                batch_ids = [
                    ids[-max_length:] if len(ids) > max_length else ids
                    for ids in batch_ids
                ]
                if special_tokens_mask is not None:
                    special_tokens_mask = [
                        mask[-max_length:] if len(mask) > max_length else mask
                        for mask in special_tokens_mask
                    ]
            elif truncation_mode == "random":
                starts = [
                    random.randint(0, len(ids) - max_length) if len(ids) > max_length else 0
                    for ids in batch_ids
                ]
                batch_ids = [
                    ids[start:start + max_length] for ids, start in zip(batch_ids, starts)
                ]
                if special_tokens_mask is not None:
                    special_tokens_mask = [
                        mask[start:start + max_length] for mask, start in zip(special_tokens_mask, starts)
                    ]
            else:
                raise ValueError(f"Unsupported truncation mode: {truncation_mode}")
        else:     # single sequence
            if truncation_mode == "start":
                batch_ids = batch_ids[:max_length]
                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask[:max_length]
            elif truncation_mode == "end":
                if len(batch_ids) > max_length:
                    batch_ids = batch_ids[-max_length:]
                    if special_tokens_mask is not None:
                        special_tokens_mask = special_tokens_mask[-max_length:]
            elif truncation_mode == "random":
                if len(batch_ids) > max_length:
                    start = random.randint(0, len(batch_ids) - max_length)
                    batch_ids = batch_ids[start:start + max_length]
                    if special_tokens_mask is not None:
                        special_tokens_mask = special_tokens_mask[start:start + max_length]
            else:
                raise ValueError(f"Unsupported truncation mode: {truncation_mode}")

        encoded_batch[self.INPUT_IDS_KEY] = batch_ids
        if special_tokens_mask is not None:
            encoded_batch[self.SPECIAL_TOKENS_MASK_KEY] = special_tokens_mask
        return encoded_batch

    def pad(
        self,
        encoded_batch: dict[str, list[Any] | Any],
        max_length: int = None,
        pad_to_max_length: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        tensorize: bool = False,
        tensorize_ignore_keys: Optional[list[str]] = None
    ) -> dict[str, list[Any] | Any]:
        r"""
        Pad input_ids or batched input_ids to the same length. Can be used in tokenization process, and
        also in DataCollator.

        Parameters
        -------------------
        batch_ids: list[list[int]], list[int]
            Tokenized batch or single sequence.

        Returns
        -------------------
        padded_batch: dict[str, list[Any] | Any]
            Padded batch, including input_ids, key_padding_mask, special_tokens_masks, etc.
        """
        batch_ids = encoded_batch[self.INPUT_IDS_KEY]
        special_tokens_mask = encoded_batch.get(self.SPECIAL_TOKENS_MASK_KEY, None)
        if max_length is None or not pad_to_max_length:
            if isinstance(batch_ids[0], list):
                max_length = max(len(ids) for ids in batch_ids)
            else:
                max_length = len(batch_ids)
        if pad_to_multiple_of is not None:
            max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of

        if isinstance(batch_ids[0], list):
            assert max_length >= max(len(ids) for ids in batch_ids),\
                "max_length must be larger than the longest sequence"
        else:
            assert max_length >= len(batch_ids), "max_length must be larger than the sequence length"

        if self.padding_side == "right":
            if isinstance(batch_ids[0], list):
                padded_batch_input_ids = [ids + [self.pad_token_id] * (max_length - len(ids)) for ids in batch_ids]
                key_padding_mask = [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in batch_ids]
                if special_tokens_mask is not None:
                    special_tokens_mask = [mask + [True] * (max_length - len(mask)) for mask in special_tokens_mask]
            else:
                padded_batch_input_ids = batch_ids + [self.pad_token_id] * (max_length - len(batch_ids))
                key_padding_mask = [1] * len(batch_ids) + [0] * (max_length - len(batch_ids))
                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask + [True] * (max_length - len(special_tokens_mask))
        elif self.padding_side == "left":
            if isinstance(batch_ids[0], list):
                padded_batch_input_ids = [[self.pad_token_id] * (max_length - len(ids)) + ids for ids in batch_ids]
                key_padding_mask = [[0] * (max_length - len(ids)) + [1] * len(ids) for ids in batch_ids]
                if special_tokens_mask is not None:
                    special_tokens_mask = [[True] * (max_length - len(mask)) + mask for mask in special_tokens_mask]
            else:
                padded_batch_input_ids = [self.pad_token_id] * (max_length - len(batch_ids)) + batch_ids
                key_padding_mask = [0] * (max_length - len(batch_ids)) + [1] * len(batch_ids)
                if special_tokens_mask is not None:
                    special_tokens_mask = [True] * (max_length - len(special_tokens_mask)) + special_tokens_mask

        encoded_batch[self.INPUT_IDS_KEY] = padded_batch_input_ids
        encoded_batch[self.KEY_PADDING_MASK_KEY] = key_padding_mask
        if special_tokens_mask is not None:
            encoded_batch[self.SPECIAL_TOKENS_MASK_KEY] = special_tokens_mask

        if tensorize:
            encoded_batch = self.tensorize(encoded_batch,
                                           tensorize_ignore_keys=tensorize_ignore_keys)
        return encoded_batch

    def tensorize(self, encoded_batch: dict[str, list[Any]], prepend_batch_dim: bool = False,
                  tensorize_ignore_keys: Optional[list[str]] = None):
        r"""
        Convert the encoded batch to tensor format.
        """
        if prepend_batch_dim:
            encoded_batch = {key: [value] for key, value in encoded_batch.items()}
            return self.tensorize(encoded_batch)
        for key, value in encoded_batch.items():   # Note not adding new keys, as we are modifying dict in-place
            if tensorize_ignore_keys is not None and key in tensorize_ignore_keys:
                continue
            if isinstance(value, torch.Tensor):
                continue
            elif isinstance(value, list):
                if isinstance(value[0], (list, int, float, bool)):
                    # Note, if value[0] is a list, it's your duty to make sure
                    # all the inner lists have the same length, otherwise add
                    # it to tensorize_ignore_keys
                    encoded_batch[key] = torch.tensor(value)
                elif isinstance(value[0], np.ndarray):
                    value = np.array(value)    # for performance issue
                    encoded_batch[key] = torch.tensor(value)
                else:
                    logger.warning(f"Unsupported type {type(value[0])} of key {key} for tensorization")
                    encoded_batch[key] = value
            elif isinstance(value, np.ndarray):
                encoded_batch[key] = torch.tensor(value)
            else:
                logger.warning(f"Unsupported type {type(value)} of key {key} for tensorization")

        return encoded_batch

    def decode(self, batch: torch.Tensor | list[ list[int] ] | list[int]):
        if isinstance(batch, torch.Tensor):
            batch = batch.tolist()
            return self.decode(batch)
        elif isinstance(batch[0], list):
            return [
                [self._id2token.get(id, self.unk_token) for id in ids]
                for ids in batch
            ]
        else:
            return [self._id2token.get(id, self.unk_token) for id in batch]

    def get_special_tokens_mask(self, padded_batch_input_ids: list[list[int]] | torch.Tensor) -> list[list[bool]] | torch.Tensor:
        special_token_ids = self.special_token_ids
        if isinstance(padded_batch_input_ids, torch.Tensor):
            mask = padded_batch_input_ids.clone()
            mask = mask.apply_(lambda x: x in special_token_ids)
            return mask.bool()
        else:
            return [[id in special_token_ids for id in ids] for ids in padded_batch_input_ids]

    @property
    def vocab(self) -> dict[str, int]:
        return self._token2id

    def __len__(self):
        r"""
        Return the size of the vocabulary of the tokenizer
        """
        return len(self.vocab)

    def _tokenize_single_sequence(self, seq: str, return_special_tokens_mask)\
        -> list[int] | tuple[list[int] | list[bool]]:
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        r"""
        Convert tokens(or one token) to ids.
        """
        if isinstance(tokens, str):
            return self._token2id[tokens]
        return [self._token2id[t] for t in tokens]

    def ordinary_tokens(self) -> list[str]:
        r"""
        Return the ordinary tokens, excluding special tokens.
        """
        raise NotImplementedError

    @property
    def special_token_ids(self) -> list[int]:
        return self.convert_tokens_to_ids(self.special_tokens)

    @property
    def ordinary_tokens_range(self):
        raise NotImplementedError


class RNACodex:
    RNA_BASE_CODE = ("A", "U", "C", "G")
    RNA_BASE_CODEN = ("A", "U", "C", "G", "N")
    IUPAC_CODE = (
        "A", "U", "C", "G", "N", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V",
        "-"  # . also stands for - (gap)
    )
    MODIFIED_BASE_CODE = (
        "A", "U", "C", "G", "I"    # TODO? any other modified nucleotides?
    )

    def __init__(
        self, code_mode: Literal["base", "basen", "iupac"] = "base", T2U: bool = True
    ):
        r"""
        RNA nucleotide codes.
        """
        super().__init__()
        self.code_mode = code_mode
        if self.code_mode == "base":
            self.rna_codes = self.RNA_BASE_CODE
        elif self.code_mode == "basen":
            self.rna_codes = self.RNA_BASE_CODEN
        elif self.code_mode == "iupac":
            self.rna_codes = self.IUPAC_CODE
        else:
            raise ValueError(f"Unsupported RNA code mode {code_mode}")

        self.T2U = T2U
        if not self.T2U:
            self.rna_codes += ("T",)

    def santinize_seq(self, seq: str) -> str:
        seq = seq.upper()
        if self.T2U:
            seq = seq.replace("T", "U")
        if self.code_mode == "iupac":
            seq = seq.replace(".", "-")

        return seq

    def __iter__(self):
        return iter(self.rna_codes)

    def __len__(self):
        return len(self.rna_codes)


class RNATokenizer(BaseTokenizer):

    def __init__(
            self,
            code_mode: Literal["base", "basen", "iupac"] = "basen",
            T2U: bool = True,
            padding_side: Literal["right", "left"] = "right",
            special_token_mode: Literal["default", "end_aware", "none"] = "default",
        ):
        r"""
        Tokenize RNA sequences.

        RNA nucleotides include A, U, C, G, and N. You can also use IUPAC codes
        Special tokens include [CLS], [PAD], [MASK], [BOS], [EOS], [UNK], depending
        on the special_token_mode.
        """
        super().__init__(
            padding_side=padding_side,
            special_token_mode=special_token_mode,
        )

        self.codex = RNACodex(code_mode=code_mode, T2U=T2U)

        num_special_tokens = len(self.special_tokens)
        self._token2id.update({c: id + num_special_tokens for id, c in enumerate(self.codex)})
        self._id2token.update({id + num_special_tokens: c for id, c in enumerate(self.codex)})
        self._ordinary_tokens_range = (num_special_tokens,
                                       num_special_tokens + len(self.codex))

    def _tokenize_single_sequence(
        self, seq: str, return_special_tokens_mask: bool = False
    ) -> list[int] | tuple[list[int] | list[bool]]:
        r"""
        Process a single RNA sequence.
        """
        seq = self.codex.santinize_seq(seq)
        input_ids = [self._token2id.get(c, self.unk_token_id) for c in seq]
        if return_special_tokens_mask:
            special_tokens_mask = [id == self.unk_token_id for id in input_ids]
            return input_ids, special_tokens_mask
        return input_ids

    @property
    def ordinary_tokens_range(self):
        return self._ordinary_tokens_range


class RNAKmerTokenizer(BaseTokenizer):
    def __init__(
        self,
        k: int = 3,
        stride: int = 1,
        fill_end_by_hashtag: bool = False,    # mainly to keep seqlen when k > 1
        code_mode: Literal["base", "basen", "iupac"] = "basen",
        T2U: bool = True,
        padding_side: str = "right",
        special_token_mode: Literal["default", "end_aware", "none"] = "default",
    ):
        r"""
        Tokenize RNA sequences into k-mers.

        RNA nucleotides include A, U, C, G, and N. You can also use IUPAC codes
        Special tokens include [CLS], [PAD], [MASK], [BOS], [EOS], [UNK], depending
        on the special_token_mode.

        Note: dilation is not meaningful for tokenizer, remove it
        """
        super().__init__(
            padding_side=padding_side,
            special_token_mode=special_token_mode
        )
        self.k = k
        self.stride = stride
        self.fill_end_by_hashtag = fill_end_by_hashtag

        self.codex = RNACodex(code_mode=code_mode, T2U=T2U)

        if len(self.codex) > 5 or self.k > 6:
            logger.warning(
                f"Warning: k-mer tokenizer with k={self.k} and codex size {len(self.codex)} is not recommended. "
                "This may lead to an extra large vocabulary size."
            )

        all_kmers = ["".join(kmer) for kmer in product(self.codex, repeat=self.k)]
        if self.fill_end_by_hashtag:
            assert self.stride == 1, "fill_end_by_hashtag is only used to keep " \
                "the sequence length unchanged when stride == 1. If you want to " \
                "provide end info to model, please use special_token_mode='end_aware'."
            self.max_pad_left = math.ceil((self.k - 1) / 2)
            self.max_pad_right = math.floor((self.k - 1) / 2)
            for i in range(1, self.max_pad_left + 1):
                all_kmers.extend("#" * i + "".join(kmer) for kmer in product(self.codex, repeat=k - i))
            for i in range(1, self.max_pad_right + 1):
                all_kmers.extend("".join(kmer) + "#" * i for kmer in product(self.codex, repeat=k - i))

        num_special_tokens = len(self.special_tokens)
        self._token2id.update({c: id + num_special_tokens for id, c in enumerate(all_kmers)})
        self._id2token.update({id + num_special_tokens: c for id, c in enumerate(all_kmers)})
        self._ordinary_tokens_range = (num_special_tokens,
                                       num_special_tokens + len(all_kmers))

    def _tokenize_single_sequence(
        self, seq: str, return_special_tokens_mask: bool = False
    ) -> list[int] | tuple[list[int], list[bool]]:
        r"""
        Process a single RNA sequence. Not adding [CLS][EOS] here.
        """
        seq = self.codex.santinize_seq(seq)

        if self.fill_end_by_hashtag:
            seq = "#" * self.max_pad_left + seq + "#" * self.max_pad_right

        input_ids = [
            self._token2id.get(
                seq[s : s + self.k],
                self.unk_token_id
            )
            for s in range(0, len(seq) - self.k + 1, self.stride)
        ]

        if return_special_tokens_mask:
            special_tokens_mask = [id == self.unk_token_id for id in input_ids]
            return input_ids, special_tokens_mask
        return input_ids

    @property
    def ordinary_tokens_range(self):
        return self._ordinary_tokens_range


class RNABPETokenizer(BaseTokenizer):
    def __init__(self):
        pass
