# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


# coding: utf-8
"""Dataset Loader for Memory Dialogs.

Author(s): noctli, skottur
(c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""

import json
import logging
import os
import pickle
import re
from itertools import chain

import numpy as np
import torch
import torch.utils.data
import tqdm

from dataset import tokenize
from torch.utils.data import Dataset


# from train import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS
# SPECIAL_TOKENS = ["<bos>", "<eos>", "<user>", "<system>", "<video>", "<pad>"]
# SPECIAL_TOKENS_DICT = {
#     "bos_token": "<bos>",
#     "eos_token": "<eos>",
#     "additional_special_tokens": ["<user>", "<system>", "<video>", "<cap>"],
#     "pad_token": "<pad>",
# }
MODEL_INPUTS = ["input_ids", "token_type_ids", "lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids", "lm_labels"]
MEMORY_BREAK = "<MM_BREAK>"
ANCHOR_TOKENS = ["<USER>", "<SYSTEM>", "<MM>", "<SOAC>", "<SOAR>", "<SOR>"]


def get_dataset(tokenizer, data_file, feature_path=None, feature_width=None):
    """Get dataset given tokenizer and data file."""
    with open(data_file, "r") as file_id:
        instance_data = json.load(file_id)

    # Read the features from the folder.
    if feature_path is not None:
        feature_map = {}
        feature_type = None
        listings = [ii for ii in os.listdir(feature_path) if ".npy" in ii]
        for file_name in listings:
            search_slots = re.findall(r"mscoco_([^_]*)_([\d]*).npy", file_name)
            extracted_type, memory_id = search_slots[0]
            if not feature_type:
                feature_type = extracted_type
            else:
                assert feature_type == extracted_type, (
                    f"Mismatch feature type: {feature_type} != {extracted_type}"
                )
            file_path = os.path.join(feature_path, file_name)
            feature_map[memory_id] = file_path
    else:
        feature_map = None
        feature_type = None

    # instance_data = instance_data[:10]
    for datum in tqdm.tqdm(instance_data, desc="Preparing dataset"):
        context = datum["predict"]
        target = datum["target"]
        # Identify memory features (if any) in the context.
        # NOTE: Make this cleaner, slightly adhoc at the moment.
        split_str = context.split(MEMORY_BREAK)
        memory_ids = []
        for ii in split_str[:-1]:
            memory_ids.append(int(ii.rsplit(" ", 1)[-1]))
        assert len(memory_ids) + 1 == len(split_str), "Invalid MM breaks!"
        # Alternatively zip the two lists.
        zipped_context = [None for _ in range(len(memory_ids) + len(split_str))]
        zipped_context[::2] = split_str
        zipped_context[1::2] = [
            {
                "memory_id": ii,
                "memory_feature_path": os.path.join(
                    feature_path, f"mscoco_{feature_type}_{ii}.npy"
                ),
            }
            for ii in memory_ids
        ]

        # Extract the token types.
        zipped_token_type_ids = []
        zipped_context_tokens = []
        current_type = None
        for context_part in zipped_context:
            if not isinstance(context_part, dict):
                tokenized_substr, substr_type_ids, current_type = tokenize_by_type(
                    context_part, tokenizer, current_type
                )
                assert len(tokenized_substr) == len(
                    substr_type_ids
                ), "String tokens and token ids should be of same length!"
                zipped_context_tokens.append(tokenized_substr)
                zipped_token_type_ids.extend(substr_type_ids)
            else:
                assert "memory_id" in context_part, "Not a memory!"
                if feature_path:
                    zipped_token_type_ids.extend(
                        [tokenizer.convert_tokens_to_ids("<MM>")] * feature_width
                    )
                zipped_context_tokens.append(context_part)
        datum["context_tokens"] = zipped_context_tokens
        datum["context_token_types"] = zipped_token_type_ids

        assert MEMORY_BREAK not in target, "Target cannot have multimodal entries!"
        datum["target_tokens"] = tokenize(target, tokenizer)
        if datum["type"] == "API":
            target_token_type_ids = [tokenizer.convert_tokens_to_ids("<SOAC>")] * len(
                datum["target_tokens"]
            )
        else:
            target_token_type_ids = [tokenizer.convert_tokens_to_ids("<SOR>")] * len(
                datum["target_tokens"]
            )
        datum["target_token_types"] = target_token_type_ids

        # Get input tokens by merging the two.
        input_tokens, input_token_types, lm_labels = merge_context_target_tokens(datum)
        datum["input_tokens"] = input_tokens
        datum["input_token_types"] = input_token_types
        datum["lm_labels"] = lm_labels
    return instance_data, feature_map


def merge_context_target_tokens(datum):
    """Merge context and target tokens."""
    input_tokens = datum["context_tokens"] + [datum["target_tokens"]]
    input_token_types = datum["context_token_types"] + datum["target_token_types"]
    lm_labels = [-1] * len(datum["context_token_types"]) + datum["target_tokens"]
    return input_tokens, input_token_types, lm_labels


def tokenize_by_type(string, tokenizer, start_type=None):
    # Raw tokenization.
    tokens = string.split(" ")
    current_type = start_type
    start_index = 0
    token_splits = []
    for index, token in enumerate(tokens):
        if token in ANCHOR_TOKENS:
            # First discovered token type, do nothing.
            if current_type is not None:
                reconstructed_str = " ".join(tokens[start_index:index])
                token_splits.append((reconstructed_str, current_type))
            start_index = index
            current_type = token
    # Repeat for the last section.
    reconstructed_str = " ".join(tokens[start_index : index + 1])
    token_splits.append((reconstructed_str, current_type))

    # Now tokenize the substrings.
    tokenized_str = []
    tokenized_type_ids = []
    for substring, current_type in token_splits:
        tokenized_substring = tokenize(substring, tokenizer)
        tokenized_str.extend(tokenized_substring)
        tokenized_type_ids.extend(
            [
                tokenizer.convert_tokens_to_ids(current_type)
                for _ in range(len(tokenized_substring))
            ]
        )
    return tokenized_str, tokenized_type_ids, current_type


class MemoryDialogDataset(Dataset):
    def __init__(self, dialogs, tokenizer, features=None, drop_rate=0.5, train=True):
        self.dialogs = dialogs
        self.features = features
        self.tokenizer = tokenizer
        self.drop_rate = drop_rate
        self.train = train

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        instance = self.dialogs[index]
        input_ids = []
        # TODO: Move this to initialization?
        for ii in instance["input_tokens"]:
            if isinstance(ii, list):
                input_ids.append(torch.Tensor(ii).long())
            else:
                if self.features:
                    memory_features = np.load(
                        ii["memory_feature_path"], allow_pickle=True
                    )[()]["features"]
                    input_ids.append({"features": memory_features})
        token_type_ids = torch.Tensor(instance["input_token_types"]).long()
        lm_labels = torch.Tensor(instance["lm_labels"]).long()
        return input_ids, token_type_ids, lm_labels


def padding(seq, pad_token):
    max_len = max([i.size(0) for i in seq])
    input_mask = torch.zeros((len(seq), max_len)).long()
    if len(seq[0].size()) == 1:
        result = torch.ones((len(seq), max_len)).long() * pad_token
    else:
        result = torch.ones(
            (len(seq), max_len, seq[0].size(-1)),
            dtype=seq[0].dtype,
            device=seq[0].device,
        )
    for i in range(len(seq)):
        result[i, : seq[i].size(0)] = seq[i]
        input_mask[i, : seq[i].size(0)] = 1.0
    return result, input_mask


def collate_fn(batch, pad_token, features=None):
    input_ids_list, token_type_ids_list, lm_labels_list, i3d_list = [], [], [], []
    for i in batch:
        input_ids_list.append(i[0])
        token_type_ids_list.append(i[1])
        lm_labels_list.append(i[2])

    token_type_ids, input_mask = padding(token_type_ids_list, pad_token)
    lm_labels, _ = padding(lm_labels_list, -1)
    return input_ids_list, token_type_ids, lm_labels, input_mask


def pad_dataset(dataset, padding=0):
    """Pad the dataset.
    This could be optimized by defining a Dataset class and pad only
    batches but this is simpler.
    """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [
            x + [padding if name != "labels" else -1] * (max_l - len(x))
            for x in dataset[name]
        ]
    return dataset
