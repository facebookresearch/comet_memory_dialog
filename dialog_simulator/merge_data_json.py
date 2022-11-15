# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
"""
    Merges multiple batches of SIMMC 2.0 files into one,
    and also outputs train, dev, devtest, and test sets.
"""
import os
import json
import csv
import random
import pickle
import numpy as np
from utils import load_data_pickle


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    # Paths for merge
    paths_to_merge = [
        #'/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/pilot_1_mem_dials_merged.p',
        "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/pilot_2_mem_dials_merged.p",
        "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_1_mem_dials_merged.p",
        "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_2_mem_dials_merged.p",
        "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_3_mem_dials_merged.p",
        "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_4_mem_dials_merged.p",
    ]

    path_out_json = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/mem_dials_merged.json"
    path_out_pickle = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/mem_dials_merged.p"

    mm_dialogs = []

    for path_in_pickle in paths_to_merge:

        # Load original synth
        mm_dialogs.extend(load_data_pickle(path_in_pickle))

    # Output
    print("Total: %d dialogs" % len(mm_dialogs))

    json.dump(
        {
            "dialogue_data": [mm_d.to_dict() for mm_d in mm_dialogs],
            "split": "all",
            "year": 2021,
            "domain": "memory",
        },
        open(path_out_json, "w"),
        indent=4,
    )

    pickle.dump(mm_dialogs, open(path_out_pickle, "wb"))

    # Split
    r_train = 0.85
    r_dev = 0.10
    r_devtest = 0.04
    r_test = 0.01
    r_mini = 0.001

    path_out_train_json = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/mem_dials_train.json"
    path_out_dev_json = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/mem_dials_dev.json"
    path_out_devtest_json = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/mem_dials_devtest.json"
    path_out_test_json = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/mem_dials_test.json"
    path_out_mini_json = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/mem_dials_mini.json"

    n_dialogs = len(mm_dialogs)
    indices = np.arange(n_dialogs)
    np.random.shuffle(indices)
    n_train = int(n_dialogs * r_train)
    n_dev = int(n_dialogs * r_dev)
    n_devtest = int(n_dialogs * r_devtest)
    n_test = int(n_dialogs * r_test)
    n_mini = int(n_dialogs * r_mini)

    train_indices = indices[:n_train]
    dev_indices = indices[n_train : n_train + n_dev]
    devtest_indices = indices[n_train + n_dev : n_train + n_dev + n_devtest]
    test_indices = indices[n_train + n_dev + n_devtest :]
    mini_indices = test_indices[:n_mini]

    mm_dialogs_train = [mm_d for i, mm_d in enumerate(mm_dialogs) if i in train_indices]
    mm_dialogs_dev = [mm_d for i, mm_d in enumerate(mm_dialogs) if i in dev_indices]
    mm_dialogs_devtest = [
        mm_d for i, mm_d in enumerate(mm_dialogs) if i in devtest_indices
    ]
    mm_dialogs_test = [mm_d for i, mm_d in enumerate(mm_dialogs) if i in test_indices]
    mm_dialogs_mini = [mm_d for i, mm_d in enumerate(mm_dialogs) if i in mini_indices]

    json.dump(
        {
            "dialogue_data": [mm_d.to_dict() for mm_d in mm_dialogs_train],
            "split": "train",
            "year": 2021,
            "domain": "memory",
        },
        open(path_out_train_json, "w"),
        indent=4,
    )

    json.dump(
        {
            "dialogue_data": [mm_d.to_dict() for mm_d in mm_dialogs_dev],
            "split": "dev",
            "year": 2021,
            "domain": "memory",
        },
        open(path_out_dev_json, "w"),
        indent=4,
    )

    json.dump(
        {
            "dialogue_data": [mm_d.to_dict() for mm_d in mm_dialogs_devtest],
            "split": "devtest",
            "year": 2021,
            "domain": "memory",
        },
        open(path_out_devtest_json, "w"),
        indent=4,
    )

    json.dump(
        {
            "dialogue_data": [mm_d.to_dict() for mm_d in mm_dialogs_test],
            "split": "test",
            "year": 2021,
            "domain": "memory",
        },
        open(path_out_test_json, "w"),
        indent=4,
    )

    json.dump(
        {
            "dialogue_data": [mm_d.to_dict() for mm_d in mm_dialogs_mini],
            "split": "mini",
            "year": 2021,
            "domain": "memory",
        },
        open(path_out_mini_json, "w"),
        indent=4,
    )
