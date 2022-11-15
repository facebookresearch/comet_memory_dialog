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
    path_in_pickle = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/mem_dials_merged.p"
    path_out_tsv = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/user_utterances.tsv"

    mm_dialogs = []
    mm_dialogs.extend(load_data_pickle(path_in_pickle))

    # Output
    print("Total: %d dialogs" % len(mm_dialogs))

    with open(path_out_tsv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t", quotechar="'")
        writer.writerow(["dialog_id", "turn_id", "user_utterance"])

        for i, mm_dialog in enumerate(mm_dialogs):
            user_turns = mm_dialog.dialog.user_turns
            dialog_id = mm_dialog.dialog.idx

            for j, user_turn in enumerate(user_turns):
                user_uttr = user_turn.frames[-1].uttr

                if user_uttr not in set(["N/A", "NA"]):
                    row = [dialog_id, j, user_uttr]
                    writer.writerow(row)
