# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
"""
    Description: merges the synthetically generated dialogs (.json, .p)
    and the tab-separated Appen annotations (.txt)
    to putput the merged dialogs in both .json and .p formats
"""
import os
import json
import csv
import random
import pickle
from utils import load_data_pickle


if __name__ == "__main__":
    # Parameters for generation
    path_tuples = [
        # Pilot 1: 50 dialogs
        # [
        #    '/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/pilot_1_mem_dials.p',
        #    '/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/paraphrased_0622.csv',
        #    '/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/pilot_1_mem_dials_merged.json',
        #    '/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/pilot_1_mem_dials_merged.p',
        # ],
        # Pilot 2: 450 dialogs
        [
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/pilot_2_mem_dials.p",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/paraphrased_0622.csv",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/pilot_2_mem_dials_merged.json",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/pilot_2_mem_dials_merged.p",
        ],
        # Batch 1: 2000 dialogs
        [
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_1_mem_dials.p",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/paraphrased_0622.csv",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_1_mem_dials_merged.json",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_1_mem_dials_merged.p",
        ],
        # Batch 2: 500 dialogs
        [
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_2_mem_dials.p",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/paraphrased_0622.csv",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_2_mem_dials_merged.json",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_2_mem_dials_merged.p",
        ],
        # Batch 3: 2000 dialogs
        [
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_3_mem_dials.p",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/paraphrased_0622.csv",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_3_mem_dials_merged.json",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_3_mem_dials_merged.p",
        ],
        # Batch 4: 6000 dialogs
        [
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_4_mem_dials.p",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/paraphrased_0622.csv",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_4_mem_dials_merged.json",
            "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_4_mem_dials_merged.p",
        ],
    ]

    for path_tuple in path_tuples:
        path_in_synth = path_tuple[0]
        path_in_appen = path_tuple[1]
        path_out_json = path_tuple[2]
        path_out_pickle = path_tuple[3]

        # Load original synth
        original_dialogs = load_data_pickle(path_in_synth)
        mm_dialogs = []

        # Load paraphrased
        fieldname_to_turn_idx = {
            "turn0_paraphrase": 0,
            "turn1_paraphrase": 1,
            "turn2_paraphrase": 2,
            "turn3_paraphrase": 3,
            "turn4_paraphrase": 4,
            "turn5_paraphrase": 5,
            "turn6_paraphrase": 6,
            "turn7_paraphrase": 7,
            "turn8_paraphrase": 8,
            "turn9_paraphrase": 9,
            "turn10_paraphrase": 10,
            "turn11_paraphrase": 11,
            "turn12_paraphrase": 12,
            "turn13_paraphrase": 13,
            "turn14_paraphrase": 14,
            "turn15_paraphrase": 15,
            "turn16_paraphrase": 16,
            "turn17_paraphrase": 17,
            "turn18_paraphrase": 18,
            "turn19_paraphrase": 19,
            "turn20_paraphrase": 20,
            "turn21_paraphrase": 21,
            "turn22_paraphrase": 22,
            "turn23_paraphrase": 23,
        }
        COL_DIALOG_ID = 88

        turn_idx_to_col = {}
        dialog_id_to_utter = {}

        with open(path_in_appen, "r", encoding="mac_roman") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            for i, line in enumerate(reader):
                if i == 0:
                    for col_id, fieldname in enumerate(line):

                        if fieldname in fieldname_to_turn_idx:
                            turn_idx = fieldname_to_turn_idx[fieldname]
                            turn_idx_to_col[turn_idx] = col_id

                else:
                    dialog_id = int(line[COL_DIALOG_ID])
                    dialog_id_to_utter[dialog_id] = []

                    for turn_idx in range(len(turn_idx_to_col)):
                        if turn_idx in turn_idx_to_col:

                            utter = line[turn_idx_to_col[turn_idx]]
                            utter = utter.strip()

                            if utter != "":
                                dialog_id_to_utter[dialog_id].append(utter)

                            else:
                                if turn_idx < 16:
                                    print(
                                        "Check dialog id %d, turn %d"
                                        % (dialog_id, turn_idx)
                                    )

        # Merge
        for i, mm_d in enumerate(original_dialogs):
            d = mm_d.dialog
            dialog_id = d.idx

            if dialog_id not in dialog_id_to_utter:
                print("Dialog %d is missing." % dialog_id)
                continue

            mm_dialogs.append(mm_d)
            n_rounds = int(len(dialog_id_to_utter[dialog_id]) / 2)

            # TODO: discarding the utterances with missing paraphrases for now
            # Causes: residuals & incompletes from annotations, etc.
            mm_dialogs[-1].dialog.user_turns = mm_dialogs[-1].dialog.user_turns[
                :n_rounds
            ]
            mm_dialogs[-1].dialog.asst_turns = mm_dialogs[-1].dialog.asst_turns[
                :n_rounds
            ]

            for j in range(n_rounds):

                try:
                    user_turn = d.user_turns[j]
                    asst_turn = d.asst_turns[j]

                    user_turn_idx = j * 2
                    asst_turn_idx = j * 2 + 1

                    user_paraphrase = dialog_id_to_utter[dialog_id][user_turn_idx]
                    asst_paraphrase = dialog_id_to_utter[dialog_id][asst_turn_idx]

                    mm_dialogs[-1].dialog.user_turns[j].frames[
                        -1
                    ].uttr = user_paraphrase
                    mm_dialogs[-1].dialog.asst_turns[j].frames[
                        -1
                    ].uttr = asst_paraphrase

                except:
                    print("Missing rounds %d from dialog %d" % (j, dialog_id))
                    print(len(dialog_id_to_utter[dialog_id]))
                    print(len(d.user_turns))

        # Output
        print("Outputting JSON file at %s..." % path_out_json)
        json.dump(
            {"dialogue_data": [mm_d.to_dict() for mm_d in mm_dialogs]},
            open(path_out_json, "w"),
            indent=4,
        )

        pickle.dump(mm_dialogs, open(path_out_pickle, "wb"))
