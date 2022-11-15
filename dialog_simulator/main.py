# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
import os
import copy
import json
import csv
import random
import pickle
from MemoryDialogSimulator import MemoryDialogSimulator
from UserSimulator import PilotUserSimulator
from AssistantSimulator import PilotAssistantSimulator
from GoalGenerator import RuleBasedGoalGenerator
from MemoryServiceAPI import MemoryServiceAPI
from utils import str_memory


if __name__ == "__main__":
    # Parameters for generation
    domain = "memory"
    random.seed(0)
    n_dialogs = 6000
    n_max_turns = 8  # 5, 8, 10
    goal_config = {
        "n_min_goals": 3,  # 4
        "n_max_goals": 6,  # 6
    }

    start_dialog_idx = 5500
    # path_memory_graph_list = '/Users/shanemoon/workspace/memory_dialog/dialog_simulator/memories/final/memory_may21_v1_100graphs.json'
    path_memory_graph_list = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/memories/final/mscoco_memory_graphs_1k.json"
    path_out_json = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_4_mem_dials.json"
    path_out_csv = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_4_mem_dials.tsv"
    path_out_pickle = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/results/batch_4_mem_dials.p"

    # Make sure we are not overwriting
    debug = False
    if not debug:
        assert not os.path.exists(path_out_json)
        assert not os.path.exists(path_out_csv)
        assert not os.path.exists(path_out_pickle)

    # Load parameters
    memory_graph_list = json.load(open(path_memory_graph_list, "r"))
    memory_graph_bank = {}

    for memory_graph in memory_graph_list:
        memory_graph_id = memory_graph["memory_graph_id"]

        for i in range(len(memory_graph["memories"])):
            memory_graph["memories"][i]["memory_graph_id"] = memory_graph_id

        memory_graph_bank[memory_graph_id] = memory_graph

    # Initialize the multimodal simulator
    sim = MemoryDialogSimulator(
        user_simulator=PilotUserSimulator(),
        assistant_simulator=PilotAssistantSimulator(),
        goal_generator=RuleBasedGoalGenerator(domain=domain),
        memory_service_api=MemoryServiceAPI(metadata={}),
        memory_graph_bank=memory_graph_bank,
        domain=domain,
    )

    # Generate dialogs
    memory_dialogs = sim.batch_generate_dialog_flows(
        n_dialogs=n_dialogs,
        n_max_turns=n_max_turns,
        start_dialog_idx=start_dialog_idx,
        goal_config=goal_config,
    )

    # Output dialogs
    # a. Pickle output
    pickle.dump(memory_dialogs, open(path_out_pickle, "wb"))

    # b. JSON output
    json.dump(
        {"dialogue_data": [m_d.to_dict() for m_d in memory_dialogs]},
        open(path_out_json, "w"),
        indent=4,
    )

    # c. print output
    for i, m_d in enumerate(memory_dialogs[:20]):
        d = m_d.dialog
        str_dialog = ""
        print(f"----- Dialog {d.idx} ----- ")
        for j in range(len(d.user_turns)):
            user_turn = d.user_turns[j]
            asst_turn = d.asst_turns[j]
            for user_frame in user_turn.frames:
                str_dialog += "U: " + user_frame.uttr + "\n"
                # str_dialog += 'U: ' + str(user_frame.nlu.act_attributes.slot_values.values()) + '\n'

            for asst_frame in asst_turn.frames:
                str_dialog += "A: " + asst_frame.uttr + "\n"
                # str_dialog += 'A: ' + str(asst_frame.nlu.act_attributes.slot_values.values()) + '\n'

        print(str_dialog)

    # d. TSV output for annotation
    url_blank = "https://simmc2.s3-us-west-1.amazonaws.com/white.png"

    with open(path_out_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t", quotechar="'")
        writer.writerow(
            [
                "dialog_id",
                "dialog",
                "img_0_url",
                "img_1_url",
                "img_2_url",
                "img_3_url",
                "img_4_url",
                "img_5_url",
                "img_6_url",
                "img_7_url",
                "img_8_url",
                "img_9_url",
                "img_10_url",
                "img_11_url",
                "img_12_url",
                "img_13_url",
                "img_14_url",
                "img_15_url",
                "img_0_desc",
                "img_1_desc",
                "img_2_desc",
                "img_3_desc",
                "img_4_desc",
                "img_5_desc",
                "img_6_desc",
                "img_7_desc",
                "img_8_desc",
                "img_9_desc",
                "img_10_desc",
                "img_11_desc",
                "img_12_desc",
                "img_13_desc",
                "img_14_desc",
                "img_15_desc",
                "metadata",
            ]
        )
        for _, m_d in enumerate(memory_dialogs):
            mg = m_d.memory_graph
            d = m_d.dialog

            dialog_data = []
            image_id = 0
            all_image_urls = [url_blank]
            all_memories = [None]

            display_image_ids = [image_id]

            for i in range(len(d.user_turns)):
                # User turn
                user_turn = d.user_turns[i]
                user_utter = "USER: " + ". ".join(
                    [frame.uttr for frame in user_turn.frames]
                )

                user_turn_data = {
                    "turn_id": i * 2,
                    "speaker": "USER",
                    "utterance": user_utter.replace("'", ""),
                    "image_id": copy.deepcopy(display_image_ids),
                    "validation": []
                    #'validation': make_validation_tokens_for_turn(user_turn)
                }

                # Assistant turn
                asst_turn = d.asst_turns[i]
                asst_utter = "ASSISTANT: " + ". ".join(
                    [frame.uttr for frame in asst_turn.frames]
                )

                memory_ids = asst_turn.frames[-1].act_attributes.to_dict()["memories"]
                if memory_ids != []:
                    display_urls = []
                    display_image_ids = []

                    for memory_id in memory_ids:
                        display_urls.extend(mg.get_memory_url(memory_id))
                        image_id += 1
                        display_image_ids.append(image_id)

                    all_image_urls.extend(display_urls)
                    all_memories.extend(mg.get_memories_by_ids(memory_ids))

                asst_turn_data = {
                    "turn_id": i * 2 + 1,
                    "speaker": "ASSISTANT",
                    "utterance": asst_utter.replace("'", ""),
                    "image_id": copy.deepcopy(display_image_ids),
                    "validation": []
                    #'validation': make_validation_tokens_for_turn(asst_turn)
                }

                dialog_data.append(user_turn_data)
                dialog_data.append(asst_turn_data)

            # This should be true, assuming each memory has one image.
            assert len(all_image_urls) == len(all_memories)

            writer.writerow(
                [
                    d.idx,
                    str(json.dumps(dialog_data)),
                    all_image_urls[0],  # url_0
                    all_image_urls[1] if len(all_image_urls) > 1 else "",
                    all_image_urls[2] if len(all_image_urls) > 2 else "",
                    all_image_urls[3] if len(all_image_urls) > 3 else "",
                    all_image_urls[4] if len(all_image_urls) > 4 else "",
                    all_image_urls[5] if len(all_image_urls) > 5 else "",
                    all_image_urls[6] if len(all_image_urls) > 6 else "",
                    all_image_urls[7] if len(all_image_urls) > 7 else "",
                    all_image_urls[8] if len(all_image_urls) > 8 else "",
                    all_image_urls[9] if len(all_image_urls) > 9 else "",
                    all_image_urls[10] if len(all_image_urls) > 10 else "",
                    all_image_urls[11] if len(all_image_urls) > 11 else "",
                    all_image_urls[12] if len(all_image_urls) > 12 else "",
                    all_image_urls[13] if len(all_image_urls) > 13 else "",
                    all_image_urls[14] if len(all_image_urls) > 14 else "",
                    all_image_urls[15] if len(all_image_urls) > 15 else "",
                    "",  # url_0
                    str_memory(all_memories[1]) if len(all_image_urls) > 1 else "",
                    str_memory(all_memories[2]) if len(all_image_urls) > 2 else "",
                    str_memory(all_memories[3]) if len(all_image_urls) > 3 else "",
                    str_memory(all_memories[4]) if len(all_image_urls) > 4 else "",
                    str_memory(all_memories[5]) if len(all_image_urls) > 5 else "",
                    str_memory(all_memories[6]) if len(all_image_urls) > 6 else "",
                    str_memory(all_memories[7]) if len(all_image_urls) > 7 else "",
                    str_memory(all_memories[8]) if len(all_image_urls) > 8 else "",
                    str_memory(all_memories[9]) if len(all_image_urls) > 9 else "",
                    str_memory(all_memories[10]) if len(all_image_urls) > 10 else "",
                    str_memory(all_memories[11]) if len(all_image_urls) > 11 else "",
                    str_memory(all_memories[12]) if len(all_image_urls) > 12 else "",
                    str_memory(all_memories[13]) if len(all_image_urls) > 13 else "",
                    str_memory(all_memories[14]) if len(all_image_urls) > 14 else "",
                    str_memory(all_memories[15]) if len(all_image_urls) > 15 else "",
                    {},  # mockup
                ]
            )
            # print(json.dumps(dialog_data))

    # (5) Summary
    print("n_dialogs:", len(memory_dialogs))
    print("n_turns:", sum([len(m_d.dialog.asst_turns) for m_d in memory_dialogs]))
