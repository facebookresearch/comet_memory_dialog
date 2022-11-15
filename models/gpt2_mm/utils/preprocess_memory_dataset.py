#! /usr/bin/env python
"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

Preprocess the memory dialog dataset.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os


MM_CONTEXT = "<MM>"
START_API_CALL = "<SOAC>"
END_API_CALL = "<EOAC>"
START_API_RESULT = "<SOAR>"
START_RESPONSE = "<SOR>"
END_SENTENCE = "<EOS>"
PAD_TOKEN = "<PAD>"
SYSTEM = "<SYSTEM>"
USER = "<USER>"


TEMPLATE_API_PREDICT = "{context} {START_API_CALL} "
TEMPLATE_API_TARGET = "{belief_state} {END_API_CALL}"
TEMPLATE_RESPONSE_PREDICT = (
    "{context} {START_API_CALL} {belief_state} {END_API_CALL} "
    "{START_API_RESULT} {api_result} {START_RESPONSE}"
)
TEMPLATE_RESPONSE_TARGET = "{response} {END_SENTENCE}"


def format_memory_dialog_json(json_path, context_length=2, train=False):
    """ """
    print(f"Reading: {json_path}")
    with open(json_path, "r") as file_id:
        data = json.load(file_id)

    if train:
        additional_special_tokens = set(
            [
                SYSTEM,
                USER,
                START_API_CALL,
                END_API_CALL,
                START_RESPONSE,
                START_API_RESULT,
                MM_CONTEXT,
            ]
        )

    instances = []
    for dialog_datum in data["dialogue_data"]:
        prev_asst_uttr = None
        prev_turn = None
        context_history = []
        for turn in dialog_datum["dialogue"]:
            user_uttr = turn["transcript"].replace("\n", " ").strip()
            user_uttr_api_call_type = turn["api_call"]["call_type"]
            user_uttr_api_result = turn.get("api_result", {})
            user_uttr_parameters = turn["transcript_annotated"][-1]["act_attributes"]
            asst_uttr = turn["system_transcript"].replace("\n", " ").strip()

            # Format main input context
            if prev_asst_uttr:
                memory_objects = prev_turn["system_transcript_annotated"][-1][
                    "act_attributes"
                ]["memories"]
            else:
                memory_objects = []

            context = format_context(
                prev_asst_uttr,
                user_uttr,
                memory_objects,
            )

            prev_asst_uttr = asst_uttr
            prev_turn = turn

            # Concat with previous contexts
            context_history.append(context)
            context = " ".join(context_history[-context_length:])

            # Format belief state
            # Skip if the api_call is unknown
            if user_uttr_api_call_type == "None":
                continue

            if (
                user_uttr_api_result == {}
                or user_uttr_api_result.get("status", "None") == "None"
            ):
                continue

            belief_state = []
            # ***** Temp fix for null participant *****
            if "participant" in user_uttr_parameters["slot_values"]:
                user_uttr_parameters["slot_values"]["participant"] = [
                    p
                    for p in user_uttr_parameters["slot_values"]["participant"]
                    if p is not None
                ]
            # ************************************************

            # Format for API Call.
            str_belief_state = format_api_call(
                user_uttr_api_call_type, user_uttr_parameters
            )

            # Track OOVs
            if train:
                additional_special_tokens.add(user_uttr_api_call_type)
                for slot_name in user_uttr_parameters["slot_values"]:
                    additional_special_tokens.add(str(slot_name))

            # Format for API Result
            str_api_result = format_api_result(user_uttr_api_result)

            new_instance = {
                "dialog_id": dialog_datum["dialogue_idx"],
                "turn_id": turn["turn_idx"],
            }
            # Model two prediction problems.
            # A: Context -> API call
            api_predict = TEMPLATE_API_PREDICT.format(
                context=context,
                START_API_CALL=START_API_CALL,
            )
            api_target = TEMPLATE_API_TARGET.format(
                belief_state=str_belief_state,
                END_API_CALL=END_API_CALL,
            )
            instances.append(
                {
                    "dialog_id": dialog_datum["dialogue_idx"],
                    "turn_id": turn["turn_idx"],
                    "predict": api_predict,
                    "target": api_target,
                    "type": "API",
                }
            )

            # B: Context API call, API result --> Response
            response_predict = TEMPLATE_RESPONSE_PREDICT.format(
                context=context,
                START_API_CALL=START_API_CALL,
                belief_state=str_belief_state,
                END_API_CALL=END_API_CALL,
                START_API_RESULT=START_API_RESULT,
                api_result=str_api_result,
                START_RESPONSE=START_RESPONSE,
            )
            response_target = TEMPLATE_RESPONSE_TARGET.format(
                response=asst_uttr, END_SENTENCE=END_SENTENCE
            )
            instances.append(
                {
                    "dialog_id": dialog_datum["dialogue_idx"],
                    "turn_id": turn["turn_idx"],
                    "predict": response_predict,
                    "target": response_target,
                    "type": "RESPONSE",
                }
            )

    if train:
        special_tokens = {"eos_token": END_SENTENCE, "pad_token": PAD_TOKEN}
        special_tokens["additional_special_tokens"] = list(additional_special_tokens)
    else:
        special_tokens = None
    return instances, data["split"], special_tokens


def format_context(prev_asst_uttr, user_uttr, memory_objects):
    context = ""
    if prev_asst_uttr:
        context += f"{SYSTEM} {prev_asst_uttr} "
        # Add multimodal contexts.
        context += represent_memory_objects(memory_objects) + " "

    context += f"{USER} {user_uttr}"
    return context


def format_api_call(user_uttr_api_call_type, user_uttr_parameters):
    str_belief_state_per_frame = (
        "{act} [ {slot_values} ] ({request_slots}) < {objects} >".format(
            act=user_uttr_api_call_type.strip(),
            slot_values=", ".join(
                [
                    f"{k.strip()} = {str(v).strip()}"
                    for k, v in user_uttr_parameters["slot_values"].items()
                ]
            ),
            request_slots=", ".join(user_uttr_parameters["request_slots"]),
            objects=", ".join([str(o) for o in user_uttr_parameters["memories"]]),
        )
    )
    return str_belief_state_per_frame


def format_api_result(user_uttr_api_result):
    simple_retrieved_info = {}

    if user_uttr_api_result["results"]["retrieved_info"] != []:

        for memory_id, info in user_uttr_api_result["results"][
            "retrieved_info"
        ].items():
            # memory_id: '[Memory ID: 1035119]'
            simple_memory_id = memory_id.split("[Memory ID: ")[-1][:-1]
            simple_retrieved_info[simple_memory_id] = {}

            for slot, value in info.items():
                if slot == "location":
                    simple_retrieved_info[simple_memory_id][slot] = value["place"]
                else:
                    simple_retrieved_info[simple_memory_id][slot] = value

    str_api_result = (
        "{api_status} [ {retrieved_info} ] < {retrieved_memories} >".format(
            api_status=user_uttr_api_result["status"],
            retrieved_info=", ".join(
                [
                    f"{k.strip()} = {str(v).strip()}"
                    for k, v in simple_retrieved_info.items()
                ]
            ).replace("'", ""),
            retrieved_memories=", ".join(
                [str(o) for o in user_uttr_api_result["results"]["retrieved_memories"]]
            ),
        )
    )
    return str_api_result


def represent_memory_objects(object_ids):
    # Stringify visual objects (JSON)
    str_objects = ", ".join([f"{oo}<MM_BREAK>" for oo in object_ids])
    return f"{MM_CONTEXT} {str_objects}"


def main(args):
    instances, split, special_tokens = format_memory_dialog_json(
        args["train_json_path"], train=True
    )
    save_file_path = os.path.join(args["save_folder"], "mem_dials_gpt2_train.json")
    with open(save_file_path, "w") as file_id:
        json.dump(instances, file_id)

    save_file_path = os.path.join(
        args["save_folder"], "mem_dials_gpt2_special_tokens.json"
    )
    with open(save_file_path, "w") as file_id:
        json.dump(special_tokens, file_id)

    for file_path in args["unseen_json_path"]:
        instances, split, _ = format_memory_dialog_json(file_path)
        save_file_path = os.path.join(
            args["save_folder"], f"mem_dials_gpt2_{split}.json"
        )
        with open(save_file_path, "w") as file_id:
            json.dump(instances, file_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_json_path",
        required=True,
        help="Path to the train dataset",
    )
    parser.add_argument(
        "--unseen_json_path",
        default=[],
        required=False,
        nargs="+",
        help="Path to other unseen datsets (val|devtest|test)",
    )
    parser.add_argument(
        "--predict_belief_state",
        action="store_true",
        help="Include belief state in the prediction",
    )
    parser.add_argument(
        "--save_folder", required=True, help="Path to save the processed files"
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
