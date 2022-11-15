#!/usr/bin/env python3
"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

    Scripts for evaluating the GPT-2 DST model predictions.

    First, we parse the line-by-line stringified format into responses
    and compute BLEU score.
"""
import argparse
import ast
import copy
import json
import re

import numpy as np
import tqdm
from gpt2_dst.utils.convert import parse_flattened_result


def convert_slots_to_dict(api_call_json):
    """Converts the slots from list of lists to a dict.

    Args:
        api_call_json: JSON containing the parsed API call
    """
    for frame_ind, frame in enumerate(api_call_json):
        slot_dict = {}
        for slot_name, slot_value in frame["slots"]:
            if re.match("\[.*\]", slot_value):
                try:
                    slot_dict[slot_name] = ast.literal_eval(slot_value)
                except:
                    # If error when parsing the slots add empty string
                    print(f"Error parsing: {slot_value} -> {frame}")
                    slot_dict[slot_name] = ""
            else:
                slot_dict[slot_name] = slot_value
        frame["slots"] = slot_dict
    return api_call_json


def parse_results_from_file(input_path, turn_info, original_data):
    """Parse targets from a flattened file to create response, dst evaluation files.

    Args:
        input_path: Path to read the responses from.
        turn_info: List of dialog, turn info.
        original_data: Original JSON target.

    Returns:
        dst_json: JSON file with DST results
        responses_json: JSON file with responses
    """
    # Collate all lines to ensure they start with either <USER> or <SYSTEM>.
    with open(input_path, "r") as file_id:
        lines = [ii.strip() for ii in file_id.readlines()]

    fixed_lines = []
    current_line = ""
    for line in lines:
        if line[:6] == "<USER>" or line[:8] == "<SYSTEM>":
            fixed_lines.append(line)
        else:
            fixed_lines[-1] += line
    print(f"Collating: {len(lines)} -> {len(fixed_lines)}")
    lines = fixed_lines

    # Identify API call string and response in each line.
    assert len(lines) == len(turn_info), "#lines and #turn_info do not match!"
    responses_json = {}
    dst_pool = {}
    for line_ind, line in enumerate(lines):
        dialog_id, turn_id, prediction_type = turn_info[line_ind]
        if prediction_type == "api_call":
            api_call_json = parse_flattened_result(line.split("<EOAC>")[0] + "<EOAC>")
            # Convert slots from list of list to dicts.
            api_call_json = convert_slots_to_dict(api_call_json)
            dst_index = (dialog_id, turn_id)
            assert dst_index not in dst_pool, "Result already exists!"
            dst_pool[dst_index] = api_call_json
            # Check if memories are integers, else skip.
            for frame_info in api_call_json:
                memories = []
                for ii in frame_info["memories"]:
                    try:
                        ii_int = int(ii)
                        memories.append(ii)
                    except:
                        pass
                frame_info["memories"] = memories

        elif prediction_type == "response":
            response_str = line.split("<EOAR>")[-1].strip()
            if dialog_id not in responses_json:
                responses_json[dialog_id] = {
                    "dialog_id": dialog_id,
                    "predictions": [],
                }
            responses_json[dialog_id]["predictions"].append(
                {
                    "turn_id": turn_id,
                    "response": response_str,
                }
            )

        else:
            raise ValueError(f"Invalid prediction_type: {prediction_type}!")
    responses_json = list(responses_json.values())

    num_missing = 0
    num_present = 0
    dst_json = copy.deepcopy(original_data)
    for dialog_datum in dst_json["dialogue_data"]:
        del dialog_datum["mentioned_memory_ids"]
        del dialog_datum["memory_graph_id"]
        dialog_id = dialog_datum["dialogue_idx"]
        for datum in dialog_datum["dialogue"]:
            del datum["transcript_annotated"]
            turn_id = datum["turn_idx"]
            index = (dialog_id, turn_id)
            if index in dst_pool:
                datum["transcript_annotated"] = dst_pool[index]
                num_present += 1
            else:
                print(f"Missing! -- {index}")
                num_missing += 1
    print(f"Missing: {num_missing} Present: {num_present}")
    return dst_json, responses_json


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_target_json", required=True, help="Path to target JSON file"
    )
    parser.add_argument(
        "--input_dialog_ids",
        required=True,
        help="Path for dialog, turn ids for input (.txt)",
    )
    parser.add_argument(
        "--input_path_predicted",
        required=True,
        help="path for model prediction output, line-separated format (.txt)",
    )
    parser.add_argument(
        "--output_path_report",
        required=True,
        help="Path to save evaluation summary (dst and response) (.json)",
    )
    args = parser.parse_args()

    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report
    # Read the input target JSON file.
    with open(args.input_target_json, "r") as file_id:
        original_data = json.load(file_id)

    # Read the dialog and turn ids.
    with open(args.input_dialog_ids, "r") as file_id:
        turn_info = [ast.literal_eval(ii.strip("\n")) for ii in file_id.readlines()]
    # Convert the data from the GPT-2 friendly format to JSON formats.
    dst_json, responses_json = parse_results_from_file(
        input_path_predicted, turn_info, original_data
    )

    # Saving both the DST and response JSON.
    dst_json_path = args.output_path_report.replace(".json", "_dst_results.json")
    print(f"Saving DST results: {dst_json_path}")
    with open(dst_json_path, "w") as file_id:
        json.dump(dst_json, file_id)
    responses_json_path = args.output_path_report.replace(
        ".json", "_response_results.json"
    )
    print(f"Saving responses: {responses_json_path}")
    with open(responses_json_path, "w") as file_id:
        json.dump(responses_json, file_id)
