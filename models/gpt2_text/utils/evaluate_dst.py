# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# !/usr/bin/env python3
"""
    Util functions for evaluating the DST model predictions.
    The script includes a main function which takes
    the original JSON data file and the predicted model output file
    (in the same format), and outputs the report.
"""
import argparse
import json
import copy
import numpy as np


def reformat_turn_intents(turn_intents, ground_truth_act=None):
    new_intents = []
    for intent in turn_intents:
        frame_intent = copy.deepcopy(intent)
        if "act_attributes" in frame_intent:
            frame_intent.update(frame_intent["act_attributes"])
            del frame_intent["act_attributes"]

        # Process ground truth examples.
        if "slot_values" in frame_intent:
            # Tuples are inmutable so we use list of two.
            frame_intent["slots"] = [
                [key, value] for key, value in frame_intent["slot_values"].items()
            ]
            # FIX: Temporarily remove "None" from participants.
            for index, (slot, values) in enumerate(frame_intent["slots"]):
                if slot == "participant":
                    frame_intent["slots"][index][1] = [
                        ii for ii in values if ii is not None
                    ]
            del frame_intent["slot_values"]
        # Process model predictions.
        else:
            frame_intent["slots"] = [
                [key, value] for key, value in frame_intent["slots"].items()
            ]

        # Removes repeated slots and sorts them for correct comparison for both
        # ground truth and model predictions.
        for index, (slot, values) in enumerate(frame_intent["slots"]):
            if type(values) is list:
                frame_intent["slots"][index][1] = sorted(list(set(values)))
            else:
                frame_intent["slots"][index][1] = [values]

        # If new act is provided, re-assign.
        if ground_truth_act:
            frame_intent["act"] = ground_truth_act
        # Convery memories from string to integer.
        if frame_intent["memories"] and ground_truth_act is None:
            frame_intent["memories"] = [
                int(ii) for ii in intent["memories"] if ii.isnumeric()
            ]
        new_intents.append(frame_intent)
    return new_intents


def evaluate_from_json(d_true, d_pred):
    """
    <list>d_true and <list>d_pred are in the following format:
    (Equivalent to "dialogue_data" field in the input data JSON file)
    [
        {
            "dialogue": [
                {
                    "belief_state": [
                        [
                            {
                                'act': <str>,
                                'slots': [
                                    [
                                        SLOT_NAME, SLOT_VALUE
                                    ], ...
                                ]
                            },
                            [End of a frame]
                            ...
                        ],
                    ]
                }
                [End of a turn]
                ...
            ],
        }
        [End of a dialogue]
        ...
    ]
    """
    d_true_flattened = []
    d_pred_flattened = []

    for i in range(len(d_true)):
        # Iterate through each dialog
        dialog_true = d_true[i]["dialogue"]
        dialog_pred = d_pred[i]["dialogue"]
        dialogue_idx = d_true[i]["dialogue_idx"]

        for j in range(len(dialog_true)):
            # Iterate through each turn
            turn_true = dialog_true[j]["belief_state"]
            turn_pred = dialog_pred[j]["belief_state"]

            turn_true["turn_idx"] = j
            turn_true["dialogue_idx"] = dialogue_idx
            d_true_flattened.append(turn_true)
            d_pred_flattened.append(turn_pred)

    return evaluate_from_flat_list(d_true_flattened, d_pred_flattened)


def evaluate_from_json_conservative(d_true, d_pred, lowercase=False):
    """
    <list>d_true and <list>d_pred are in the following format:
    (Equivalent to "dialogue_data" field in the input data JSON file)
    [
        {
            "dialogue": [
                {
                    "belief_state": [
                        [
                            {
                                'act': <str>,
                                'slots': [
                                    [
                                        SLOT_NAME, SLOT_VALUE
                                    ], ...
                                ]
                            },
                            [End of a frame]
                            ...
                        ],
                    ]
                }
                [End of a turn]
                ...
            ],
        }
        [End of a dialogue]
        ...
    ]
    """
    d_true_flattened = []
    d_pred_flattened = []

    num_present = 0
    num_absent = 0
    dst_pool = {ii["dialogue_idx"]: ii for ii in d_pred}
    for gt_datum in d_true:
        # Iterate through each dialog
        dialog_true = gt_datum["dialogue"]
        dialogue_idx = gt_datum["dialogue_idx"]

        if dialogue_idx not in dst_pool:
            print(f"Missing: {dialogue_idx}")
            num_absent += len(gt_datum["dialogue"])
            continue
        # num_present += len(gt_datum["dialogue"])
        dialog_pred = dst_pool[dialogue_idx]["dialogue"]

        for turn_id in range(len(dialog_true)):
            # Iterate through each turn
            if "transcript_annotated" not in dialog_pred[turn_id]:
                print(f"Missing: {dialogue_idx} {turn_id}")
                num_absent += 1
                continue
            num_present += 1

            turn_true = dialog_true[turn_id]["transcript_annotated"]
            turn_pred = dialog_pred[turn_id]["transcript_annotated"]

            # API calls are formatted as acts.
            reformatted_act = dialog_true[turn_id]["api_call"]["call_type"]
            turn_true = reformat_turn_intents(turn_true, reformatted_act)
            turn_pred = reformat_turn_intents(turn_pred)

            d_true_flattened.append(turn_true)
            d_pred_flattened.append(turn_pred)

    # print(len(d_true_flattened))
    # print(len(d_pred_flattened))
    print(f"# present: {num_present} # absent: {num_absent}")
    return evaluate_from_flat_list(
        d_true_flattened, d_pred_flattened, lowercase=lowercase
    )


def evaluate_from_flat_list(d_true, d_pred, lowercase=False):
    """
    <list>d_true and <list>d_pred are in the following format:
    (Each element represents a single turn, with (multiple) frames)
    [
        [
            {
                'act': <str>,
                'slots': [
                    [
                        SLOT_NAME, SLOT_VALUE
                    ], ...
                ]
            },
            [End of a frame]
            ...
        ],
        [End of a turn]
        ...
    ]
    """
    c = initialize_count_dict()

    # Count # corrects & # wrongs
    for i in range(len(d_true)):
        true_turn = d_true[i]
        pred_turn = d_pred[i]
        turn_evaluation = evaluate_turn(true_turn, pred_turn, lowercase=lowercase)

        c = add_dicts(c, turn_evaluation)

    # Calculate metrics
    joint_accuracy = c["n_correct_beliefs"] / c["n_frames"]

    act_rec, act_prec, act_f1 = rec_prec_f1(
        n_correct=c["n_correct_acts"], n_true=c["n_true_acts"], n_pred=c["n_pred_acts"]
    )

    slot_rec, slot_prec, slot_f1 = rec_prec_f1(
        n_correct=c["n_correct_slots"],
        n_true=c["n_true_slots"],
        n_pred=c["n_pred_slots"],
    )

    request_slot_rec, request_slot_prec, request_slot_f1 = rec_prec_f1(
        n_correct=c["n_correct_request_slots"],
        n_true=c["n_true_request_slots"],
        n_pred=c["n_pred_request_slots"],
    )

    object_rec, object_prec, object_f1 = rec_prec_f1(
        n_correct=c["n_correct_objects"],
        n_true=c["n_true_objects"],
        n_pred=c["n_pred_objects"],
    )

    # Calculate std err
    act_f1_stderr = d_f1(c["n_true_acts"], c["n_pred_acts"], c["n_correct_acts"])
    slot_f1_stderr = d_f1(c["n_true_slots"], c["n_pred_slots"], c["n_correct_slots"])
    request_slot_f1_stderr = d_f1(
        c["n_true_request_slots"],
        c["n_pred_request_slots"],
        c["n_correct_request_slots"],
    )
    object_f1_stderr = d_f1(
        c["n_true_objects"], c["n_pred_objects"], c["n_correct_objects"]
    )

    return {
        "joint_accuracy": joint_accuracy,
        "act_rec": act_rec,
        "act_prec": act_prec,
        "act_f1": act_f1,
        "act_f1_stderr": act_f1_stderr,
        "slot_rec": slot_rec,
        "slot_prec": slot_prec,
        "slot_f1": slot_f1,
        "slot_f1_stderr": slot_f1_stderr,
        "request_slot_rec": request_slot_rec,
        "request_slot_prec": request_slot_prec,
        "request_slot_f1": request_slot_f1,
        "request_slot_f1_stderr": request_slot_f1_stderr,
        "object_rec": object_rec,
        "object_prec": object_prec,
        "object_f1": object_f1,
        "object_f1_stderr": object_f1_stderr,
    }


def evaluate_turn(true_turn, pred_turn, lowercase=False):
    count_dict = initialize_count_dict()

    # Must preserve order in which frames appear.
    for frame_idx in range(len(true_turn)):
        # For each frame
        true_frame = true_turn[frame_idx]
        if frame_idx >= len(pred_turn):
            pred_frame = {}
        else:
            pred_frame = pred_turn[frame_idx]

        count_dict = add_dicts(
            count_dict,
            evaluate_frame(true_frame, pred_frame, strict=False, lowercase=lowercase),
        )

    return count_dict


def evaluate_frame(true_frame, pred_frame, strict=True, lowercase=False):
    """
    If strict=True,
        For each dialog_act (frame), set(slot values) must match.
        If dialog_act is incorrect, its set(slot values) is considered wrong.
    """
    count_dict = initialize_count_dict()
    count_dict["n_frames"] += 1

    # Compare Dialog Actss
    true_act = true_frame["act"] if "act" in true_frame else None
    pred_act = pred_frame["act"] if "act" in pred_frame else None
    if not lowercase:
        b_correct_act = true_act == pred_act
    else:
        # Lowercase evaluation.
        b_correct_act = true_act.lower() == str(pred_act).lower()
    count_dict["n_correct_acts"] += b_correct_act
    count_dict["n_true_acts"] += "act" in true_frame
    count_dict["n_pred_acts"] += "act" in pred_frame

    # Compare Slots
    if not lowercase:
        true_frame_slot_values = {f"{k}={v}" for k, v in true_frame.get("slots", [])}
        pred_frame_slot_values = {f"{k}={v}" for k, v in pred_frame.get("slots", [])}
    else:
        true_frame_slot_values = {
            f"{k}={v}".lower() for k, v in true_frame.get("slots", [])
        }
        pred_frame_slot_values = {
            f"{k}={v}".lower() for k, v in pred_frame.get("slots", [])
        }

    count_dict["n_true_slots"] += len(true_frame_slot_values)
    count_dict["n_pred_slots"] += len(pred_frame_slot_values)

    if strict and not b_correct_act:
        pass
    else:
        count_dict["n_correct_slots"] += len(
            true_frame_slot_values.intersection(pred_frame_slot_values)
        )

    # if len(true_frame_slot_values.intersection(pred_frame_slot_values)) != len(pred_frame_slot_values):
    # print(true_frame_slot_values)
    # print(pred_frame_slot_values)
    # print(len(true_frame_slot_values.intersection(pred_frame_slot_values)) == len(pred_frame_slot_values))
    # print('--')

    # Compare Request slots
    true_frame_request_slot_values = {rs for rs in true_frame.get("request_slots", [])}
    pred_frame_request_slot_values = {rs for rs in pred_frame.get("request_slots", [])}
    # print(true_frame_request_slot_values)
    if not lowercase:
        true_frame_request_slot_values = {
            rs for rs in true_frame.get("request_slots", [])
        }
        pred_frame_request_slot_values = {
            rs for rs in pred_frame.get("request_slots", [])
        }
    else:
        true_frame_request_slot_values = {
            rs.lower() for rs in true_frame.get("request_slots", [])
        }
        pred_frame_request_slot_values = {
            rs.lower() for rs in pred_frame.get("request_slots", [])
        }

    count_dict["n_true_request_slots"] += len(true_frame_request_slot_values)
    count_dict["n_pred_request_slots"] += len(pred_frame_request_slot_values)

    if strict and not b_correct_act:
        pass
    else:
        count_dict["n_correct_request_slots"] += len(
            true_frame_request_slot_values.intersection(pred_frame_request_slot_values)
        )

    # Compare Objects
    true_frame_object_values = {
        object_id for object_id in true_frame.get("memories", [])
    }
    pred_frame_object_values = {
        object_id for object_id in pred_frame.get("memories", [])
    }

    count_dict["n_true_objects"] += len(true_frame_object_values)
    count_dict["n_pred_objects"] += len(pred_frame_object_values)

    if strict and not b_correct_act:
        pass
    else:
        count_dict["n_correct_objects"] += len(
            true_frame_object_values.intersection(pred_frame_object_values)
        )

    # Joint
    count_dict["n_correct_beliefs"] += (
        b_correct_act
        and true_frame_slot_values == pred_frame_slot_values
        and true_frame_request_slot_values == pred_frame_request_slot_values
        and true_frame_object_values == pred_frame_object_values
    )

    return count_dict


def add_dicts(d1, d2):
    return {k: d1[k] + d2[k] for k in d1}


def rec_prec_f1(n_correct, n_true, n_pred):
    rec = n_correct / n_true if n_true != 0 else 0
    prec = n_correct / n_pred if n_pred != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

    return rec, prec, f1


def d_f1(n_true, n_pred, n_correct):
    # 1/r + 1/p = 2/F1
    # dr / r^2 + dp / p^2 = 2dF1 /F1^2
    # dF1 = 1/2 F1^2 (dr/r^2 + dp/p^2)
    dr = b_stderr(n_true, n_correct)
    dp = b_stderr(n_pred, n_correct)

    r = n_correct / n_true if n_true else 0
    p = n_correct / n_pred if n_pred else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0

    d_f1 = 0.5 * f1**2 * (dr / r**2 + dp / p**2) if p * r != 0 else 0
    return d_f1


def b_stderr(n_total, n_pos):
    return np.std(b_arr(n_total, n_pos)) / np.sqrt(n_total)


def b_arr(n_total, n_pos):
    out = np.zeros(int(n_total))
    out[: int(n_pos)] = 1.0
    return out


def initialize_count_dict():
    c = {
        "n_frames": 0.0,
        "n_true_acts": 0.0,
        "n_pred_acts": 0.0,
        "n_correct_acts": 0.0,
        "n_true_slots": 0.0,
        "n_pred_slots": 0.0,
        "n_correct_slots": 0.0,
        "n_true_request_slots": 0.0,
        "n_pred_request_slots": 0.0,
        "n_correct_request_slots": 0.0,
        "n_true_objects": 0.0,
        "n_pred_objects": 0.0,
        "n_correct_objects": 0.0,
        "n_correct_beliefs": 0.0,
    }
    return copy.deepcopy(c)


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_target", help="path for target (.json)")
    parser.add_argument(
        "--input_path_predicted", help="path for model prediction output (.json)"
    )
    parser.add_argument(
        "--output_path_report", help="path for saving evaluation summary (.json)"
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        default=False,
        help="Evaluate a lowercase model",
    )

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report

    # Read the JSON file input
    # json_predicted must have the same structure as the original input JSON
    # e.g. {'dialogue_data': [ ... ]}
    json_target = json.load(open(input_path_target, "r"))
    json_predicted = json.load(open(input_path_predicted, "r"))

    # Evaluate
    report = evaluate_from_json_conservative(
        json_target["dialogue_data"],
        json_predicted["dialogue_data"],
        lowercase=args.lowercase,
    )
    # report = evaluate_from_json(json_target['dialogue_data'], json_predicted['dialogue_data'])
    print(report)

    # Save report
    with open(output_path_report, "w") as f_out:
        json.dump(report, f_out)
