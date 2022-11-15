# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#!/usr/bin/env python3
"""
    Script for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).

    The reformatted data is used as input for the GPT-2 based
    DST model baseline.
"""
import json
import re
import os


# DSTC style dataset fieldnames
FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_API_CALL = "api_call"
FIELDNAME_API_RESULT = "api_result"
FIELDNAME_USER_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_OF_API_CALL = "=> <SOAC>:"
END_OF_API_CALL = "<EOAC>"
START_OF_API_RESULT = "<SOAR>"
END_OF_API_RESULT = "<EOAR>"
START_OF_RESPONSE = "<SOR>"
END_OF_SENTENCE = "<EOS>"
SYSTEM = "<SYSTEM>"
USER = "<USER>"

TEMPLATE_PREDICT_API = "{context} {START_OF_API_CALL} "
TEMPLATE_TARGET_API = "{context} {START_OF_API_CALL} {belief_state} {END_OF_API_CALL}"
TEMPLATE_PREDICT_RESPONSE = (
    "{context} {START_OF_API_CALL} {belief_state} {END_OF_API_CALL} "
    "{api_result} {END_OF_API_RESULT} "
)
TEMPLATE_TARGET_RESPONSE = (
    "{context} {START_OF_API_CALL} {belief_state} {END_OF_API_CALL} "
    "{api_result} {END_OF_API_RESULT} "
    "{response} {END_OF_SENTENCE}"
)
TEMPLATE_PREDICT = "{context} {START_OF_API_CALL} "
TEMPLATE_TARGET = (
    "{context} {START_OF_API_CALL} {belief_state} {END_OF_API_CALL} "
    "{api_result} {END_OF_API_RESULT} "
    "{response} {END_OF_SENTENCE}"
)

# No belief state predictions and target.
TEMPLATE_PREDICT_NOBELIEF = "{context} {START_OF_RESPONSE} "
TEMPLATE_TARGET_NOBELIEF = "{context} {START_OF_RESPONSE} {response} {END_OF_SENTENCE}"


def convert_json_to_flattened(
    input_path_json,
    output_path_predict,
    output_path_target,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=True,
    input_path_special_tokens="",
    output_path_special_tokens="",
):
    """
    Input: JSON representation of the dialogs
    Output: line-by-line stringified representation of each turn
    """

    with open(input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]

    # Predictions and targets for:
    #    (a) API call
    #    (b) Response Generation
    # Dialog id and turn id for each instance.
    predicts = []
    targets = []
    dialog_turn_info = []
    if input_path_special_tokens != "":
        with open(input_path_special_tokens, "r") as f_in:
            special_tokens = json.load(f_in)
    else:
        special_tokens = {"eos_token": END_OF_SENTENCE}
        additional_special_tokens = [SYSTEM, USER]
        if use_belief_states:
            additional_special_tokens.append(END_OF_API_CALL)
            additional_special_tokens.append(END_OF_API_RESULT)
        else:
            additional_special_tokens.append(START_OF_RESPONSE)
        if use_multimodal_contexts:
            additional_special_tokens.extend(
                [START_OF_MULTIMODAL_CONTEXTS, END_OF_MULTIMODAL_CONTEXTS]
            )
        special_tokens["additional_special_tokens"] = additional_special_tokens

    if output_path_special_tokens != "":
        # If a new output path for special tokens is given,
        # we track new OOVs
        oov = set()

    for _, dialog in enumerate(data):

        prev_asst_uttr = None
        prev_turn = None
        lst_context = []

        for turn in dialog[FIELDNAME_DIALOG]:
            user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()
            user_uttr_api_call_type = turn[FIELDNAME_API_CALL]["call_type"]
            user_uttr_api_result = turn.get(FIELDNAME_API_RESULT, {})
            user_uttr_parameters = turn[FIELDNAME_USER_STATE][-1]["act_attributes"]
            asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()

            # Format main input context
            if prev_asst_uttr and use_multimodal_contexts:
                memory_objects = prev_turn[FIELDNAME_SYSTEM_STATE][-1][
                    "act_attributes"
                ]["memories"]
            else:
                memory_objects = []

            context = format_context(
                prev_asst_uttr, user_uttr, memory_objects, use_multimodal_contexts
            )

            prev_asst_uttr = asst_uttr
            prev_turn = turn

            # Add multimodal contexts -- user shouldn't have access to ground-truth
            """
            if use_multimodal_contexts:
                memory_objects = turn[FIELDNAME_API_CALL]['act_attributes']['memories']
                context += ' ' + represent_memory_objects(memory_objects)
            """

            # Concat with previous contexts
            lst_context.append(context)
            context = " ".join(lst_context[-len_context:])

            # Format belief state
            if use_belief_states:
                # Skip if the api_call is unknown
                if user_uttr_api_call_type == "None":
                    continue

                if (
                    user_uttr_api_result == {}
                    or user_uttr_api_result.get("status", "None") == "None"
                ):
                    continue

                belief_state = []
                # for bs_per_frame in user_uttr_api_call_type:

                # ***** Temp fix for null participant *****
                if "participant" in user_uttr_parameters["slot_values"]:
                    user_uttr_parameters["slot_values"]["participant"] = [
                        p
                        for p in user_uttr_parameters["slot_values"]["participant"]
                        if p is not None
                    ]
                # ************************************************

                # Format for API Call
                str_belief_state = format_api_call(
                    user_uttr_api_call_type, user_uttr_parameters
                )

                # Track OOVs
                if output_path_special_tokens != "":
                    oov.add(user_uttr_api_call_type)
                    for slot_name in user_uttr_parameters["slot_values"]:
                        oov.add(str(slot_name))
                        # slot_name, slot_value = kv[0].strip(), kv[1].strip()
                        # oov.add(slot_name)
                        # oov.add(slot_value)

                # Format for API Result
                str_api_result = format_api_result(user_uttr_api_result)

                # A. Format the predicts and targets for API call.
                predict = TEMPLATE_PREDICT_API.format(
                    context=context,
                    START_OF_API_CALL=START_OF_API_CALL,
                )
                predicts.append(predict)
                target = TEMPLATE_TARGET_API.format(
                    context=context,
                    START_OF_API_CALL=START_OF_API_CALL,
                    belief_state=str_belief_state,
                    END_OF_API_CALL=END_OF_API_CALL,
                )
                targets.append(target)
                dialog_turn_info.append(
                    str((dialog["dialogue_idx"], turn["turn_idx"], "api_call"))
                )

                # B. Format the predicts and targets for response.
                predict = TEMPLATE_PREDICT_RESPONSE.format(
                    context=context,
                    START_OF_API_CALL=START_OF_API_CALL,
                    belief_state=str_belief_state,
                    END_OF_API_CALL=END_OF_API_CALL,
                    api_result=str_api_result,
                    END_OF_API_RESULT=END_OF_API_RESULT,
                )
                predicts.append(predict)
                target = TEMPLATE_TARGET_RESPONSE.format(
                    context=context,
                    START_OF_API_CALL=START_OF_API_CALL,
                    belief_state=str_belief_state,
                    END_OF_API_CALL=END_OF_API_CALL,
                    api_result=str_api_result,
                    END_OF_API_RESULT=END_OF_API_RESULT,
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                )
                targets.append(target)
                dialog_turn_info.append(
                    str((dialog["dialogue_idx"], turn["turn_idx"], "response"))
                )
            else:
                # Format the main input
                predict = TEMPLATE_PREDICT_NOBELIEF.format(
                    context=context, START_OF_RESPONSE=START_OF_RESPONSE
                )
                predicts.append(predict)

                # Format the main output
                target = TEMPLATE_TARGET_NOBELIEF.format(
                    context=context,
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                    START_OF_RESPONSE=START_OF_RESPONSE,
                )
                targets.append(target)

    # Create a directory if it does not exist
    directory = os.path.dirname(output_path_predict)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    directory = os.path.dirname(output_path_target)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Output into text files
    with open(output_path_predict, "w") as f_predict:
        X = "\n".join(predicts)
        f_predict.write(X)

    with open(output_path_target, "w") as f_target:
        Y = "\n".join(targets)
        f_target.write(Y)

    output_path_dialog_info = output_path_target.replace(".txt", "_dialog_turn.txt")
    with open(output_path_dialog_info, "w") as f_target:
        Y = "\n".join(dialog_turn_info)
        f_target.write(Y)

    if output_path_special_tokens != "":
        # Create a directory if it does not exist
        directory = os.path.dirname(output_path_special_tokens)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(output_path_special_tokens, "w") as f_special_tokens:
            # Add oov's (acts and slot names, etc.) to special tokens as well
            special_tokens["additional_special_tokens"].extend(list(oov))
            json.dump(special_tokens, f_special_tokens)


def format_context(prev_asst_uttr, user_uttr, memory_objects, use_multimodal_contexts):

    context = ""
    if prev_asst_uttr:
        context += f"{SYSTEM} : {prev_asst_uttr} "
        if use_multimodal_contexts:
            # Add multimodal contexts
            context += represent_memory_objects(memory_objects) + " "

    context += f"{USER} : {user_uttr}"
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
    """
    target_attributes = ['pos', 'color', 'type', 'class_name', 'decor_style']

    list_str_objects = []
    for obj_name, obj in memory_objects.items():
        s = obj_name + ' :'
        for target_attribute in target_attributes:
            if target_attribute in obj:
                target_value = obj.get(target_attribute)
                if target_value == '' or target_value == []:
                    pass
                else:
                    s += f' {target_attribute} {str(target_value)}'
        list_str_objects.append(s)

    str_objects = ' '.join(list_str_objects)
    """
    str_objects = ", ".join([str(o) for o in object_ids])
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"


def parse_flattened_results_from_file(path):
    results = []
    with open(path, "r") as f_in:
        for line in f_in:
            parsed = parse_flattened_result(line)
            results.append(parsed)

    return results


def parse_flattened_result(to_parse):
    """
    Parse out the belief state from the raw text.
    Return an empty list if the belief state can't be parsed

    Input:
    - A single <str> of flattened result
      e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

    Output:
    - Parsed result in a JSON format, where the format is:
        [
            {
                'act': <str>  # e.g. 'DA:REQUEST',
                'slots': [
                    <str> slot_name,
                    <str> slot_value
                ]
            }, ...  # End of a frame
        ]  # End of a dialog
    """
    # dialog_act_regex = re.compile(r'([\w:?.?]*)  *\[([^\]]*)\] *\(([^\]]*)\) *\<([^\]]*)\>')
    dialog_act_regex = re.compile(r"([\w:?.?]*)  *\[(.*)\] *\(([^\]]*)\) *\<([^\]]*)\>")
    slot_regex = re.compile(r"([A-Za-z0-9_.-:]*)  *= *(\[([^\]]*)\]|[^,]*)")
    # TODO: More elegant way to match in a non-greedy way. Needs testing.
    # slot_regex = re.compile(r"([A-Za-z0-9_.-:]*)  *= *(\[(.*?)\]|[^,]*)")
    request_regex = re.compile(r"([A-Za-z0-9_.-:]+)")
    object_regex = re.compile(r"([A-Za-z0-9]+)")

    belief = []

    # Parse
    splits = to_parse.strip().split(START_OF_API_CALL)
    if len(splits) == 2:
        to_parse = splits[1].strip()
        splits = to_parse.split(END_OF_API_CALL)

        if len(splits) == 2:
            # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
            to_parse = splits[0].strip()

            for dialog_act in dialog_act_regex.finditer(to_parse):
                d = {
                    "act": dialog_act.group(1),
                    "slots": [],
                    "request_slots": [],
                    "memories": [],
                }

                for slot in slot_regex.finditer(dialog_act.group(2)):
                    d["slots"].append([slot.group(1).strip(), slot.group(2).strip()])

                for request_slot in request_regex.finditer(dialog_act.group(3)):
                    d["request_slots"].append(request_slot.group(1).strip())

                for object_id in object_regex.finditer(dialog_act.group(4)):
                    d["memories"].append(object_id.group(1).strip())

                if d != {}:
                    belief.append(d)

    return belief


def test_example(to_parse):
    """Tests parser on an example string.

    Args:
        to_parse: String to parse.
    """
    print(to_parse)
    result = parse_flattened_result(to_parse)
    for slot, value in result[0]["slots"]:
        print(f"{slot} = {value}")
    print("-" * 50)


if __name__ == "__main__":
    test_examples = [
        " => <SOAC>:  API_CALL_TYPE.SEARCH  [ location = Alki Beach ] () <  >  <EOAC>",
        " => <SOAC>:  API_CALL_TYPE.GET_INFO  [  ] (time) < 1022778 >  <EOAC>",
        "  => <SOAC>:  API_CALL_TYPE.SEARCH  [ activity = ['cooking sausages', 'another activity'], time = 3 ] () <  >  <EOAC>",
        "  => <SOAC>:  API_CALL_TYPE.SEARCH  [ location = Bear Mountain,  participant = ['Shane', 'Satwik'], activity = ['eating', 'typing'], time = 2021 ] () <  >  <EOAC>",
    ]
    for example in test_examples:
        test_example(example)
