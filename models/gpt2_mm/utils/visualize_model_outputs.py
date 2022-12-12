# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

#! /usr/bin/env python

import argparse
import json
import random


HTML_TEXT_COLUMN = "<td>{text}</td>"
HTML_IMAGE_COLUMN = '{memory_id} <br> <img src="{url}" alt="{memory_id}" width="128" height="128"> <br><br>'
HTML_ROW = """
    <tr>
        {row_columns}
    </tr>
"""
HTML_TABLE = """<table border='1' style='border-collapse:collapse'>\n{}\n</table>"""
HTML_SPACE = "&nbsp;"


def get_memory_id_to_url_mapping(args):
    memories_to_url = {}
    for memory_file in args["memory_files"]:
        with open(memory_file, 'r', encoding='utf-8') as input_file:
            graph_data = json.load(input_file)
            for graph in graph_data:
                for memory in graph['memories']:
                    memories_to_url[memory['memory_id']
                                    ] = memory['media'][0]['url']
    print(f"{len(memories_to_url)} memories loaded")
    return memories_to_url


def process_dst_outputs(model_output):
    """Process the model outputs for the DST task."""
    output_dict = {}
    for dialog_datum in model_output["dialogue_data"]:
        dialog_id = dialog_datum["dialogue_idx"]
        for turn_datum in dialog_datum["dialogue"]:
            turn_id = turn_datum["turn_idx"]
            key = (dialog_id, turn_id)
            output_dict[key] = turn_datum
    return output_dict


def process_response_outputs(model_output):
    """Process the model outputs for the response generation task."""
    output_dict = {}
    for dialog_datum in model_output:
        dialog_id = dialog_datum["dialog_id"]
        for turn_datum in dialog_datum["predictions"]:
            turn_id = turn_datum["turn_id"]
            key = (dialog_id, turn_id)
            output_dict[key] = turn_datum
    return output_dict


def format_ground_truth(gt_turn):
    """Formats the ground truth object for the given turn.

    Args:
        gt_turn: Ground truth for current turn

    Returns:
        context_str: String for context
        api_str: String for API call
        memory_str: String for memories
        response_str: String for response
    """
    turn_id = gt_turn["turn_idx"]
    utterance = gt_turn["transcript"]
    response = gt_turn["system_transcript"]
    api_call = gt_turn["api_call"]["call_type"]
    slots = gt_turn["transcript_annotated"][0]["act_attributes"]["slot_values"]
    request_slots = gt_turn["transcript_annotated"][0]["act_attributes"][
        "request_slots"
    ]
    memories = gt_turn["transcript_annotated"][0]["act_attributes"]["memories"]
    context_str = f"U-{turn_id}: {utterance}"
    api_str = f"API: {api_call}\nSlots: {slots}\nRequest_slots: {request_slots}"
    memory_str = f"Memories: {memories}"
    response_str = f"A-{turn_id}: {response}"
    return context_str, api_str, memory_str, response_str


def format_save_table(table_contents, save_path):
    """Formats and save HTML table given the contents.

    Args:
        table_contents: Contents of HTML table
        save_path: Path to save the HTML table
    """
    num_rows = len(table_contents)
    if table_contents:
        num_columns = len(table_contents[0])
    else:
        return None
    rows_html = []
    # Add header
    rows_html.append(HTML_ROW.format(row_columns="".join([HTML_TEXT_COLUMN.format(
        text=col.replace("\n", "</br>")) for col in table_contents[0]])))
    for row in table_contents[1:]:
        cols_html = ""
        for col in row[:-1]:
            col = col.replace("\n", "</br>")
            cols_html += HTML_TEXT_COLUMN.format(text=col)
        if row[-1] != {}:
            cols_html += "<td>"
            for memory_id, url in row[-1].items():
                cols_html += HTML_IMAGE_COLUMN.format(
                    url=url, memory_id=memory_id)
            cols_html += "</td>"
        rows_html.append(HTML_ROW.format(row_columns=cols_html))
    table_html = HTML_TABLE.format("".join(rows_html))

    print("Saving: {}".format(save_path))
    with open(save_path, "w") as file_id:
        file_id.write(table_html)


def format_model_outputs(dst_output, response_output, memory_to_url_mapping):
    """Format outputs from the model.

    Args:
        dst_output: Model output for DST
        response_output: Model output for response generation

    Returns:
        api_str: String for API call
        memory_str: String for memories
        response_str: String for response
    """
    if dst_output:
        if dst_output.get("transcript_annotated", None) and dst_output.get("api_result", None):
            output = dst_output["transcript_annotated"][0]
            api_call = output["act"]
            slots = output["slots"]
            request_slots = output["request_slots"]
            memories = output["memories"]
            api_str = f"Predicted API call: {api_call}\nSlots: {slots}\nRequest_slots: {request_slots}"
            memory_str = f"Memories: {memories}"
            api_response = dst_output["api_result"]
            api_status = api_response["status"]
            retrieved_memories = api_response["results"]["retrieved_memories"]
            retrieved_memory_urls = {}
            if memory_to_url_mapping != {}:
                retrieved_memory_urls = {memory:
                                         memory_to_url_mapping[memory] for memory in retrieved_memories}
            retrieved_info = api_response["results"]["retrieved_info"]
            api_response_str = f"API response status: {api_status}\nRetrieved_memories: {retrieved_memories}\n \
                                Retrieved info:{retrieved_info}"
        else:
            api_str = "N/A"
            memory_str = "N/A"
            api_response_str = "N/A"
            retrieved_memory_urls = {}
    else:
        api_str = ""
        memory_str = ""
        api_response_str = ""
        retrieved_memory_urls = {}
    if response_output:
        response_str = response_output["response"]
    else:
        response_str = "N/A"
    return api_str, memory_str, response_str, api_response_str, retrieved_memory_urls


def simplify_name(file_name):
    """Simplifies the file name for easier display."""
    return file_name.split("/")[-1]


def main(args):
    with open(args["memory_gt_json"], "r") as file_id:
        gt_data = json.load(file_id)
        gt_dict = process_dst_outputs(gt_data)
        dialog_lens = {
            ii["dialogue_idx"]: len(ii["dialogue"]) for ii in gt_data["dialogue_data"]
        }

    # Reading DST outputs.
    dst_outputs = []
    if args["model_dst_output_jsons"]:
        for file_name in args["model_dst_output_jsons"]:
            with open(file_name, "r") as file_id:
                model_outputs = json.load(file_id)
                dst_outputs.append(process_dst_outputs(model_outputs))

    # Reading response generation outputs.
    response_outputs = []
    if args["model_response_output_jsons"]:
        for file_name in args["model_response_output_jsons"]:
            with open(file_name, "r") as file_id:
                model_outputs = json.load(file_id)
                response_outputs.append(
                    process_response_outputs(model_outputs))
    num_models = max(len(response_outputs), len(dst_outputs))

    # NOTE: If both dst and response outputs are given, script assumes 1-1 map.

    # Randomly sample some dialog ids.
    sampled_dialog_ids = random.sample(
        dialog_lens.keys(), args["num_instances_visualize"]
    )

    # Instance the COCO api for getting the link to the memory url-s
    memory_to_url_mapping = get_memory_id_to_url_mapping(
        args) if args["visualize_memories"] else {}
    rows = []

    for dialog_id in sampled_dialog_ids:
        history = []
        for turn_id in range(dialog_lens[dialog_id]):
            columns = []
            key = (dialog_id, turn_id)
            turn_gt = gt_dict[key]
            context_str, api_str, memory_str, response_str = format_ground_truth(
                turn_gt
            )
            history.append(context_str)
            # Context as first column.
            columns.append("\n".join(history))
            history.extend([HTML_SPACE * 4 + memory_str, response_str + "\n"])
            # Ground truth as second column.
            columns.append("\n".join([api_str, memory_str, response_str]))
            if dst_outputs:
                turn_dst = [ii[key] for ii in dst_outputs]
            else:
                turn_dst = [""] * num_models
            if response_outputs:
                turn_responses = [ii.get(key, "") for ii in response_outputs]
            else:
                turn_responses = [""] * num_models

            for ii in zip(turn_dst, turn_responses):
                api_str, memory_str, response_str, api_response_str, memory_urls = format_model_outputs(
                    *ii, memory_to_url_mapping)
                columns.append(
                    "\n".join([api_str, memory_str, api_response_str, response_str]))
                columns.append(memory_urls)
            rows.append(columns)

    # Add header.
    header = ["Context", "Ground Truth"]
    if args["model_dst_output_jsons"]:
        dst_names = [simplify_name(ii)
                     for ii in args["model_dst_output_jsons"]]
    else:
        dst_names = [""] * num_models
    if args["model_response_output_jsons"]:
        response_names = [
            simplify_name(ii) for ii in args["model_response_output_jsons"]
        ]
    else:
        response_names = [""] * num_models
    header += [f"{ii}</br>{jj}" for ii, jj in zip(dst_names, response_names)]
    if args["visualize_memories"]:
        header += ["Retrieved Memories"]
    rows = [header] + rows

    html_body = format_save_table(rows, args["html_save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument(
        "--memory_gt_json", required=True, help="JSON file with GT data"
    )
    parser.add_argument(
        "--model_dst_output_jsons",
        nargs="+",
        default=[],
        help="JSONs with outputs (for DST)",
    )
    parser.add_argument(
        "--model_response_output_jsons",
        nargs="+",
        default=[],
        help="JSONs with outputs (for response generation)",
    )
    parser.add_argument(
        "--num_instances_visualize",
        type=int,
        default=10,
        help="Number of dialogs to visualize",
    )
    parser.add_argument(
        "--html_save_path",
        default="visualize_model_output.html",
        help="Path to save the HTML file visualization",
    )
    parser.add_argument(
        "--visualize_memories",
        action="store_true",
        help="Whether to visualize memories on output or not.",
    )
    parser.add_argument(
        "--coco_annotations_file",
        default=None,
        help="Path to coco annotations file.",
    )
    parser.add_argument(
        "--memory_files",
        default=[],
        nargs='+',
        help="Path to the memory graphs metadata."
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    # Ensure either of dst or response generation outputs are non-empty.
    valid_inputs = (
        parsed_args["model_response_output_jsons"]
        or parsed_args["model_dst_output_jsons"]
    )
    assert valid_inputs, "One of DST or response outputs should be provided!"
    if parsed_args["visualize_memories"]:
        assert (parsed_args["memory_files"] !=
                []), "In order to visualize memories the path to the memory graph metadata should be specified"
    main(parsed_args)
