# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
import random
import json
from MemoryDialogModel import PilotMemoryDialogModel
from Data import MemoryGraph, MemoryDialog, Turn
from MemoryServiceAPI import MemoryServiceAPI

import sys

sys.path.append("/Users/shanemoon/workspace/memory_dialog/models/")
from gpt2_dst.scripts.run_generation import load_model


class InteractiveDialogHandler:
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop("model", None)
        self.memory_graph = kwargs.pop("memory_graph", None)
        self.api = kwargs.pop("api", None)

        # Start an empty dialog data
        self.memory_dialog = MemoryDialog(memory_graph=self.memory_graph)
        self.memory_dialog.initialize()

    def execute_turn(self, user_query: str) -> Turn:
        """
        Given user_query, construct an API call,
        get the API response, and return an Assistant Turn.
        """

        # Construct the API request
        try:
            user_turn, api_request = self.model.construct_api_request(
                user_query, self.memory_dialog
            )
            print("============== API Request ==============")
            print(api_request)
            print("=========================================\n")

            # Call API to get responses back
            api_response = self.api.call_api(api_request)
            print("============== API Response ==============")
            print(api_response)
            print("==========================================\n")

            # Update the display based on the API results
            self.model.update_display(api_response)

            # Generate an Assistant response based on the API response
            assistant_turn = self.model.construct_assistant_response(
                user_query, api_request, api_response, self.memory_dialog
            )
            print("============== Assistant Response ==============")
            print(assistant_turn)
            print("================================================\n")

            # Update the memory_dialog with the new user and assistant turns
            self.memory_dialog.dialog.add_user_turn(user_turn)
            self.memory_dialog.dialog.add_asst_turn(assistant_turn)

            # Update the model
            self.model.prev_asst_uttr = assistant_turn.frames[-1].uttr
            self.model.turn_id += 1

            return assistant_turn

        except:
            return None

    def run_loop_command_prompt(self):

        while True:
            print()
            user_query = input(">> Enter your query (or type quit): ")
            if user_query == "quit":
                break

            response = self.execute_turn(user_query=user_query)


if __name__ == "__main__":
    # Define paths
    # path_memory_graph_list = '/Users/shanemoon/workspace/memory_dialog/dialog_simulator/memories/final/mscoco_memory_graphs_1k.json'
    path_memory_graph_list = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/memories/final/mscoco_memory_graphs_mini.json"
    path_model = (
        "/Users/shanemoon/workspace/memory_dialog/models/gpt2_dst/save/model_v2"
    )
    path_parameter_ontology = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/all_parameter_ontology.json"

    # Hyperparameters for the demo
    random_memory_graph = False

    # Load parameters
    memory_graph_list = json.load(open(path_memory_graph_list, "r"))
    memory_graph_bank = {}

    for memory_graph in memory_graph_list:
        memory_graph_id = memory_graph["memory_graph_id"]

        for i in range(len(memory_graph["memories"])):
            memory_graph["memories"][i]["memory_graph_id"] = memory_graph_id

        memory_graph_bank[memory_graph_id] = memory_graph

    parameter_ontology = json.load(open(path_parameter_ontology, "r"))

    # Select a Memory Graph
    if random_memory_graph:
        memory_graph = MemoryGraph(
            data=memory_graph_bank[random.choice(list(memory_graph_bank.keys()))]
        )

    else:
        memory_graph_id = "RbXAfFDz8r72"
        memory_graph = MemoryGraph(data=memory_graph_bank[memory_graph_id])

    # Load the model parameters
    gpt2_model, tokenizer, length = load_model(
        model_type="gpt2", model_name_or_path=path_model, device="cpu", length=150
    )

    # Instsantiate the dialog handler
    model = PilotMemoryDialogModel(
        model=gpt2_model,
        tokenizer=tokenizer,
        length=length,
        parameter_ontology=parameter_ontology,
    )

    api = MemoryServiceAPI()
    dialog_handler = InteractiveDialogHandler(
        model=model, memory_graph=memory_graph, api=api
    )

    # Run loop
    dialog_handler.run_loop_command_prompt()
