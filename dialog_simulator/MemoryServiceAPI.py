# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
import random
from typing import Dict, Tuple
from Data import APIRequest, APIResponse, MemoryTime, MemoryLocation
from constants import API_CALL_TYPE, API_STATUS, GoalType
from utils import str_memory
from datetime import datetime

random.seed(0)


class MemoryServiceAPI:
    def __init__(self, *args, **kwargs):
        self.metadata = kwargs.pop("metadata", {})

    def call_api(self, api_request: APIRequest) -> APIResponse:

        status = None
        results = None

        if api_request.call_type == API_CALL_TYPE.SEARCH:
            results, status = self.search(api_request)

        elif api_request.call_type == API_CALL_TYPE.REFINE_SEARCH:
            results, status = self.refine_search(api_request)

        elif api_request.call_type == API_CALL_TYPE.GET_RELATED:
            results, status = self.get_related(api_request)

        elif api_request.call_type == API_CALL_TYPE.GET_INFO:
            results, status = self.get_info(api_request)

        elif api_request.call_type == API_CALL_TYPE.SHARE:
            results, status = self.share(api_request)

        # Construct a response
        api_response = APIResponse(status=status, results=results, request=api_request)

        return api_response

    def search(self, api_request: APIRequest) -> Tuple[Dict, API_STATUS]:
        # Unpack API Request
        search_filter = api_request.parameters["slot_values"]
        memory_dialog = api_request.memory_dialog

        # Unpack more parameters
        n_max_results = api_request.parameters.get("n_max_results", 2)
        exclude_memory_ids = api_request.parameters.get("exclude_memory_ids", set())

        # Prepare search candidates
        search_candidates = memory_dialog.get_memories()

        # Prepare search output
        retrieved_memories = []

        # Execute search
        for memory in search_candidates:
            # If there was an exlusion request, modify the search candidates
            if int(memory.data["memory_id"]) in exclude_memory_ids:
                continue

            # TODO: ****** implement *****
            meet_criteria = True

            for slot, value in search_filter.items():
                # TODO: handle special cases
                if slot == "time":

                    if search_filter.get("time", None) in {
                        "right before",
                        "right after",
                        "on the same day",
                    }:
                        # This is an error case -- that can happen
                        # due to the wrong model behaviors.
                        print("Wrong request ...")
                        meet_criteria = False
                        break

                    memory_time = MemoryTime(str_datetime=memory.data["time"])
                    search_time = MemoryTime(str_datetime=value)

                    if not memory_time.is_within(search_time):
                        meet_criteria = False
                        break

                elif slot == "location":
                    memory_location = MemoryLocation(data=memory.data["location"])
                    search_location = MemoryLocation(data=value)

                    if not memory_location.is_within(search_location):
                        meet_criteria = False
                        break

                elif slot == "participant":
                    memory_participants = {
                        p["name"] for p in memory.data["participant"]
                    }
                    search_participants = [p["name"] for p in value]

                    for search_participant in search_participants:
                        if search_participant not in memory_participants:
                            meet_criteria = False
                            break

                elif slot == "activity":
                    memory_activities = {
                        a["activity_name"] for a in memory.data["activity"]
                    }
                    search_activities = [a["activity_name"] for a in value]

                    for search_activity in search_activities:
                        if search_activity not in memory_activities:
                            meet_criteria = False
                            break

                else:
                    # General cases
                    if type(memory.data[slot]) == list:
                        pass

                        if value not in memory.data[slot]:
                            meet_criteria = False
                            break

                    else:
                        if value != memory.data[slot]:
                            meet_criteria = False
                            break

            if meet_criteria:
                retrieved_memories.append(memory)

        # ** TODO: check if search_filter and retrieved_memories match **
        # print('=====')
        # print('search_filter', search_filter)
        # print('-----')
        # print('retrieved_memories', retrieved_memories)

        # Rank and return only n_results
        n_results = random.randint(1, n_max_results)

        if len(retrieved_memories) > n_results:
            random.shuffle(retrieved_memories)
            retrieved_memories = retrieved_memories[:n_results]

        # Output
        results = {"retrieved_memories": retrieved_memories}

        if results["retrieved_memories"] != []:
            status = API_STATUS.SEARCH_FOUND
        else:
            status = API_STATUS.SEARCH_NOT_FOUND

        return (results, status)

    def refine_search(self, api_request: APIRequest) -> Tuple[Dict, API_STATUS]:
        # Adjust the search based on the memory_dialog
        memory_dialog = api_request.memory_dialog

        # Search for previous search filter
        prev_filter = None
        for i in reversed(range(len(memory_dialog.dialog.asst_turns))):
            asst_turn = memory_dialog.dialog.asst_turns[i]
            turn_goal = asst_turn.goal

            if turn_goal.goal_type in {GoalType.SEARCH, GoalType.GET_RELATED}:
                # TODO: change it to reflect multi goal parameters
                prev_filter = turn_goal.goal_parameters[0].filter
                break

        # Reconstruct the goal to include the previous search parameters
        if prev_filter is not None:
            search_filter = api_request.parameters["slot_values"]

            # Previous request
            for k, v in prev_filter.items():
                search_filter[k] = v

            # New request
            for k, v in api_request.parameters["slot_values"].items():
                search_filter[k] = v

            api_request.parameters["slot_values"] = search_filter

        else:
            # This dialog is not allowed -- Refine should always
            # happen after a Search or GET_RELATED. Hence abort.
            ### TODO: abort gracefully
            print("***** Refine error *****")
            assert False

        # Exclude memories that are already discussed
        api_request.parameters[
            "exclude_memory_ids"
        ] = memory_dialog.dialog.mentioned_memory_ids

        return self.search(api_request)

    def get_related(self, api_request: APIRequest) -> Tuple[Dict, API_STATUS]:
        # Unpack API Request
        search_filter = api_request.parameters["slot_values"]

        if search_filter.get("time", None) in {
            "right before",
            "right after",
            "on the same day",
        }:
            # This is a special request to retrieve
            # related memories in the same time group (from the same day)
            return self.get_connected(api_request, search_filter.get("time"))

        else:
            # Treat it as a modified search request
            # where slot values are taken from the input memories
            request_slots = api_request.parameters["request_slots"]
            memories = api_request.parameters["memories"]
            memory_dialog = api_request.memory_dialog

            # If request_slots is not specified, randomly sample a few slots
            if request_slots == []:

                request_slot_candidates = {
                    "time",
                    "location",
                    "activity",
                    "participant",
                }

                # If a value is specified for a slot, exclude it
                # from the candidates
                request_slot_candidates -= search_filter.keys()

                request_slots = random.choices(
                    population=list(request_slot_candidates), k=random.randint(1, 1)
                )

            for request_slot in request_slots:
                for memory in memories:
                    request_slot_value = memory.data[request_slot]

                    # TODO: make it take multiple values
                    search_filter[request_slot] = request_slot_value

            # Make a search request with the updated filter
            api_request.parameters["slot_values"] = search_filter

            # Exclude memories that are already discussed
            api_request.parameters[
                "exclude_memory_ids"
            ] = memory_dialog.dialog.mentioned_memory_ids

            return self.search(api_request)

    def get_connected(
        self, api_request: APIRequest, time_constraint: str
    ) -> Tuple[Dict, API_STATUS]:

        _ = api_request.parameters["slot_values"]

        ## TODO: handle multiple memories
        target_memory = api_request.parameters["memories"][0]
        memory_graph = api_request.memory_dialog.memory_graph
        target_memory_index = -1
        for i, memory in enumerate(memory_graph.memories):
            if memory.data["memory_id"] == target_memory.data["memory_id"]:
                target_memory_index = i
                break

        # Get connected memories
        connected_memory_indices = memory_graph.get_events(target_memory_index)[
            "memories"
        ]
        connected_memories = []

        # Compare time
        target_time = datetime.fromisoformat(target_memory.data["time"])

        for idx in connected_memory_indices:
            if idx == target_memory_index:
                continue

            connected_memory = memory_graph.memories[idx]
            connected_memory_time = datetime.fromisoformat(
                connected_memory.data["time"]
            )

            if time_constraint == "right after":
                if target_time < connected_memory_time:
                    connected_memories.append(connected_memory)
            elif time_constraint == "right before":
                if target_time > connected_memory_time:
                    connected_memories.append(connected_memory)
            elif time_constraint == "on the same day":
                connected_memories.append(connected_memory)

        # Output
        results = {"retrieved_memories": connected_memories}

        if results["retrieved_memories"] != []:
            status = API_STATUS.SEARCH_FOUND
        else:
            status = API_STATUS.SEARCH_NOT_FOUND

        return (results, status)

    def get_info(self, api_request: APIRequest) -> Tuple[Dict, API_STATUS]:
        # Unpack API Request
        request_slots = api_request.parameters.get("request_slots", [])
        memories = api_request.parameters.get("memories", [])

        # Unpack more parameters
        # TODO

        # Prepare get_info output
        lookup_results = {
            "retrieved_memories": memories,
            "retrieved_info": {},
            "request_slots": request_slots,
        }

        # If request_slots is not specified, randomly sample a few slots
        if request_slots == []:

            if len(memories) > 0:
                memory = memories[0]
                request_slots = [k for k in memory.data if random.random() > 0.8]

        def summarize_info(memory_data, slot):

            if slot == "location":
                return memory_data[slot]["geo_tag"]

            else:
                return memory_data[slot]

        # Look up info
        for memory in memories:

            # Add the requested info
            s_memory = str_memory(memory, verbose=False)

            if request_slots == []:
                # Give all relevant information
                lookup_results["retrieved_info"][s_memory] = {
                    slot: summarize_info(memory.data, slot)
                    for slot in ["time", "location", "participant", "activity"]
                }

            else:
                lookup_results["retrieved_info"][s_memory] = {}
                for slot in request_slots:
                    if slot in memory.data:
                        lookup_results["retrieved_info"][s_memory][
                            slot
                        ] = summarize_info(memory.data, slot)

                # Add extra info
                # TODO

        # TODO: status can be INFO_NOT_FOUND
        status = API_STATUS.INFO_FOUND

        return (lookup_results, status)

    def share(self, api_request) -> Tuple[Dict, API_STATUS]:
        # Unpack API Request
        memories = api_request.parameters["memories"]

        # Prepare output
        results = {"retrieved_memories": memories}

        status = API_STATUS.SHARED

        return (results, status)
