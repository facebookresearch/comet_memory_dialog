# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
import json, random, traceback, os
from typing import List, Tuple
from constants import TurnSpeaker, DialogAct, API_STATUS
from Data import Dialog, MemoryDialog, MemoryGraph, Turn, Goal
from UserSimulator import PilotUserSimulator
from AssistantSimulator import PilotAssistantSimulator
from GoalGenerator import RuleBasedGoalGenerator
from MemoryServiceAPI import MemoryServiceAPI
from utils import build_parameter_ontology

random.seed(0)


class MemoryDialogSimulator:
    def __init__(self, *args, **kwargs):
        # Initialize user simulator, assistant simulator, memory_graphs etc.
        self.domain = kwargs.pop("domain")
        self._memory_service_api = kwargs.pop("memory_service_api", MemoryServiceAPI())
        self._user_simulator = kwargs.pop("user_simulator", PilotUserSimulator())
        self._assistant_simulator = kwargs.pop(
            "assistant_simulator", PilotAssistantSimulator()
        )
        self._goal_generator = kwargs.pop(
            "goal_generator", RuleBasedGoalGenerator(domain=self.domain)
        )
        self._memory_graph_bank = kwargs.pop("memory_graph_bank", {})

        self._user_simulator.register_memory_service_api(self._memory_service_api)
        self._assistant_simulator.register_memory_service_api(self._memory_service_api)

    def set_user_simulator(self, user_simulator):
        self._user_simulator = user_simulator

    def set_assistant_simulator(self, assistant_simulator):
        self._assistant_simulator = assistant_simulator

    def set_goal_generator(self, goal_generator):
        self._goal_generator = goal_generator

    def set_memory_service_api(self, memory_service_api):
        self._memory_service_api = memory_service_api

    def sample_goals(self, memory_graph, goal_config) -> List[Goal]:
        return self._goal_generator.sample_goals(
            memory_graph=memory_graph, goal_config=goal_config
        )

    def sample_memory_graph(self) -> MemoryGraph:
        if self._memory_graph_bank == {}:
            # Empty memory graph
            return MemoryGraph()

        # Randomly sample a memory
        # TODO: allow for more organized way of sampling memories
        memory_graph_id = random.choice(list(self._memory_graph_bank.keys()))
        memory_graph = self._memory_graph_bank[memory_graph_id]

        return MemoryGraph(data=memory_graph)

    def batch_generate_dialog_flows(
        self,
        n_dialogs: int,
        n_max_turns: int,
        start_dialog_idx: int,
        goal_config: dict = {},
    ) -> List[MemoryGraph]:

        # Batch generate multiple dialogs using the same simulators
        memory_dialogs = []

        for i in range(n_dialogs):
            # Continue until generation is successful
            generation_success = False

            while not generation_success:
                try:
                    # Sample a memory graph (user)
                    memory_graph = self.sample_memory_graph()

                    # Create an empty memory dialog
                    memory_dialog = MemoryDialog(memory_graph=memory_graph)

                    # Generate Goal Config
                    goal_config["parameter_ontology"] = build_parameter_ontology(
                        memory_dialog.memory_graph,
                        self._memory_service_api.metadata,
                        self.domain,
                    )

                    # Sample goals for this dialog
                    goals = self.sample_goals(
                        memory_graph=memory_dialog.memory_graph, goal_config=goal_config
                    )

                    # Generate dialog flow
                    memory_dialog = self.generate_dialog_flow(
                        goals, memory_dialog, n_max_turns
                    )
                    memory_dialog.dialog.idx = start_dialog_idx + i

                    # If everything is successful, append to memory_dialogs
                    generation_success = True
                    memory_dialogs.append(memory_dialog)

                except:
                    # TODO: Make a more robust abort strategy
                    print("** Error in generating dialog. Ignoring this one. **")
                    traceback.print_exc()
                    print()

        return memory_dialogs

    def generate_dialog_flow(
        self,
        goals: List[Goal],
        memory_dialog: MemoryDialog,
        n_max_turns: int,
        initialize=True,
    ) -> MemoryDialog:

        if initialize:
            # Initialize memory_dialog
            memory_dialog.initialize()

        # Iterate and generate a dialog turn by turn
        i = 0
        while not goals == [] and i < n_max_turns:

            # Pick a goal
            current_goal = goals.pop(0)
            goal_met = False
            print("Goal:", current_goal)

            while not goal_met and i < n_max_turns:

                # Generate a turn
                memory_dialog = self.generate_turn(current_goal, memory_dialog)

                # End of a turn: update dialog & goals
                i += 1
                goal_met = memory_dialog.is_goal_met(current_goal)

        is_valid_dialog = self.validate_dialog(memory_dialog)
        if not is_valid_dialog:
            # If something is not right about this dialog, abort.
            # TODO: abort gracefully
            assert False

        return memory_dialog

    def generate_turn(self, goal: Goal, memory_dialog: MemoryDialog) -> MemoryDialog:

        # TODO: extend it for multiple frames per turn

        # (1) Generate a User turn, given a target goal and a memory_dialog
        # Generate dialog act and slots
        user_frame = self._user_simulator.execute_turn(goal, memory_dialog)

        # Template based utterance generation
        user_frame = self._user_simulator.generate_uttr(user_frame, goal)

        # Instantiate a user turn, and update the memory_dialog
        user_turn = Turn([user_frame], TurnSpeaker.USER, goal)
        memory_dialog.dialog.add_user_turn(user_turn)
        print("U:", user_turn)

        # (2) Generate a Assistant turn, given a target goal and a memory_dialog
        # Generate dialog act and slots
        asst_frame, api_request, api_result = self._assistant_simulator.execute_turn(
            goal, memory_dialog
        )

        # Template based utterance generation
        asst_frame = self._assistant_simulator.generate_uttr(asst_frame, goal)

        # Instantiate a user turn, and update the memory_dialog
        asst_turn = Turn([asst_frame], TurnSpeaker.ASSISTANT, goal)
        memory_dialog.dialog.add_asst_turn(asst_turn)
        print("A:", asst_turn)

        # Add goals and api_calls
        memory_dialog.dialog.add_goal(goal)
        memory_dialog.dialog.add_api_call(api_request)
        memory_dialog.dialog.add_api_result(api_result)

        return memory_dialog

    def validate_dialog(self, memory_dialog: MemoryDialog) -> bool:
        # Check for any undesirable traits of a dialog
        n_turns = len(memory_dialog.dialog.asst_turns)

        # (1) Multiple sharing of the same memory
        set_shared_memory_ids = set()
        for user_turn in memory_dialog.dialog.user_turns:
            # TODO: Handle multiple frames per turn
            dialog_act = user_turn.frames[-1].dialog_act

            if dialog_act == DialogAct.REQUEST_SHARE:
                memories_to_share = user_turn.frames[-1].act_attributes.memories
                for m in memories_to_share:
                    memory_id = m.data["memory_id"]
                    if memory_id in set_shared_memory_ids:
                        # If this memory_id is already shared, abort
                        return False
                    set_shared_memory_ids.add(memory_id)

        # (2) Too frequent search fails
        n_search_fails = 0
        for api_result in memory_dialog.dialog.api_results:
            status = api_result.status
            if status == API_STATUS.SEARCH_NOT_FOUND:
                n_search_fails += 1

            if (n_turns <= 4 and n_search_fails >= 2) or (
                n_turns > 4 and n_search_fails >= 3
            ):
                return False

        # Otherwise, this dialog is good.
        return True
