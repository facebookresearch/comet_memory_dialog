# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
import random
import numpy as np
from typing import List, Tuple
from SimulatorBase import SimulatorBase
from constants import GoalType, DialogAct, GoalMemoryRefType
from Data import MemoryDialog, Goal, GoalParameter, Frame, ActAttributes, APIResponse
from MemoryServiceAPI import MemoryServiceAPI
from utils import (
    str_slot_values,
    str_request_slots,
    str_memories,
    get_template,
    get_slot_values_simple_from_json,
)

random.seed(0)


class UserSimulator(SimulatorBase):
    def __init__(self, *args, **kwargs):
        super(UserSimulator, self).__init__(*args, **kwargs)
        self.memory_service_api = None


class ModelBasedUserSimulator(UserSimulator):
    def __init__(self, *args, **kwargs):
        super(ModelBasedUserSimulator, self).__init__(*args, **kwargs)

    def fit_goal_to_intent(self, args):
        # Define the goal to intent mapping behavior
        pass

    def is_servable(self, goal: Goal) -> bool:
        # Check whether this simulator can serve the input goal.
        pass

    def execute_turn(
        self,
        goal: Goal,
        memory_dialog: MemoryDialog,
        memory_service_api: MemoryServiceAPI,
    ) -> Frame:
        # Need to define this behavior e.g. as a config, a model, etc.
        pass

    def generate_uttr(self, frame: Frame, goal: Goal) -> str:
        pass


class RuleBasedUserSimulator(UserSimulator):
    def __init__(self, *args, **kwargs):
        super(RuleBasedUserSimulator, self).__init__(*args, **kwargs)

    def fit_goal_to_intent(self, args):
        # Define the goal to intent mapping behavior
        pass

    def is_servable(self, goal: Goal) -> bool:
        # Check whether this simulator can serve the input goal.
        pass

    def execute_turn(
        self,
        goal: Goal,
        memory_dialog: MemoryDialog,
        memory_service_api: MemoryServiceAPI,
    ) -> Frame:

        # Need to define this behavior e.g. as a config, a model, etc.
        pass


class HybridUserSimulator(UserSimulator):
    def is_servable(self, goal: Goal) -> bool:
        # Check whether this simulator can serve the input goal.
        pass

    def execute_turn(
        self,
        goal: Goal,
        memory_dialog: MemoryDialog,
        memory_service_api: MemoryServiceAPI,
    ) -> Frame:

        # If a Goal is servable by the model based simulator,
        # generate with a model based simulator first.
        # Otherwise resort to the predefined rules.
        pass

    def generate_uttr(self, frame: Frame, goal: Goal) -> str:
        pass


class PilotUserSimulator(UserSimulator):
    """
    Includes the simplest implementation of a UserSimulator.
    Use this class as a guide for implementing more complex
    simulators.
    """

    def __init__(self, *args, **kwargs):

        super(PilotUserSimulator, self).__init__(*args, **kwargs)

        # Simple interaction deterministic mapping
        self._goal_to_handler = {
            GoalType.UNKNOWN: self.UserGoalHandler(),
            GoalType.SEARCH: self.UserSearchGoalHandler(),
            GoalType.REFINE_SEARCH: self.UserRefineSearchGoalHandler(),
            GoalType.GET_RELATED: self.UserGetRelatedGoalHandler(),
            GoalType.GET_INFO: self.UserGetInfoGoalHandler(),
            # GoalType.GET_AggregatedINFO: self.UserGetAggregatedInfoGoalHandler(),
            GoalType.SHARE: self.UserShareGoalHandler(),
        }

    def is_servable(self, goal: Goal) -> bool:
        # Check whether this simulator can serve the input goal.
        return True

    def execute_turn(self, goal: Goal, memory_dialog: MemoryDialog) -> Frame:

        handler = self._goal_to_handler[goal.goal_type]
        return handler.execute_turn(goal, memory_dialog, self.memory_service_api)

    def generate_uttr(self, frame: Frame, goal: Goal) -> Frame:

        handler = self._goal_to_handler[goal.goal_type]
        uttr = handler.generate_uttr(frame, goal, self.memory_service_api)
        frame.uttr = uttr
        return frame

    class UserGoalHandler:
        def __init__(self, *args, **kwargs):
            self.available_user_main_acts = [
                DialogAct.UNKNOWN,
            ]

            self.available_user_disambiguation_acts = [DialogAct.INFORM_DISAMBIGUATE]

            self._uttr_template_disambiguate_memories = {
                DialogAct.INFORM_DISAMBIGUATE: ["I mean these ones: {memories}"],
            }

        def execute_turn(
            self,
            goal: Goal,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ) -> Frame:

            if len(memory_dialog.dialog.asst_turns) > 0:
                last_asst_turn = memory_dialog.dialog.asst_turns[-1]
            else:
                last_asst_turn = None

            if last_asst_turn is None or (
                not last_asst_turn.is_disambiguation_request()
            ):

                # 1. User does a main act according to the Goal
                if True:
                    # 1. (1) Main Act
                    # Get a random dialog act label
                    user_dialog_act = random.choice(self.available_user_main_acts)

                    # Randomly fill the act_attributes
                    list_act_attributes = []

                    for goal_parameter in goal.goal_parameters:

                        act_attributes = ActAttributes(
                            slot_values=self.get_slot_values(goal_parameter),
                            slot_values_resolved=self.get_slot_values_resolved(
                                goal_parameter
                            ),
                            request_slots=self.get_request_slots(goal_parameter),
                            memories=self.get_memories(
                                goal.goal_type,
                                goal_parameter,
                                memory_dialog,
                                memory_service_api,
                            ),
                        )
                        list_act_attributes.append(act_attributes)

                else:
                    # 1. (2) Answer follow-up questions
                    # TODO
                    pass

            else:
                # 2. Answer disambiguation request
                user_dialog_act, list_act_attributes = self.disambiguate_last_turn(
                    memory_dialog
                )

            # Return an Frame memory with the generated intent and attributes
            # TODO: handle multiple goal parameters & multiple acts
            return Frame("", user_dialog_act, list_act_attributes[0])

        def get_slot_values(self, goal_parameter: GoalParameter):
            return get_slot_values_simple_from_json(goal_parameter.filter)

        def get_slot_values_resolved(self, goal_parameter: GoalParameter):
            # return {k: str(v) for k, v in goal_parameter.filter.items()}
            return goal_parameter.filter

        def get_request_slots(self, goal_parameter: GoalParameter):
            return goal_parameter.request_slots

        def get_memories(
            self,
            goal_type: GoalType,
            goal_parameter: GoalParameter,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ) -> List:

            return get_memories(
                goal_type,
                goal_parameter,
                memory_dialog,
                memory_service_api,
                n_min_memories=0,
                n_max_memories=2,
            )

        def generate_uttr(
            self, frame: Frame, goal: Goal, memory_service_api: MemoryServiceAPI
        ) -> str:

            if frame.dialog_act in set([DialogAct.INFORM_DISAMBIGUATE]):
                template = get_template(
                    self._uttr_template_disambiguate_memories, frame
                )
                return template.format(
                    memories=str_memories(
                        frame.act_attributes.memories, memory_service_api, verbose=False
                    )
                )

            else:
                return self.generate_uttr_main(frame, goal, memory_service_api)

        def generate_uttr_main(
            self, frame: Frame, goal: Goal, memory_service_api: MemoryServiceAPI
        ) -> str:
            template = get_template(self._uttr_template, frame)

            uttr = template.format(
                search_filter=str_slot_values(frame.act_attributes.slot_values),
                request_slots=str_request_slots(frame.act_attributes.request_slots),
                memories=str_memories(
                    frame.act_attributes.memories, memory_service_api, verbose=False
                ),
            )
            return uttr

        def disambiguate_last_turn(self, memory_dialog: MemoryDialog):
            # TODO: Make it more robust
            user_dialog_act = random.choice(self.available_user_disambiguation_acts)

            assert len(memory_dialog.dialog.user_turns) > 0

            # **** TODO **** : handle multiple goal parameters & multiple acts
            # **** TODO 8*** : pick the right frame instead of choosing the last frame
            list_act_attributes = [
                memory_dialog.dialog.user_turns[-1].frames[-1].act_attributes
            ]

            return user_dialog_act, list_act_attributes

    class UserSearchGoalHandler(UserGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.available_user_main_acts = [
                DialogAct.REQUEST_GET,
            ]

            self._uttr_template = {
                DialogAct.REQUEST_GET: [
                    "Show me photos.",
                    "I am looking for some photos.",
                ],
            }

            self._uttr_template_s = {
                DialogAct.REQUEST_GET: [
                    "Show me photos with {search_filter}.",
                    "I am looking for some photos with {search_filter}.",
                ],
            }

        def get_request_slots(self, goal_parameter: GoalParameter):
            return []

        def get_memories(
            self,
            goal_type: GoalType,
            goal_parameter: GoalParameter,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ):
            return []

        def generate_uttr_main(
            self, frame: Frame, goal: Goal, memory_service_api: MemoryServiceAPI
        ) -> str:

            search_filter = frame.act_attributes.slot_values
            if search_filter == {}:
                template = get_template(self._uttr_template, frame)
            else:
                template = get_template(self._uttr_template_s, frame)

            uttr = template.format(search_filter=str_slot_values(search_filter))

            return uttr

    class UserRefineSearchGoalHandler(UserGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.available_user_main_acts = [
                DialogAct.INFORM_REFINE,
            ]

            self._uttr_template = {
                DialogAct.INFORM_REFINE: [
                    "Do I have any other photos?",
                    "Are there any other photos?",
                ],
            }

            self._uttr_template_s = {
                DialogAct.INFORM_REFINE: [
                    "I would like to refine/change my search to include {search_filter}.",
                    "Refine/change my search to include {search_filter}.",
                    "Do I have any other photos that also include {search_filter}?",
                ],
            }

        def get_slot_values(self, goal_parameter: GoalParameter):
            # TODO: Need to account for invalid refine, e.g. looking for wooden area rugs
            return get_slot_values_simple_from_json(goal_parameter.filter)

        def get_request_slots(self, goal_parameter: GoalParameter):
            return []

        def get_memories(
            self,
            goal_type: GoalType,
            goal_parameter: GoalParameter,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ):
            return []

        def generate_uttr_main(
            self, frame: Frame, goal: Goal, memory_service_api: MemoryServiceAPI
        ) -> str:

            search_filter = frame.act_attributes.slot_values

            if len(search_filter) > 0:
                template = get_template(self._uttr_template_s, frame)

            elif len(search_filter) == 0:
                template = get_template(self._uttr_template, frame)

            else:
                print("This should not happen")

            uttr = template.format(
                search_filter=str_slot_values(frame.act_attributes.slot_values),
                request_slots=str_request_slots(frame.act_attributes.request_slots),
                memories=str_memories(
                    frame.act_attributes.memories, memory_service_api, verbose=False
                ),
            )

            return uttr

    class UserGetRelatedGoalHandler(UserGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.available_user_main_acts = [
                DialogAct.INFORM_GET,
            ]

            self._uttr_template_o = {
                DialogAct.INFORM_GET: [
                    "I would like to see something similar/related to {memories}.",
                    "Is there anything related to {memories}.",
                    "Is there any other photo/video related to {memories}.",
                    "Do I have any other photos/videos similar/related to {memories}?",
                    "Could you show me any other photos/videos like {memories}?",
                    "Show me other photos/videos like {memories}.",
                ]
            }

            self._uttr_template_or = {
                DialogAct.INFORM_GET: [
                    "I would like to see something related to {memories} with the similar/same {request_slots}.",
                    "Is there anything related to {memories} with the similar/same {request_slots}.",
                    "Is there any other photo/video related to {memories} with the similar/same {request_slots}.",
                    "Do I have any other photo/video like {memories} with the similar/same {request_slots}?",
                    "Could you show me any other photo/video related to {memories} with the similar/same {request_slots}?",
                    "Show me other photos/videos like {memories} with the similar/same {request_slots}?",
                ]
            }

            self._uttr_template_os = {
                DialogAct.INFORM_GET: [
                    "I would like to see something related to {memories}, and/but with {search_filter}.",
                    "Is there anything related to {memories}, and/but with {search_filter}.",
                    "Is there any other photo/video related to {memories}, and/but with {search_filter}.",
                    "Do I have any other photo/video like {memories} , and/but with {search_filter}?",
                    "Could you show me any other photo/video related to {memories}, and/but with {search_filter}?",
                    "Show me other photos/videos like {memories}, and/but with {search_filter}.",
                ]
            }

            self._uttr_template_ors = {
                DialogAct.INFORM_GET: [
                    "I would like to see something related "
                    "to {memories} on {request_slots}, but with {search_filter}.",
                    "Is there anything related "
                    "to {memories} on {request_slots}, but with {search_filter}.",
                    "Show me something like "
                    "{memories} on paremters: {request_slots}, but with {search_filter}.",
                    "Do I have any photos/videos like "
                    "{memories} on paremters: {request_slots}, but with {search_filter}?",
                ]
            }

        def generate_uttr_main(
            self, frame: Frame, goal: Goal, memory_service_api: MemoryServiceAPI
        ) -> str:

            search_filter = frame.act_attributes.slot_values
            request_slots = frame.act_attributes.request_slots
            memories = frame.act_attributes.memories

            if len(request_slots) > 0 and len(search_filter) > 0:
                template = get_template(self._uttr_template_ors, frame)

            elif len(request_slots) > 0 and len(search_filter) == 0:
                template = get_template(self._uttr_template_or, frame)

            elif len(request_slots) == 0 and len(search_filter) > 0:
                template = get_template(self._uttr_template_os, frame)

            elif len(request_slots) == 0 and len(search_filter) == 0:
                template = get_template(self._uttr_template_o, frame)

            else:
                print("This should not happen")

            uttr = template.format(
                search_filter=str_slot_values(frame.act_attributes.slot_values),
                request_slots=str_request_slots(frame.act_attributes.request_slots),
                memories=str_memories(
                    frame.act_attributes.memories, memory_service_api, verbose=False
                ),
            )

            return uttr

        def get_memories(
            self,
            goal_type: GoalType,
            goal_parameter: GoalParameter,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ):

            return get_memories(
                goal_type,
                goal_parameter,
                memory_dialog,
                memory_service_api,
                n_min_memories=1,
                n_max_memories=1,
            )

    class UserGetInfoGoalHandler(UserGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.available_user_main_acts = [
                DialogAct.ASK_GET,
            ]

            self._uttr_template = {
                DialogAct.ASK_GET: [
                    "Can I get {request_slots} of {memories}?",
                    "Do you know {request_slots} of {memories}?",
                    "(Who/where/when/what/...) {request_slots} of {memories}?",
                ],
            }

        def get_slot_values(self, goal_parameter: GoalParameter):
            return {}

        def get_memories(
            self,
            goal_type: GoalType,
            goal_parameter: GoalParameter,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ):

            n_max_memories = 2 if random.random() > 0.9 else 1

            return get_memories(
                goal_type,
                goal_parameter,
                memory_dialog,
                memory_service_api,
                n_min_memories=1,
                n_max_memories=n_max_memories,
            )

    class UserCompareGoalHandler(UserGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.available_user_main_acts = [
                DialogAct.REQUEST_COMPARE,
            ]

            self._uttr_template_o = {
                DialogAct.REQUEST_COMPARE: [
                    "How do they compare: {memories}?",
                ]
            }

            self._uttr_template_or = {
                DialogAct.REQUEST_COMPARE: [
                    "How do they compare on {request_slots}: {memories}?"
                ]
            }

        def get_slot_values(self, goal_parameter: GoalParameter):
            return {}

        def get_memories(
            self,
            goal_type: GoalType,
            goal_parameter: GoalParameter,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ):

            return get_memories(
                goal_type,
                goal_parameter,
                memory_dialog,
                memory_service_api,
                n_min_memories=2,
                n_max_memories=2,
            )

        def generate_uttr_main(
            self, frame: Frame, goal: Goal, memory_service_api: MemoryServiceAPI
        ) -> str:

            request_slots = frame.act_attributes.request_slots
            memories = frame.act_attributes.memories

            if len(request_slots) > 0:
                template = get_template(self._uttr_template_or, frame)

            else:
                template = get_template(self._uttr_template_o, frame)

            uttr = template.format(
                request_slots=str_request_slots(frame.act_attributes.request_slots),
                memories=str_memories(
                    frame.act_attributes.memories, memory_service_api, verbose=False
                ),
            )

            return uttr

    class UserShareGoalHandler(UserGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.available_user_main_acts = [
                DialogAct.REQUEST_SHARE,
            ]

            self._uttr_template = {
                DialogAct.REQUEST_SHARE: [
                    "Please share: {memories}.",
                    "Could you please share: {memories}?",
                    "I like these: {memories} - could you please share them.",
                    "Love these photos: {memories} - please share them.",
                ]
            }

        def get_request_slots(self, goal_parameter: GoalParameter):
            return []

        def get_slot_values(self, goal_parameter: GoalParameter):
            return {}

        def get_memories(
            self,
            goal_type: GoalType,
            goal_parameter: GoalParameter,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ):

            # Need to pick from previous turns
            n_max_memories = 2 if random.random() > 0.7 else 1

            return get_memories(
                goal_type,
                goal_parameter,
                memory_dialog,
                memory_service_api,
                n_min_memories=1,
                n_max_memories=n_max_memories,
            )


def get_memories(
    goal_type: GoalType,
    goal_parameter: GoalParameter,
    memory_dialog: MemoryDialog,
    memory_service_api: MemoryServiceAPI,
    n_min_memories=0,
    n_max_memories=2,
) -> List:

    # TODO: implement
    n_memories = random.randint(n_min_memories, n_max_memories)
    candidate_memories = []

    # (1) Determine where to choose the memory from
    if goal_parameter.reference_type == GoalMemoryRefType.PREV_TURN:
        # Candidate memories are from the immediate previous turn
        # TODO: add a more robust report-abort if candidates are empty
        # ** TODO ** : pick the right frame instead of just the last one
        candidate_memories.extend(
            memory_dialog.dialog.asst_turns[-1].frames[-1].act_attributes.memories
        )

    elif goal_parameter.reference_type == GoalMemoryRefType.DIALOG:
        # Candidate memories are anywhere from the previous dialog
        # TODO: add a more robust report-abort if candidates are empty
        # ** TODO ** : pick the right frame instead of just the last one
        for turn in memory_dialog.dialog.asst_turns + memory_dialog.dialog.user_turns:
            candidate_memories.extend(turn.frames[-1].act_attributes.memories)

    elif goal_parameter.reference_type == GoalMemoryRefType.GRAPH:
        # Candidate memories are anywhere from the scene
        candidate_memories = memory_dialog.memory_graph.get_memories()

    else:
        print("Object reference not specified")
        pass

    # (2) Weighted sampling: favor the ones that are talked the most
    memory_id_to_memory_dedup = {}
    memory_id_to_count = {}

    for memory in candidate_memories:
        memory_id = memory.data["memory_id"]

        # Count
        memory_id_to_count[memory_id] = memory_id_to_count.get(memory_id, 0.0) + 1

        # Dedup for each memory_id
        if memory_id not in memory_id_to_memory_dedup:
            memory_id_to_memory_dedup[memory_id] = memory
        else:
            pass

    candidate_memories_dedup = []
    candidate_memories_p = []
    sum_counts = sum([c for c in memory_id_to_count.values()])
    sum_counts = 1.0 if sum_counts == 0 else sum_counts

    for memory_id in memory_id_to_count:
        candidate_memories_dedup.append(memory_id_to_memory_dedup[memory_id])

        candidate_memories_p.append(memory_id_to_count[memory_id] / sum_counts)

    return np.random.choice(
        candidate_memories_dedup, p=candidate_memories_p, size=n_memories, replace=False
    )

    return candidate_memories

    """
    
    # e.g. COMPARE / GET_RELATED / GET_INFO should be used only
    # among memories with the same type
    if goal_type in \
        set([GoalType.COMPARE, GoalType.GET_RELATED, GoalType.GET_INFO]):
        
        memory_types = []
        for candidate_memory in candidate_memories:
            prefab_path = candidate_memory['prefab_path']
            obj_metadata = memory_service_api.lookup(prefab_path)
            memory_types.append(obj_metadata['type'])
        
        target_memory_type = random.choice(memory_types)
        candidate_memories = [
            o for o in candidate_memories \
                if memory_service_api.lookup(o['prefab_path'])['type'] == target_memory_type
        ]
    """
