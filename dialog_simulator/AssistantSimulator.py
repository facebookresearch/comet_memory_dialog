# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
import random
from typing import List, Tuple
from SimulatorBase import SimulatorBase
from constants import GoalType, DialogAct, API_STATUS, API_CALL_TYPE
from Data import (
    MemoryDialog,
    Goal,
    Frame,
    ActAttributes,
    APIRequest,
    APIResponse,
    GoalParameter,
)
from MemoryServiceAPI import MemoryServiceAPI
from utils import str_slot_values, str_request_slots, str_memories, get_template

random.seed(0)


class AssistantSimulator(SimulatorBase):
    def __init__(self, *args, **kwargs):
        super(AssistantSimulator, self).__init__(*args, **kwargs)
        self.memory_service_api = None


class ModelBasedAssistantSimulator(AssistantSimulator):
    def __init__(self, *args, **kwargs):
        super(ModelBasedAssistantSimulator, self).__init__(*args, **kwargs)

    def fit_goal_to_intent(self, args):
        # Define the goal to intent mapping behavior
        pass

    def is_servable(self, goal: Goal) -> bool:
        # Check whether this simulator can serve the input goal.
        pass

    def execute_turn(
        self, goal: Goal, memory_dialog: MemoryDialog
    ) -> Tuple[Frame, APIRequest, APIResponse]:
        # Need to define this behavior e.g. as a config, a model, etc.
        pass

    def generate_uttr(self, frame: Frame, goal: Goal) -> str:
        pass


class RuleBasedAssistantSimulator(AssistantSimulator):
    def __init__(self, *args, **kwargs):
        super(RuleBasedAssistantSimulator, self).__init__(*args, **kwargs)

    def fit_goal_to_intent(self, args):
        # Define the goal to intent mapping behavior
        pass

    def is_servable(self, goal: Goal) -> bool:
        # Check whether this simulator can serve the input goal.
        pass

    def execute_turn(
        self, goal: Goal, memory_dialog: MemoryDialog
    ) -> Tuple[Frame, APIRequest, APIResponse]:
        # Need to define this behavior e.g. as a config, a model, etc.
        pass

    def generate_uttr(self, frame: Frame, goal: Goal) -> str:
        pass


class PilotAssistantSimulator(AssistantSimulator):
    """
    Includes the simplest implementation of a AssistantSimulator.
    Use this class as a guide for implementing more complex
    simulators.
    """

    def __init__(self, *args, **kwargs):
        super(PilotAssistantSimulator, self).__init__(*args, **kwargs)

        # Simple interaction deterministic mapping
        self._goal_to_handler = {
            GoalType.UNKNOWN: self.AssistantGoalHandler(),
            GoalType.SEARCH: self.AssistantSearchGoalHandler(),
            GoalType.REFINE_SEARCH: self.AssistantRefineSearchGoalHandler(),
            GoalType.GET_RELATED: self.AssistantGetRelatedGoalHandler(),
            GoalType.GET_INFO: self.AssistantGetInfoGoalHandler(),
            # GoalType.GET_AGGREGATED_INFO: self.AssistantGetAggregatedInfoGoalHandler(),
            GoalType.SHARE: self.AssistantShareGoalHandler(),
        }

    def is_servable(self, goal: Goal) -> bool:
        # Check whether this simulator can serve the input goal.
        return True

    def execute_turn(
        self, goal: Goal, memory_dialog: MemoryDialog
    ) -> Tuple[Frame, APIRequest, APIResponse]:

        handler = self._goal_to_handler[goal.goal_type]
        return handler.execute_turn(goal, memory_dialog, self.memory_service_api)

    def generate_uttr(self, frame: Frame, goal: Goal) -> Frame:

        handler = self._goal_to_handler[goal.goal_type]
        uttr = handler.generate_uttr(frame, goal, self.memory_service_api)
        frame.uttr = uttr
        return frame

    class AssistantGoalHandler:
        def execute_turn(
            self,
            goal: Goal,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ) -> Tuple[Frame, APIRequest, APIResponse]:
            return (
                Frame("", DialogAct.UNKNOWN, ActAttributes()),
                APIRequest(),
                APIResponse(),
            )

        def generate_uttr(
            self, frame: Frame, goal: Goal, memory_service_api: MemoryServiceAPI
        ) -> str:

            template = get_template(self._uttr_template, frame)
            verbose_memory = True if random.random() < 0.35 else False

            uttr = template.format(
                slot_values=str_slot_values(frame.act_attributes.slot_values),
                request_slots=str_request_slots(frame.act_attributes.request_slots),
                memories=str_memories(
                    frame.act_attributes.memories,
                    memory_service_api,
                    verbose=verbose_memory,
                ),
            )

            return uttr

    class AssistantSearchGoalHandler(AssistantGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.available_asst_acts = [
                DialogAct.INFORM_GET,
                # DialogAct.PROMPT_REFINE
            ]

            self.user_search_acts = set(
                [DialogAct.REQUEST_GET, DialogAct.INFORM_REFINE, DialogAct.INFORM_GET]
            )

            self.asst_search_acts = set(
                [
                    DialogAct.INFORM_GET,
                ]
            )

            self._uttr_template = {
                DialogAct.INFORM_GET: [
                    "Here is what I found: {memories}.",
                    "Check out these photos: (summarize) {memories}.",
                    "How is what I found: {memories}. They match some of the criteria: {slot_values}.",
                    "I found these photos: {memories}.",
                    "Here is what I found: {memories}. [[ Please comment on the retrieved photos. ]]",
                    "Here is what I found: {memories}. [[ Briefly summarize what is visible in the photos. ]]",
                ]
            }

            self._uttr_template_no_results = {
                DialogAct.INFORM_GET: [
                    "Sorry, I could not find any photo/video for {slot_values}.",
                    "Sorry, I could not find any photo/video.",
                    "I could not find any photo that matches the criteria {slot_values}.",
                ]
            }

        def generate_uttr(
            self, frame: Frame, goal: Goal, memory_service_api: MemoryServiceAPI
        ) -> str:

            memories = frame.act_attributes.memories

            if len(memories) > 0:
                template = get_template(self._uttr_template, frame)

            else:
                template = get_template(self._uttr_template_no_results, frame)

            verbose_memory = True if random.random() < 0.35 else False

            uttr = template.format(
                slot_values=str_slot_values(frame.act_attributes.slot_values),
                request_slots=str_request_slots(frame.act_attributes.request_slots),
                memories=str_memories(
                    frame.act_attributes.memories,
                    memory_service_api,
                    verbose=verbose_memory,
                ),
            )

            return uttr

        def execute_turn(
            self,
            goal: Goal,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ) -> Tuple[Frame, APIRequest, APIResponse]:

            assert len(memory_dialog.dialog.user_turns) > 0
            last_user_turn = memory_dialog.dialog.user_turns[-1]

            # Check the routing logic here
            if last_user_turn.has_dialog_acts(self.user_search_acts):

                # 1. User requests SEARCH with parameters
                # Get a random dialog act label
                asst_dialog_act = random.choice(self.available_asst_acts)
                api_response = APIResponse()
                api_request = APIRequest()

                if asst_dialog_act in self.asst_search_acts:
                    # 1. (1) Return Search results
                    # Randomly fill the act_attributes
                    list_act_attributes = []

                    for goal_parameter in goal.goal_parameters:

                        # Construct an API request
                        # ** TODO **: grab the correct frame, instead of the last frame
                        requested_act_attributes = last_user_turn.frames[
                            -1
                        ].act_attributes
                        api_parameters = {
                            "slot_values": requested_act_attributes.slot_values_resolved
                        }

                        if goal_parameter.request_slots != []:
                            api_parameters[
                                "request_slots"
                            ] = goal_parameter.request_slots

                        call_type = None
                        if goal.goal_type in set([GoalType.REFINE_SEARCH]):
                            call_type = API_CALL_TYPE.REFINE_SEARCH
                        else:
                            call_type = API_CALL_TYPE.SEARCH

                        api_request = APIRequest(
                            call_type=call_type,
                            parameters=api_parameters,
                            memory_dialog=memory_dialog,
                        )

                        # Send in the request and get the API Response back
                        api_response = memory_service_api.call_api(api_request)

                        # Construct Act Attributes from the API Response
                        act_attributes = ActAttributes()

                        if api_response.status == API_STATUS.SEARCH_FOUND:
                            act_attributes = ActAttributes(
                                slot_values=requested_act_attributes.slot_values,
                                slot_values_resolved=requested_act_attributes.slot_values_resolved,
                                request_slots=[],
                                memories=api_response.results.get(
                                    "retrieved_memories", []
                                ),
                            )

                        elif api_response.status == API_STATUS.SEARCH_NOT_FOUND:
                            # TODO: we can put a special logic here
                            act_attributes = ActAttributes(
                                slot_values=requested_act_attributes.slot_values,
                                slot_values_resolved=requested_act_attributes.slot_values_resolved,
                                request_slots=[],
                                memories=api_response.results.get(
                                    "retrieved_memories", []
                                ),
                            )

                        list_act_attributes.append(act_attributes)

                else:
                    # 1. (2) Follow-up questions
                    # 1. (3) Check disambiguation request
                    # TODO
                    pass

            else:
                # 2. Handle disambiguation info
                # TODO
                pass

            # Return an Frame object with the generated intent and attributes
            # TODO: handle multiple goal parameters & multiple acts
            return (
                Frame("", asst_dialog_act, list_act_attributes[0]),
                api_request,
                api_response,
            )

    class AssistantSearchGoalHandler(AssistantSearchGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class AssistantRefineSearchGoalHandler(AssistantSearchGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def execute_turn(
            self,
            goal: Goal,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ) -> Tuple[Frame, APIRequest, APIResponse]:

            # Execute
            return super().execute_turn(goal, memory_dialog, memory_service_api)

    class AssistantGetRelatedGoalHandler(AssistantGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.available_asst_acts = [
                DialogAct.INFORM_GET,
            ]

            self._uttr_template = {
                DialogAct.INFORM_GET: [
                    "Here are some of the related photos I found: {memories}.",
                    "Here are the related memories: {memories}.",
                    "Here are the related memories I found: {memories}.",
                    "Here are the related memories: {memories}. They match some of the criteria: {request_slots}.",
                    "Here are the related memories: {memories}. [[ Please comment on the retrieved photos ]].",
                    "Here are the related memories: {memories}. [[ Please summarize what is visible in the photos briefly ]].",
                ]
            }

            self._uttr_template_no_request_slots = {
                DialogAct.INFORM_GET: ["Here are the related memories: {memories}."]
            }

            self._uttr_template_no_results = {
                DialogAct.INFORM_GET: [
                    "I could not find any related memory that matches the criteria.",
                    "Sorry, I could not find any related memory. Anything else I can help?",
                ]
            }

        def generate_uttr(
            self, frame: Frame, goal: Goal, memory_service_api: MemoryServiceAPI
        ) -> str:

            memories = frame.act_attributes.memories
            request_slots = frame.act_attributes.request_slots

            if len(memories) > 0:
                if len(request_slots) > 0:
                    template = get_template(self._uttr_template, frame)
                else:
                    template = get_template(self._uttr_template_no_request_slots, frame)

            else:
                template = get_template(self._uttr_template_no_results, frame)

            verbose_memory = True if random.random() < 0.35 else False

            uttr = template.format(
                slot_values=str_slot_values(frame.act_attributes.slot_values),
                request_slots=str_request_slots(frame.act_attributes.request_slots),
                memories=str_memories(
                    frame.act_attributes.memories,
                    memory_service_api,
                    verbose=verbose_memory,
                ),
            )

            return uttr

        def execute_turn(
            self,
            goal: Goal,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ) -> Tuple[Frame, APIRequest, APIResponse]:

            assert len(memory_dialog.dialog.user_turns) > 0
            last_user_turn = memory_dialog.dialog.user_turns[-1]

            if True:
                # 1. User requests GET SIMILAR with parameters
                # Get a random dialog act label
                asst_dialog_act = random.choice(self.available_asst_acts)
                api_response = APIResponse()
                api_request = APIRequest()

                if True:
                    # 1. (1) Return GET_RELATED results
                    # Randomly fill the act_attributes
                    list_act_attributes = []

                    for goal_parameter in goal.goal_parameters:

                        # Construct an API request
                        api_request = APIRequest(
                            call_type=API_CALL_TYPE.GET_RELATED,
                            parameters={
                                ##### TODO: fix it so it grabs the right frame (instead of the last frame)
                                "memories": last_user_turn.frames[
                                    -1
                                ].act_attributes.memories,
                                "request_slots": last_user_turn.frames[
                                    -1
                                ].act_attributes.request_slots,
                                "slot_values": goal_parameter.filter,  ## TODO
                            },
                            memory_dialog=memory_dialog,
                        )

                        # Send in the request and get the API Response back
                        api_response = memory_service_api.call_api(api_request)

                        # Construct Act Attributes from the API Response
                        act_attributes = ActAttributes()

                        if api_response.status == API_STATUS.SEARCH_FOUND:
                            act_attributes = ActAttributes(
                                slot_values=api_response.results.get(
                                    "retrieved_info", {}
                                ),
                                request_slots=api_response.results.get(
                                    "request_slots", []
                                ),
                                memories=api_response.results.get(
                                    "retrieved_memories", []
                                ),
                            )

                        elif api_response.status == API_STATUS.SEARCH_NOT_FOUND:
                            # TODO: we can put a special logic here
                            act_attributes = ActAttributes(
                                slot_values=api_response.results.get(
                                    "retrieved_info", {}
                                ),
                                request_slots=api_response.results.get(
                                    "request_slots", []
                                ),
                                memories=api_response.results.get(
                                    "retrieved_memories", []
                                ),
                            )

                        list_act_attributes.append(act_attributes)

                else:
                    # 1. (2) Follow-up questions
                    # 1. (3) Check disambiguation request
                    # TODO
                    pass

            else:
                # 2. Handle disambiguation info
                # TODO
                pass

            # Return an Frame object with the generated intent and attributes
            # TODO: handle multiple goal parameters & multiple acts
            return (
                Frame("", asst_dialog_act, list_act_attributes[0]),
                api_request,
                api_response,
            )

    class AssistantGetInfoGoalHandler(AssistantGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.available_asst_main_acts = [
                DialogAct.INFORM_GET,
            ]
            self.available_asst_disambiguation_acts = [
                DialogAct.REQUEST_DISAMBIGUATE,
            ]

            self._uttr_template = {
                DialogAct.INFORM_GET: [
                    "Here is the info on {request_slots}: {slot_values}",
                    "I found the info on {request_slots}: {slot_values}",
                    "Here is the info I found: {slot_values}",
                ],
                DialogAct.REQUEST_DISAMBIGUATE: [
                    "Which photo or video do you mean?",
                    "Could you clarify which photo or video you are referring to?",
                ],
            }

        def execute_turn(
            self,
            goal: Goal,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ) -> Tuple[Frame, APIRequest, APIResponse]:

            assert len(memory_dialog.dialog.user_turns) > 0
            last_user_turn = memory_dialog.dialog.user_turns[-1]
            api_request = APIRequest()

            if not last_user_turn.is_disambiguation_response():
                # 1. User requests GET INFO with parameters
                api_response = APIResponse()

                # Request for disambiguation at a random rate
                n_mentioned_memories = len(memory_dialog.dialog.mentioned_memory_ids)
                if n_mentioned_memories > 1:
                    skip_disambiguation = random.random() > 0.4
                else:
                    # Only one or less memory was mentioned
                    skip_disambiguation = True

                if skip_disambiguation:
                    (
                        asst_dialog_act,
                        list_act_attributes,
                        api_request,
                        api_response,
                    ) = self.main_act(goal, memory_dialog, memory_service_api)

                else:
                    # 1. (2) Raise disambiguation request
                    # TODO
                    asst_dialog_act = random.choice(
                        self.available_asst_disambiguation_acts
                    )
                    list_act_attributes = [ActAttributes()]
                    api_response = APIResponse()

            else:
                # 2. Handle disambiguation info
                (
                    asst_dialog_act,
                    list_act_attributes,
                    api_request,
                    api_response,
                ) = self.main_act(goal, memory_dialog, memory_service_api)

            # Return an Frame object with the generated intent and attributes
            # TODO: handle multiple goal parameters & multiple acts
            return (
                Frame("", asst_dialog_act, list_act_attributes[0]),
                api_request,
                api_response,
            )

        def main_act(
            self,
            goal: Goal,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ):
            last_user_turn = memory_dialog.dialog.user_turns[-1]

            # 1. (1) Return info results
            # Get a random dialog act label
            asst_dialog_act = random.choice(self.available_asst_main_acts)
            list_act_attributes = []

            for goal_parameter in goal.goal_parameters:

                # Construct an API request
                api_request = APIRequest(
                    call_type=API_CALL_TYPE.GET_INFO,
                    parameters={
                        ##### TODO: fix it so it grabs the right frame (instead of the last frame)
                        "memories": last_user_turn.frames[-1].act_attributes.memories,
                        "request_slots": last_user_turn.frames[
                            -1
                        ].act_attributes.request_slots,
                    },
                    memory_dialog=memory_dialog,
                )

                # Send in the request and get the API Response back
                api_response = memory_service_api.call_api(api_request)

                # Construct Act Attributes from the API Response
                act_attributes = ActAttributes()

                if api_response.status == API_STATUS.INFO_FOUND:

                    act_attributes = ActAttributes(
                        slot_values=api_response.results.get("retrieved_info", {}),
                        request_slots=api_response.results.get("request_slots", []),
                        memories=api_response.results.get("retrieved_memories", []),
                    )

                elif api_response.status == API_STATUS.INFO_NOT_FOUND:
                    # TODO
                    pass

                list_act_attributes.append(act_attributes)

            return asst_dialog_act, list_act_attributes, api_request, api_response

    class AssistantShareGoalHandler(AssistantGoalHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.available_asst_acts = [
                DialogAct.CONFIRM_SHARE,
            ]

            self._uttr_template = {
                DialogAct.CONFIRM_SHARE: [
                    "Confirmed. I will share {memories}.",
                    "Confirmed. I will share them.",
                ],
            }

        def execute_turn(
            self,
            goal: Goal,
            memory_dialog: MemoryDialog,
            memory_service_api: MemoryServiceAPI,
        ) -> Tuple[Frame, APIRequest, APIResponse]:
            assert len(memory_dialog.dialog.user_turns) > 0
            last_user_turn = memory_dialog.dialog.user_turns[-1]

            if True:
                # 1. User requests SHARE with parameters
                # Get a random dialog act label
                asst_dialog_act = random.choice(self.available_asst_acts)
                api_response = APIResponse()
                api_request = APIRequest()

                if True:
                    # 1. (1) Return info results
                    list_act_attributes = []

                    for goal_parameter in goal.goal_parameters:

                        # Construct an API request
                        api_request = APIRequest(
                            call_type=API_CALL_TYPE.SHARE,
                            parameters={
                                ## TODO: fix so it grabs the right frame
                                "memories": last_user_turn.frames[
                                    -1
                                ].act_attributes.memories,
                            },
                            memory_dialog=memory_dialog,
                        )

                        # Send in the request and get the API Response back
                        api_response = memory_service_api.call_api(api_request)

                        # Construct Act Attributes from the API Response
                        act_attributes = ActAttributes()

                        if api_response.status == API_STATUS.SHARED:

                            act_attributes = ActAttributes(
                                slot_values={},
                                request_slots=[],
                                memories=api_response.results.get(
                                    "retrieved_memories", []
                                ),
                            )

                        list_act_attributes.append(act_attributes)

                else:
                    # 1. (2) Raise disambiguation request
                    # TODO
                    pass

            else:
                # 2. Handle disambiguation info
                # TODO
                pass

            # Return an Frame object with the generated intent and attributes
            # TODO: handle multiple goal parameters & multiple acts
            return (
                Frame("", asst_dialog_act, list_act_attributes[0]),
                api_request,
                api_response,
            )
