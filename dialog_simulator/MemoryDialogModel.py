# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
from constants import API_CALL_TYPE, TurnSpeaker, DialogAct
from Data import Turn, Frame, ActAttributes, MemoryDialog, APIResponse, APIRequest
from typing import Dict, Tuple
import sys

sys.path.append("/Users/shanemoon/workspace/memory_dialog/models/")
from gpt2_dst.scripts.run_generation import generate_sequences
from gpt2_dst.utils.convert import (
    format_context,
    format_api_call,
    format_api_result,
    parse_flattened_result,
    TEMPLATE_PREDICT,
    TEMPLATE_PREDICT_RESPONSE,
    START_OF_API_CALL,
    END_OF_API_CALL,
    END_OF_API_RESULT,
    END_OF_SENTENCE,
)
from utils import resolve_sv_entities


class MemoryDialogModelBase:
    def __init__(self, *args, **kwargs):
        self.displayed_memories = []

    def predict_api_call(self, query: str, memory_dialog: MemoryDialog) -> Dict:

        return {
            "call_type": API_CALL_TYPE.UNDEFINED,
            "slot_values": {},
            "request_slots": [],
            "memories": [],  # <list> of <Memory> objects
        }

    def construct_api_request(
        self, query: str, memory_dialog: MemoryDialog
    ) -> Tuple[Turn, APIRequest]:

        # Predict / extract call_type and parameters from query
        predicted = self.predict_api_call(query, memory_dialog)

        # Cast user query into a Turn instance
        query_frame = Frame(
            uttr=query,
            dialog_act=predicted["dialog_act"],
            act_attributes=ActAttributes(
                slot_values=predicted["slot_values"],
                request_slots=predicted["request_slots"],
                # <list> of <Memory> objects
                memories=predicted["memories"],
            ),
        )

        # For now, we assume one frame per turn
        user_turn = Turn(frames=[query_frame], speaker=TurnSpeaker.USER, goal=None)

        # Gegenerate an API request from the predicted values
        str_call_type = predicted["call_type"]
        try:
            call_type = eval(str_call_type)
        except Exception:
            call_type = API_CALL_TYPE.UNDEFINED

        api_parameters = {
            "slot_values": predicted["slot_values"],
            "request_slots": predicted["request_slots"],
            "memories": predicted["memories"],  # <list> of <Memory> objects
            "n_max_results": 2,
        }

        # Call API
        api_request = APIRequest(
            call_type=call_type, parameters=api_parameters, memory_dialog=memory_dialog
        )

        return user_turn, api_request

    def update_display(self, api_response: APIResponse):

        if api_response.status is not None:
            retrieved_memories = (
                api_response.to_dict().get("results", {}).get("retrieved_memories", [])
            )
            self.displayed_memories = retrieved_memories

    def predict_assistant_response(
        self,
        query: str,
        api_call: APIRequest,
        api_response: APIResponse,
        memory_dialog: MemoryDialog,
    ) -> Dict:

        return {
            "uttr": "",
            "dialog_act": DialogAct.UNKNOWN,
            "slot_values": {},
            "request_slots": [],
            "memories": [],
        }

    def construct_assistant_response(
        self,
        query: str,
        api_call: APIRequest,
        api_response: APIResponse,
        memory_dialog: MemoryDialog,
    ) -> Turn:

        predicted = self.predict_assistant_response(
            query, api_call, api_response, memory_dialog
        )

        response_frame = Frame(
            uttr=predicted["uttr"],
            dialog_act=predicted["dialog_act"],
            act_attributes=ActAttributes(
                slot_values=predicted["slot_values"],
                slot_values_resolved={},
                request_slots=predicted["request_slots"],
                memories=predicted["memories"],
            ),
        )

        # For now, we assume one frame per turn
        assistant_turn = Turn(
            frames=[response_frame], speaker=TurnSpeaker.ASSISTANT, goal=None
        )

        return assistant_turn


class PilotMemoryDialogModel(MemoryDialogModelBase):
    def __init__(self, *args, **kwargs):
        super(PilotMemoryDialogModel, self).__init__(*args, **kwargs)

        self.model = kwargs.pop("model")
        self.tokenizer = kwargs.pop("tokenizer")
        self.length = kwargs.pop("length")
        self.parameter_ontology = kwargs.pop("parameter_ontology")

        self.prev_asst_uttr = None
        self.lst_context = []
        self.turn_id = 0

    def predict_api_call(self, query: str, memory_dialog: MemoryDialog) -> Dict:

        # Form the prompt
        to_predict = self.form_prompt_for_api_call(
            self.lst_context, self.prev_asst_uttr, query
        )

        # Generate the sequence
        generated = generate_sequences(
            self.model, self.tokenizer, to_predict, verbose=False
        )[0]

        # Extract the api_call
        parsed_api_call, _ = self.parse_assistant_response(generated)

        call_type = parsed_api_call.get("act", None)
        slot_values = {k: v for k, v in parsed_api_call.get("slots", [])}
        request_slots = parsed_api_call.get("request_slots", [])
        memory_ids = parsed_api_call.get("memories", [])
        memories = memory_dialog.memory_graph.get_memories_by_ids(memory_ids)

        # Entity Resolution for locations, etc.
        slot_values = resolve_sv_entities(slot_values, self.parameter_ontology)

        # Form an API call
        return {
            "call_type": call_type,
            "dialog_act": DialogAct.UNKNOWN,
            "slot_values": slot_values,
            "request_slots": request_slots,
            "memories": memories,  # <list> of <Memory> objects
        }

    def predict_assistant_response(
        self,
        query: str,
        api_call: APIRequest,
        api_response: APIResponse,
        memory_dialog: MemoryDialog,
    ) -> Dict:

        # Form the prompt
        to_predict = self.form_prompt_for_response(
            self.lst_context, self.prev_asst_uttr, query, api_call, api_response
        )

        # Generate the sequence
        generated = generate_sequences(
            self.model, self.tokenizer, to_predict, verbose=False
        )[0]

        _, response_text = self.parse_assistant_response(generated)
        self.prev_asst_uttr = response_text

        if api_response.results is not None:
            memories = api_response.results.get("retrieved_memories", [])
        else:
            memories = []

        return {
            "uttr": response_text,
            "dialog_act": DialogAct.UNKNOWN,
            "slot_values": {},
            "request_slots": [],
            "memories": memories,  # <list> of <Memory> objects
        }

    def form_prompt_for_api_call(
        self, lst_context, prev_asst_uttr, user_uttr, len_context=2
    ):

        # Format main input context
        context = format_context(
            prev_asst_uttr,
            user_uttr,
            self.displayed_memories,
            use_multimodal_contexts=True,
        )

        # Concat with previous contexts
        lst_context.append(context)
        context = " ".join(lst_context[-len_context:])

        # Format the main input
        predict = TEMPLATE_PREDICT.format(
            context=context,
            START_OF_API_CALL=START_OF_API_CALL,
        )

        print("============== Prompt Sequence ==============")
        print(predict)
        print("=============================================")
        return predict

    def form_prompt_for_response(
        self,
        lst_context,
        prev_asst_uttr,
        user_uttr,
        api_call,
        api_response,
        len_context=2,
    ):

        # Format main input context
        # Context should already have been formatted
        context = " ".join(lst_context[-len_context:])

        # Format API call
        json_api_call = api_call.to_dict(simple=True)
        str_api_call = format_api_call(
            json_api_call["call_type"], json_api_call["parameters"]
        )

        # Format API result
        json_api_response = api_response.to_dict()
        str_api_result = format_api_result(json_api_response)

        # Format the main input
        predict = TEMPLATE_PREDICT_RESPONSE.format(
            context=context,
            START_OF_API_CALL=START_OF_API_CALL,
            belief_state=str_api_call,
            END_OF_API_CALL=END_OF_API_CALL,
            api_result=str_api_result,
            END_OF_API_RESULT=END_OF_API_RESULT,
        )

        print("============== Prompt Sequence ==============")
        print(predict)
        print("=============================================")
        return predict

    def parse_assistant_response(self, generated):
        print("============== Generated Sequence ==============")
        print(generated)
        print("================================================")
        parsed = parse_flattened_result(generated)

        if parsed == []:
            parsed_api_call = {}

        else:
            # For now, we only consider one api_call per turn
            parsed_api_call = parsed[-1]

        if parsed_api_call == {}:
            response_text = "I could not understand. Could you repeat please?"

        if END_OF_API_RESULT in generated:
            response_text = generated.split(END_OF_API_RESULT)[-1]
            response_text = response_text.replace(END_OF_SENTENCE, "")

        else:
            response_text = "(No system response)"

        return parsed_api_call, response_text
