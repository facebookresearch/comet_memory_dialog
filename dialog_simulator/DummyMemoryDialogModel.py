# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#!/usr/bin/env python3
from constants import API_CALL_TYPE, TurnSpeaker, DialogAct
from Data import Turn, Frame, ActAttributes, MemoryDialog, APIResponse, APIRequest
from typing import Dict, Tuple


class DummyMemoryDialogModel(MemoryDialogModelBase):
    def __init__(self, *args, **kwargs):
        super(DummyMemoryDialogModel, self).__init__(*args, **kwargs)

    def predict_api_call(self, query: str) -> Dict:
        return {
            "call_type": API_CALL_TYPE.SEARCH,
            "dialog_act": DialogAct.UNKNOWN,
            "slot_values": {},
            "request_slots": [],
            "memories": [],
        }

    def predict_assistant_response(
        self, query: str, api_response: APIResponse, memory_dialog: MemoryDialog
    ):

        response_str = (
            "User asked:"
            + query
            + ". Dialog history: "
            + str(memory_dialog)
            + ". API response:"
            + str(api_response)
        )

        return {
            "uttr": response_str,
            "dialog_act": DialogAct.UNKNOWN,
            "slot_values": {},
            "request_slots": [],
            "memories": api_response.results.get("retrieved_memories"),
        }
