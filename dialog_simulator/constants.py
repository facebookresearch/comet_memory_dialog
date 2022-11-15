# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
from enum import Enum


class GoalType(Enum):
    UNKNOWN = "unknown"
    SEARCH = "search"
    REFINE_SEARCH = "refine_search"
    GET_RELATED = "get_related"
    GET_INFO = "get_info"
    GET_AGGREGATED_INFO = "get_aggregated_info"
    SHARE = "share"
    CHITCHAT = "chitchat"


class DialogAct(Enum):
    UNKNOWN = "unknown"

    INFORM_GET = "INFORM:GET"
    INFORM_REFINE = "INFORM:REFINE"
    INFORM_PREFER = "INFORM:PREFER"
    INFORM_DISPREFER = "INFORM:DISPREFER"
    INFORM_SHARE = "INFORM:SHARE"
    INFORM_DISAMBIGUATE = "INFORM:DISAMBIGUATE"
    INFORM_CHITCHAT = "INFORM:CHITCHAT"

    REQUEST_GET = "REQUEST:GET"
    REQUEST_REFINE = "REQUEST:REFINE"
    REQUEST_PREFER = "REQUEST:PREFER"
    REQUEST_DISPREFER = "REQUEST:DISPREFER"
    REQUEST_SHARE = "REQUEST:SHARE"
    REQUEST_DISAMBIGUATE = "REQUEST:DISAMBIGUATE"

    CONFIRM_GET = "CONFIRM:GET"
    CONFIRM_REFINE = "CONFIRM:REFINE"
    CONFIRM_PREFER = "CONFIRM:PREFER"
    CONFIRM_DISPREFER = "CONFIRM:DISPREFER"
    CONFIRM_SHARE = "CONFIRM:SHARE"
    CONFIRM_DISAMBIGUATE = "CONFIRM:DISAMBIGUATE"

    PROMPT_GET = "PROMPT:GET"
    PROMPT_REFINE = "PROMPT:REFINE"
    PROMPT_PREFER = "PROMPT:PREFER"
    PROMPT_DISPREFER = "PROMPT:DISPREFER"
    PROMPT_SHARE = "PROMPT:SHARE"
    PROMPT_DISAMBIGUATE = "PROMPT:DISAMBIGUATE"

    ASK_GET = "ASK:GET"
    ASK_REFINE = "ASK:REFINE"
    ASK_PREFER = "ASK:PREFER"
    ASK_DISPREFER = "ASK:DISPREFER"
    ASK_SHARE = "ASK:SHARE"
    ASK_DISAMBIGUATE = "ASK:DISAMBIGUATE"


class GoalMemoryRefType(Enum):
    PREV_TURN = "PREV_TURN"
    DIALOG = "DIALOG"
    GRAPH = "GRAPH"
    NOT_SPECIFIED = "Not Specified"


class ObjectRefType(Enum):
    R1 = "R1"  # Unique object in the scene
    R2 = "R2"  # Object in the dialog history, same view point
    R3 = "R3"  # Object in the dialog history, previous view point
    NOT_SPECIFIED = "Not Specified"


class API_STATUS(Enum):
    SEARCH_FOUND = "Search Founud"
    SEARCH_NOT_FOUND = "Search Not Founud"
    INFO_FOUND = "Info Found"
    INFO_NOT_FOUND = "Info Not Found"
    SHARED = "Shared"


class API_CALL_TYPE(Enum):
    SEARCH = "Search"
    REFINE_SEARCH = "Refine Search"
    GET_INFO = "Get Info"
    SHARE = "Share"
    GET_RELATED = "Get Related"
    UNDEFINED = "Undefined"


class TurnSpeaker(Enum):
    USER = "User"
    ASSISTANT = "Assistant"


numeric_slots = {"time"}

non_visual_slots = {
    "location",
    "time",
}

visual_slots = {"participant", "activity"}

all_slots = {"time", "location", "participant", "activity"}
