# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#!/usr/bin/env python3
from __future__ import annotations
from constants import GoalType, GoalMemoryRefType, DialogAct
from utils import str_memories, int_memory_ids, get_slot_values_simple_from_json
import pickle
from datetime import datetime


class MemoryDialog:
    def __init__(self, *args, **kwargs):
        self.memory_graph = kwargs.pop("memory_graph", {})  # JSON format
        self.dialog = kwargs.pop("dialog", None)
        self.domain = kwargs.pop("domain", None)

    def initialize(self):
        self.dialog = Dialog(domain=self.domain)

    def update(self, *args, **kwargs):
        # Reflects change in scenes or in dialogs
        # TODO: implement
        if "memory_graph" in kwargs:
            self.memory_graph = kwargs.pop("memory_graph")

        if "dialog" in kwargs:
            self.dialog = kwargs.pop("dialog")

    def is_goal_met(self, goal):
        # TODO: implement a more robust goal checking logic
        # For now, we look where there is a hanging 'disambiguation' request

        if self.dialog.asst_turns == []:
            return False

        last_asst_turn = self.dialog.asst_turns[-1]
        goal_met = not last_asst_turn.is_disambiguation_request()

        return goal_met

    def to_dict(self):
        out = self.dialog.to_dict()
        out["memory_graph_id"] = self.memory_graph.get_id()

        return out

    def get_memories(self):
        return self.memory_graph.get_memories()


class MemoryGraph:
    def __init__(self, *args, **kwargs):
        json_data = kwargs.pop("data", {})
        self.load_data(json_data)

    def load_data(self, json_data):
        self.id = json_data["memory_graph_id"]
        self.memories = [Memory(data=m) for m in json_data["memories"]]
        self.groups = json_data["memory_groups"]

        # Construct the memory to day/event mapping.
        self.trip_map = {}
        self.day_map = {}
        self.event_map = {}
        for trip_ind, trip_datum in enumerate(self.groups):
            for day_ind, day_datum in enumerate(trip_datum["days"]):
                for event_ind, event_datum in enumerate(day_datum["events"]):
                    for memory_id in event_datum["memories"]:
                        self.trip_map[memory_id] = trip_ind
                        self.day_map[memory_id] = day_ind
                        self.event_map[memory_id] = event_ind

    def get_day_events(self, memory_id):
        """Get the day events given memory_id."""
        trip_datum = self.groups[self.trip_map[memory_id]]
        return trip_datum["days"][self.day_map[memory_id]]

    def get_events(self, memory_id):
        """Get the events given memory_id."""
        day_datum = self.get_day_events(memory_id)
        return day_datum["events"][self.event_map[memory_id]]

    def get_id(self):
        return self.id

    def get_memories(self):
        return self.memories

    def get_memory_by_id(self, memory_id):
        for memory in self.memories:
            if int(memory.data["memory_id"]) == int(memory_id):
                return memory

    def get_memories_by_ids(self, memory_ids):
        return [self.get_memory_by_id(memory_id) for memory_id in memory_ids]

    def get_memory_url(self, memory_id):
        for memory in self.memories:
            if memory.data["memory_id"] == memory_id:
                return memory.get_memory_url()

        return []


class Memory:
    def __init__(self, *args, **kwargs):
        self.data = kwargs.pop("data", {})
        self.load_data(self.data)

    def __str__(self):
        return "Memory ID: {id} ({narrations}), Time: {time}, Loc: {location}".format(
            id=self.data["memory_id"],
            narrations=self.data["narrations"],
            time=self.data["time"],
            location=str(self.data["location"]["geo_tag"].get("place", "")),
        )

    def load_data(self, json_data):
        # ** TODO **
        """
        self.id = json_data['memory_id']
        self.time = json_data['time']
        self.start_time = json_data['start_time']
        self.end_time = json_data['end_time']
        self.narrations = json_data['narrations']
        self.media = json_data['media']
        self.location = json_data['location']
        self.participant = json_data['participant']
        self.activity = json_data['activity']
        self.object = json_data['object']
        """
        pass

    def get_memory_url(self):
        return [a["url"] for a in self.data["media"]]


class ActAttributes:
    def __init__(self, *args, **kwargs):
        self.slot_values = kwargs.pop("slot_values", {})  # slot_value pairs
        self.slot_values_resolved = kwargs.pop("slot_values_resolved", {})
        self.request_slots = kwargs.pop("request_slots", [])
        self.memories = kwargs.pop("memories", [])  # list of Memory objects

    def __str__(self):
        out = "{slot_values} | {request_slots} | {memories}".format(
            slot_values=str(self.slot_values),
            request_slots=str(self.request_slots),
            memories=str_memories(self.memories),
        )

        return out

    def to_dict(self):
        return {
            "slot_values": self.slot_values,
            #'slot_values_resolved': self.slot_values_resolved,
            "request_slots": self.request_slots,
            "memories": int_memory_ids(self.memories),
        }


class Frame:
    def __init__(self, uttr: str, dialog_act: DialogAct, act_attributes: ActAttributes):

        self.uttr = uttr
        self.dialog_act = dialog_act
        self.act_attributes = act_attributes

    def __str__(self):
        out = "{uttr} | {dialog_act} | {act_attributes}".format(
            uttr=str(self.uttr),
            dialog_act=self.dialog_act.value,
            act_attributes=str(self.act_attributes),
        )

        return out

    def to_dict(self):
        return {
            "uttr": self.uttr,
            "act": self.dialog_act.value,
            "act_attributes": self.act_attributes.to_dict(),
        }

    def is_disambiguation_request(self):
        return self.dialog_act in set(
            [DialogAct.REQUEST_DISAMBIGUATE, DialogAct.ASK_DISAMBIGUATE]
        )

    def is_disambiguation_response(self):
        return self.dialog_act in set([DialogAct.INFORM_DISAMBIGUATE])


class Turn:
    def __init__(self, frames, speaker, goal=None):
        self.frames = frames
        self.speaker = speaker
        self.goal = goal

    def __str__(self):
        out = "{frames}".format(
            frames=" / ".join([str(frame) for frame in self.frames])
        )

        return out

    def is_disambiguation_request(self):
        return True in set(frame.is_disambiguation_request() for frame in self.frames)

    def is_disambiguation_response(self):
        return True in set(frame.is_disambiguation_response() for frame in self.frames)

    def get_uttr(self):
        return ". ".join([f.uttr for f in self.frames])

    def get_frames_to_dict(self):
        return [f.to_dict() for f in self.frames]

    def has_dialog_acts(self, dialog_acts):
        """
        Return whether this turn contains
        any of the input target dialog acts in its frames.
        """
        for frame in self.frames:
            if frame.dialog_act in dialog_acts:
                return True

        return False


class Dialog:
    def __init__(self, idx=None, domain=None):
        self.user_turns = []
        self.asst_turns = []
        self.goals = []
        self.api_calls = []
        self.api_results = []
        self.idx = idx
        self.domain = domain
        self.mentioned_memory_ids = set([])

    def __str__(self):
        str_turns = []
        for i in range(len(self.user_turns)):
            user_turn = self.user_turns[i]
            asst_turn = self.asst_turns[i]

            str_turns.append(f"[Turn {i}] U: {user_turn}, A: {asst_turn}")

        return str([t for t in str_turns])

    def to_dict(self):
        out = {
            "dialogue": [],
            "dialogue_idx": self.idx,
            "domain": self.domain,
            "mentioned_memory_ids": list(self.mentioned_memory_ids),
        }

        for i in range(len(self.user_turns)):
            user_turn = self.user_turns[i]
            asst_turn = self.asst_turns[i]
            goal = self.goals[i]
            api_call = self.api_calls[i]

            turn_data = {
                "turn_idx": i,
                "system_transcript": asst_turn.get_uttr(),
                "system_transcript_annotated": asst_turn.get_frames_to_dict(),
                "transcript": user_turn.get_uttr(),
                "transcript_annotated": user_turn.get_frames_to_dict(),
                "goal_type": str(goal.goal_type),
                "api_call": api_call.to_dict(),
                #'api_result': api_result.to_dict()
            }

            try:
                # Some earlier data is missing api_result
                api_result = self.api_results[i]
                turn_data["api_result"] = api_result.to_dict()
            except:
                api_result = {}

            out["dialogue"].append(turn_data)

        return out

    def add_turn(self, user_turn, asst_turn):
        self.add_user_turn(user_turn)
        self.add_asst_turn(asst_turn)

    def add_goal(self, goal):
        self.goals.append(goal)

    def add_api_call(self, api_call):
        self.api_calls.append(api_call)

    def add_api_result(self, api_result):
        self.api_results.append(api_result)

    def add_user_turn(self, user_turn):
        self.user_turns.append(user_turn)

        for frame in user_turn.frames:
            for m in frame.act_attributes.memories:
                self.mentioned_memory_ids.add(m.data["memory_id"])

    def add_asst_turn(self, asst_turn):
        self.asst_turns.append(asst_turn)

        for frame in asst_turn.frames:
            for m in frame.act_attributes.memories:
                self.mentioned_memory_ids.add(m.data["memory_id"])


class APIRequest:
    def __init__(self, *args, **kwargs):
        self.call_type = kwargs.pop("call_type", None)
        self.parameters = kwargs.pop("parameters", None)
        self.memory_dialog = kwargs.pop("memory_dialog", None)

    def __str__(self):
        out = "call_type: {call_type}, parameters: {parameters}".format(
            call_type=self.call_type, parameters=str(self.parameters)
        )
        return out

    def to_dict(self, simple=False):
        if self.parameters is not None:
            parameters = {
                "slot_values": self.parameters.get("slot_values", []),
                "request_slots": self.parameters.get("request_slots", {}),
                "memories": int_memory_ids(self.parameters.get("memories"))
                if "memories" in self.parameters
                else [],
            }

            if simple:
                parameters["slot_values"] = get_slot_values_simple_from_json(
                    parameters["slot_values"]
                )

        else:
            parameters = {}

        return {"call_type": str(self.call_type), "parameters": parameters}


class APIResponse:
    def __init__(self, *args, **kwargs):
        self.status = kwargs.pop("status", None)
        self.request = kwargs.pop("request", None)
        self.results = kwargs.pop("results", {})

    def __str__(self):
        out = "status: {status}, results: {results}".format(
            status=self.status, results=str(self.results)
        )
        return out

    def to_dict(self):
        return {
            "status": str(self.status),
            "results": {
                "retrieved_memories": int_memory_ids(
                    self.results.get("retrieved_memories", [])
                ),
                "retrieved_info": self.results.get("retrieved_info", []),
            },
        }


class GoalParameter:
    def __init__(self, *args, **kwargs):
        self.filter = kwargs.pop("filter", {})  # slot_value pairs
        self.reference_type = kwargs.pop(
            "reference_type", GoalMemoryRefType.NOT_SPECIFIED
        )
        self.request_slots = kwargs.pop(
            "request_slots", []
        )  # need to map to Multimodal Context

    def __str__(self):
        out = "{filter} | {reference_type} | {request_slots}".format(
            filter=str(self.filter),
            reference_type=self.reference_type.value,
            request_slots=str(self.request_slots),
        )

        return out


class Goal(object):
    def __init__(self, *args, **kwargs):
        self.goal_type = kwargs.pop("goal_type", GoalType.UNKNOWN)
        self.goal_parameters = kwargs.pop("goal_parameters", [])

    def __str__(self):
        out = "{goal_type} | {goal_parameters}".format(
            goal_type=str(self.goal_type),
            goal_parameters=[str(p) for p in self.goal_parameters],
        )

        return out


class MemoryTime(object):

    NOT_SPECIFIED = -1

    def __init__(self, *args, **kwargs):
        # Allows for not_specified time for easy calculation
        self.year = kwargs.pop("year", self.NOT_SPECIFIED)
        self.month = kwargs.pop("month", self.NOT_SPECIFIED)
        self.day = kwargs.pop("day", self.NOT_SPECIFIED)
        self.hour = kwargs.pop("hour", self.NOT_SPECIFIED)
        self.minute = kwargs.pop("minute", self.NOT_SPECIFIED)
        self.second = kwargs.pop("second", self.NOT_SPECIFIED)

        if "str_datetime" in kwargs:
            self.load_datetime(kwargs.pop("str_datetime"))

    def load_datetime(self, str_datetime: str):
        # datetime: "2021-04-10 10:00:00"
        try:
            datetime_obj = datetime.fromisoformat(str_datetime)
            self.year = datetime_obj.year
            self.month = datetime_obj.month
            self.day = datetime_obj.day
            self.hour = datetime_obj.hour
            self.minute = datetime_obj.minute
            self.second = datetime_obj.second

        except:
            year_month = str_datetime.split("-")
            if len(year_month) == 1:
                self.year = int(year_month[0])
            else:
                self.year = int(year_month[0])
                self.month = int(year_month[1])

    def is_within(self, target_memory_time: self):
        # return whether self is within target_memory_time
        # for now, we assume that either year and/or month is provided
        if target_memory_time.year is not self.NOT_SPECIFIED:
            if self.year != target_memory_time.year:
                return False

        if target_memory_time.month is not self.NOT_SPECIFIED:
            if self.month != target_memory_time.month:
                return False

        return True

    def __str__(self):
        if self.day is self.NOT_SPECIFIED:
            if self.month is self.NOT_SPECIFIED:
                if self.year is self.NOT_SPECIFIED:
                    return ""
                else:
                    return "%d" % self.year
            else:
                return "%d-%02d" % (self.year, self.month)

        full_format = "%d-%02d-%02d %02d:%02d:%02d" % (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
        )

        return full_format


class MemoryLocation(object):
    def __init__(self, *args, **kwargs):
        self.data = kwargs.pop("data", {})

    def is_within(self, target_memory_location: self):
        # return whether self is within target_memory_time
        memory_geo_tag = self.data["geo_tag"]
        target_geo_tag = target_memory_location.data["geo_tag"]

        if "place" in target_geo_tag:
            return target_geo_tag["place"] == memory_geo_tag.get("place", "")

        elif "city" in target_geo_tag:
            return target_geo_tag["city"] == memory_geo_tag.get("city", "")

        elif "state" in target_geo_tag:
            return target_geo_tag["state"] == memory_geo_tag.get("state", "")

        elif "country" in target_geo_tag:
            return target_geo_tag["country"] == memory_geo_tag.get("country", "")

        return False


if __name__ == "__main__":

    # Memory Time operation test
    memory_time_1 = MemoryTime(year=2016, month=3)
    memory_time_2 = MemoryTime(year=2016, month=12)
    memory_time_3 = MemoryTime(year=2016)
    memory_time_4 = MemoryTime(year=2020)
    memory_time_5 = MemoryTime(str_datetime="2020-10-23 10:00:00")
    memory_time_6 = MemoryTime(str_datetime="2020-10")

    print(memory_time_1)
    print(memory_time_2)
    print(memory_time_3)
    print(memory_time_4)
    print(memory_time_5)
    print(memory_time_6)
    print(memory_time_1.is_within(memory_time_2))
    print(memory_time_1.is_within(memory_time_3))
    print(memory_time_1.is_within(memory_time_4))
    print(memory_time_5.is_within(memory_time_4))

    goal = Goal(
        goal_type=GoalType.GET_RELATED,
        goal_parameters=[GoalParameter(filter={"time": memory_time_5})],
    )

    print(goal)

    # Memory Graph Test
    import json

    path_memory_graph_list = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/memories/pilot/memory_may21_v1_100graphs.json"
    memory_graph_list = json.load(open(path_memory_graph_list, "r"))
    target_memory_graph_id = "St8BTzNuLCRb"
    target_memory_graph_idx = -1
    for i, memory_graph in enumerate(memory_graph_list):
        if target_memory_graph_id == memory_graph["memory_graph_id"]:
            target_memory_graph_idx = i
            break

    print(target_memory_graph_idx)
    sample_memory_graph = memory_graph_list[target_memory_graph_idx]

    mg = MemoryGraph(data=sample_memory_graph)
    target_memory_index = 1
    day_events = mg.get_day_events(memory_id=target_memory_index)
    events = mg.get_events(memory_id=target_memory_index)

    print("Target memory id:", target_memory_index)
    print("Day events indices:", day_events)
    print("Events indices:", events)
    print("Event memories:", [str(mg.memories[idx]) for idx in events["memories"]])
