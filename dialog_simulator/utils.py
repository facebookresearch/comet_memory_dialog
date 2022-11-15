# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
from constants import visual_slots, all_slots
import random

random.seed(0)


def build_parameter_ontology(memory_graph, metadata, domain=None, ontology=None):

    if ontology is None:
        ontology = {
            "visual": {},
            "non_visual": {},
            "all": {},
        }

    for memory in memory_graph.get_memories():

        for slot, value in memory.data.items():

            if slot not in all_slots:
                continue

            slot_category = "visual" if slot in visual_slots else "non_visual"

            if slot not in ontology["all"]:
                ontology["all"][slot] = []
                ontology[slot_category][slot] = []

            if value not in ontology["all"][slot]:
                ontology["all"][slot].append(value)
                ontology[slot_category][slot].append(value)

    return ontology


def batch_build_parameter_ontology(memory_graph_bank):
    ontology = {
        "visual": {},
        "non_visual": {},
        "all": {},
    }

    for i, memory_graph in enumerate(memory_graph_bank.values()):
        if i % 100 == 0:
            print("Processing memory graph %d" % i)

        ontology = build_parameter_ontology(
            memory_graph=memory_graph, metadata={}, ontology=ontology
        )
    return ontology


def str_memory(memory, memory_service_api=None, verbose=True):
    """
    memory: <Memory> object
    """
    memory_index = str(memory.data["memory_id"])
    memory_activity = str(
        ", ".join([a["activity_name"] for a in memory.data["activity"]])
    )
    time = str(memory.data["time"])[:-3] + " (" + memory.data["time_part"] + ")"
    location = memory.data["location"]["geo_tag"].get("place", "")

    if verbose:
        template = (
            "[Memory ID: {memory_index} ({memory_activity}), {time}, @ {location}]"
        )
    else:
        template = "[Memory ID: {memory_index}]"

    return template.format(
        memory_index=memory_index,
        memory_activity=memory_activity,
        time=time,
        location=location,
    )


def str_slot_values(slot_values):
    return "{ " + ", ".join([f"{k}: {v}" for k, v in slot_values.items()]) + " }"


def str_request_slots(request_slots):
    return "{ " + ", ".join([s for s in request_slots]) + " }"


def str_memories(memories, memory_service_api=None, verbose=True):
    # memories: <list> of <Memory> objects
    return (
        "{ "
        + str([str_memory(o, memory_service_api, verbose) for o in memories])
        + " }"
    )


def int_memory_ids(memories):
    return [int(m.data["memory_id"]) for m in memories]


def get_template(template_map, nlu_label):
    return random.choice(template_map.get(nlu_label.dialog_act))


def load_data_pickle(path_pickle):
    import pickle

    return pickle.load(open(path_pickle, "rb"))


def weighted_choice(population, weights):
    return random.choices(population=population, weights=weights, k=1)[0]


def get_slot_values_simple_from_json(
    slot_values,
    location_target="place",
    participant_target="name",
    activity_target="activity_name",
):
    if slot_values == None:
        return {}

    out = {}

    for slot, value in slot_values.items():
        if slot == "location":
            out[slot] = get_location_simple_from_json(value, target=location_target)

        elif slot == "participant":
            out[slot] = get_participant_simple_from_json(
                value, target=participant_target
            )

        elif slot == "activity":
            out[slot] = get_activity_simple_from_json(value, target=activity_target)

        else:
            out[slot] = str(value)

    return out


def get_location_simple_from_json(location_json, target="place"):
    """
    JSON format:
     "location":{
        "gps":{
           "lat":40.00,
           "lon":100.00
        },
        "geo_tag":{
           "place":"Summit at Snoqualmie",
           "city":"Seattle",
           "state":"Washington",
           "country":"USA"
        }
    """
    if target in location_json["geo_tag"]:
        return location_json["geo_tag"][target]

    return location_json["geo_tag"].get("city")


def get_participant_simple_from_json(participant_json, target="name"):
    """
    JSON format:
     "participant":[
        {
           "name":"John",
           "memory_graph_id":1
        },
        {
           "name":"Mary",
           "memory_graph_id":2
        }
     ],
    """
    return [p[target] for p in participant_json]


def get_activity_simple_from_json(activity_json, target="activity_name"):
    """
    JSON format:
     "activity":[
        {
           "activity_name":"skiing"
        }
     ]
    """
    return [a[target] for a in activity_json]


def get_edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def resolve_sv_entities(slot_values: dict, parameter_ontology: dict) -> dict:

    if "location" in slot_values:
        str_location = slot_values["location"]
        resolved_location_obj = resolve_location(
            str_location, parameter_ontology["all"]["location"], True
        )
        slot_values["location"] = resolved_location_obj

    if "participant" in slot_values:
        str_participant = slot_values["participant"]
        resolved_participant_obj = resolve_participant(
            str_participant, parameter_ontology["all"]["participant"], True
        )
        slot_values["participant"] = resolved_participant_obj

    if "activity" in slot_values:
        str_activity = slot_values["activity"]
        resolved_activity_obj = resolve_activity(
            str_activity, parameter_ontology["all"]["activity"], True
        )
        slot_values["activity"] = resolved_activity_obj

    return slot_values


def resolve_location(str_location: str, location_ontology: list, fuzzy: bool) -> dict:

    print("Resolving location: %s" % str_location)

    # Strict match
    for target_location_obj in location_ontology:
        if str_location.lower() == target_location_obj["geo_tag"]["place"].lower():
            return target_location_obj

    # If strict match doesn't work & fuzzy == True:
    if fuzzy:
        print("Trying fuzzy match for location %s" % str_location)
        for target_location_obj in location_ontology:
            edit_distance = get_edit_distance(
                str_location.lower(), target_location_obj["geo_tag"]["place"].lower()
            )

            if edit_distance < 7:
                print("Fuzzy match found for location %s" % str_location)
                return target_location_obj

    print("Match not found for location %s" % str_location)
    return {}


def resolve_list_entities(
    str_entity: str, entity_ontology: list, fuzzy: bool, target_key: str
) -> dict:
    """
    (input) str_entities: [
        'element_1', ...
        e.g. 'skiing', 'snowboarding'
    ]

    (target) list_entities: [
        {
            'target_key': <str>,
            e.g. 'activity_name': 'skiing'
        }
    ]
    """
    # First, try converting the str to a list
    try:
        set_entity = set(name.lower() for name in eval(str_entity))

        # Strict match
        for target_entity_obj in entity_ontology:
            target_entity = set(
                str(p.get(target_key, "")).lower() for p in target_entity_obj
            )

            if set_entity == target_entity:
                return target_entity_obj

        # Fuzzy match 1
        if fuzzy and len(set_entity) > 1:
            print("Trying fuzzy match for entity %s" % str_entity)
            match_thershold = max(1, int(len(set_entity) / 2) - 1)

            for target_entity_obj in entity_ontology:
                target_entity = set(
                    str(p.get(target_key, "")).lower() for p in target_entity_obj
                )

                if len(set_entity.intersection(target_entity)) >= match_thershold:
                    print("Fuzzy match found for %s" % str_entity)
                    return target_entity_obj
    except:
        print("Can't convert to list.")
        # Fuzzy match 2
        if fuzzy:
            print("Trying fuzzy match for entity %s" % str_entity)
            for target_entity_obj in entity_ontology:
                edit_distance = get_edit_distance(
                    str_entity.lower().replace("'", ""),
                    str(
                        [str(p.get(target_key, "")).lower() for p in target_entity_obj]
                    ).replace("'", ""),
                )

                if edit_distance < 9:
                    print("Fuzzy match found for %s" % str_entity)
                    return target_entity_obj

    print("Match not found for %s" % str_entity)
    return {}


def resolve_participant(
    str_participant: str, participant_ontology: list, fuzzy: bool
) -> dict:

    print("Resolving participant: %s" % str_participant)
    return resolve_list_entities(
        str_entity=str_participant,
        entity_ontology=participant_ontology,
        fuzzy=fuzzy,
        target_key="name",
    )


def resolve_activity(str_activity: str, activity_ontology: list, fuzzy: bool) -> dict:

    print("Resolving activity: %s" % str_activity)
    return resolve_list_entities(
        str_entity=str_activity,
        entity_ontology=activity_ontology,
        fuzzy=fuzzy,
        target_key="activity_name",
    )


if __name__ == "__main__":
    # Test resolve entities
    import json

    path_parameter_ontology = "/Users/shanemoon/workspace/memory_dialog/dialog_simulator/final_data/all_parameter_ontology.json"

    parameter_ontology = json.load(open(path_parameter_ontology, "r"))

    list_slot_values = [
        # Strict match
        {
            "location": "Seattle Downtown",
            "participant": "['Carl', 'Bryan', 'Emily']",
            "activity": "['cooking sausages']",
        },
        # Fuzzy match by set intersection
        {
            "location": "seattle downtow",
            "participant": "['Carl', 'Shane']",
            "activity": "['cooking sausages', 'peeling potatoes']",
        },
        # Fuzzy match with incomplete list formats
        {
            "location": "Bay Area",
            "participant": "Carl Bryan Emily",
            "activity": "[cooking sausages",
        },
    ]

    for slot_values in list_slot_values:
        print("------------------------------------")
        print(resolve_sv_entities(slot_values, parameter_ontology))
