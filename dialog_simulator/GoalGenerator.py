# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
import random
from constants import (
    GoalType,
    GoalMemoryRefType,
    numeric_slots,
    non_visual_slots,
    visual_slots,
    all_slots,
)
from Data import Goal, GoalParameter, MemoryTime
from utils import weighted_choice
import copy

random.seed(0)


class RuleBasedGoalGenerator:
    def __init__(self, *args, **kwargs):
        self.non_visual_slots = non_visual_slots
        self.visual_slots = visual_slots
        self.all_slots = all_slots

    def sample_goals(self, *args, **kwargs):
        memory_graph = kwargs.pop("memory_graph", None)
        goal_config = kwargs.pop("goal_config", {})
        n_min_goals = goal_config.get("n_min_goals", 3)
        n_max_goals = goal_config.get("n_max_goals", 5)
        n_goals = random.randint(n_min_goals, n_max_goals)

        goal_type_list = [
            GoalType.SEARCH,
            GoalType.REFINE_SEARCH,
            GoalType.GET_RELATED,
            GoalType.GET_INFO,
            GoalType.GET_AGGREGATED_INFO,
            GoalType.SHARE,
            GoalType.CHITCHAT,
        ]
        goal_type_list_weights_start = [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            # 1, 0, 0, 0, 1, 0, 0,
        ]

        goal_type_list_weights_mid = [
            0.8,
            1.1,
            1.7,
            1.1,
            0,
            0.1,
            0,
            # 1, 0.8, 0.8, 1, 1, 0.5, 0.5,
        ]

        goal_type_list_weights_end = [
            0.3,
            0.5,
            0.6,
            0.5,
            0,
            3,
            0,
            # 0.5, 0.5, 0.5, 0.5, 0.5, 3, 1,
        ]

        # Randomly sample from the goal type list
        # For now, we enforce the goals to start with BROWSE
        # and end with ADD_TO_CART
        # TODO: allow for a more flexible way of generating
        # goal types
        goal_types = (
            random.choices(
                population=goal_type_list, weights=goal_type_list_weights_start, k=1
            )
            + random.choices(
                population=goal_type_list,
                weights=goal_type_list_weights_mid,
                k=n_goals - 2,
            )
            + random.choices(
                population=goal_type_list, weights=goal_type_list_weights_end, k=1
            )
        )

        # Make a complete goal with an accompanying set of goal parameters
        # for each goal_type
        goals = []
        for goal_type in goal_types:
            # For now, we pass in a random set of goal_parameters
            goal_parameters = self.sample_goal_parameters(
                goal_type, memory_graph, goal_config
            )
            goals.append(Goal(goal_type=goal_type, goal_parameters=goal_parameters))

        return goals

    def sample_goal_parameters(self, goal_type, memory_graph, goal_config):
        # Sample goal parameters according to the input sample

        # TODO: IMPLEMENT **
        goal_parameters = []
        parameter_ontology = goal_config["parameter_ontology"]

        # (1) Pick a search filter
        search_filter = {}

        if goal_type in set(
            [GoalType.SEARCH, GoalType.REFINE_SEARCH, GoalType.GET_RELATED]
        ):

            if goal_type == GoalType.GET_RELATED:
                n_slots = weighted_choice(population=[1, 2], weights=[0.93, 0.07])
            else:
                n_slots = weighted_choice(population=[1, 2], weights=[0.75, 0.25])

            # Candidate slots: exclude a few slots that
            # are semantically infeasible
            # **** TODO ****: confirm that there is no slot to exclude
            candidate_slots = self.all_slots - set([""])

            search_filter_slots = random.choices(
                population=list(candidate_slots), k=n_slots
            )

            for search_filter_slot in search_filter_slots:
                # We first randomly assign a value for a randomly selected slot
                if search_filter_slot == "time":
                    # Instead of choosing a specific datetime,
                    # search by year or month instead.
                    random_datetime = MemoryTime(
                        str_datetime=random.choice(
                            parameter_ontology["all"].get(search_filter_slot)
                        )
                    )

                    if random.random() > 0.1:
                        search_filter_value = str(MemoryTime(year=random_datetime.year))

                    else:
                        search_filter_value = str(
                            MemoryTime(
                                year=random_datetime.year, month=random_datetime.month
                            )
                        )

                    if goal_type == GoalType.GET_RELATED:
                        # A special value for refine_search: 'next' and 'prev'
                        # e.g. "where did we go next?"
                        if random.random() > 0.3:
                            search_filter_value = random.choice(
                                ["right after", "right before", "on the same day"]
                            )

                elif search_filter_slot == "location":
                    # TODO: Instead of choosing a specific location,
                    # occasionally search with a coarser query.
                    search_filter_value = random.choice(
                        parameter_ontology["all"].get(search_filter_slot)
                    )

                    if random.random() > 0.7:
                        search_filter_value = copy.deepcopy(search_filter_value)
                        search_filter_value["geo_tag"].get("place")

                else:
                    # TODO: handle subsampling of participants & activities
                    search_filter_value = random.choice(
                        parameter_ontology["all"].get(search_filter_slot)
                    )

                if search_filter_value != "":
                    search_filter[search_filter_slot] = search_filter_value

        # (2) Pick an object reference type
        object_reference_type = GoalMemoryRefType.NOT_SPECIFIED

        if goal_type in set([GoalType.GET_RELATED, GoalType.GET_INFO, GoalType.SHARE]):

            object_reference_type = weighted_choice(
                population=[
                    GoalMemoryRefType.PREV_TURN,
                    GoalMemoryRefType.DIALOG,
                    GoalMemoryRefType.GRAPH,
                ],
                weights=[0.8, 0.2, 0.0],
            )

        # (3) Pick slots to request (e.g. in questions)
        request_slots = []

        if goal_type in set([GoalType.GET_INFO]):
            # We randomly sample slots to ask
            # ****** TODO *******: make sure it's not asking about
            # the parameters that were already in search filter

            ask_from_visual_slot = random.random() > 0.9

            if ask_from_visual_slot:
                # ask about visual_slots (rare): people, activity
                n_request_slots = 1
                request_slots.extend(
                    random.sample(self.non_visual_slots, n_request_slots)
                )

            else:
                # ask about non_visual_slots: time, location
                n_request_slots = weighted_choice(population=[1, 2], weights=[0.8, 0.2])
                request_slots.extend(
                    random.sample(self.non_visual_slots, n_request_slots)
                )

        elif goal_type in set([GoalType.GET_RELATED]):
            # We randomly sample slots to ask
            # iff search_filter is empty
            if len(search_filter) == 0:
                n_request_slots = weighted_choice(population=[0, 1], weights=[0.4, 0.6])
                request_slots.extend(random.sample(self.all_slots, n_request_slots))

        elif goal_type in set([GoalType.GET_AGGREGATED_INFO]):
            # ****** TODO *******
            pass

        # (4) Compile it into a goal parameter
        goal_parameter = GoalParameter(
            filter=search_filter,
            reference_type=object_reference_type,
            request_slots=request_slots,
        )
        goal_parameters.append(goal_parameter)

        return goal_parameters
