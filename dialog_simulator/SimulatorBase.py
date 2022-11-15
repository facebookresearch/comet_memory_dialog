# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#!/usr/bin/env python3
from Data import MemoryDialog, Goal, Frame
from typing import List


class SimulatorBase:
    def register_memory_service_api(self, memory_service_api):
        self.memory_service_api = memory_service_api

    def fit_goal_to_intent(self, args):
        # Define the goal to intent mapping behavior
        pass

    def is_servable(self, goal: Goal) -> bool:
        # Check whether this simulator can serve the input goal.
        pass

    def generate_nlu_label(self, goal: Goal, context: MemoryDialog) -> Frame:
        # Need to define this behavior first e.g. as a config, a model, etc.
        pass

    def generate_uttr(self, nlu_label: Frame) -> str:
        pass
