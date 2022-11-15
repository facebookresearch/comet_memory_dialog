# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#! /usr/bin/env python
"""
Gets the best model given all the checkpoints.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import re


def main(args):
    for folder_name in args["model_checkpoint_folder"]:
        listing = [ii for ii in os.listdir(folder_name) if "checkpoint-" in ii]
        valid_metrics = {}
        for checkpoint_name in listing:
            checkpoint_folder = os.path.join(folder_name, checkpoint_name)
            eval_path = os.path.join(checkpoint_folder, "eval_results.txt")
            epoch_search = re.search(r"checkpoint-(\d*)", checkpoint_name)
            with open(eval_path, "r") as file_id:
                result = [ii.strip("\n") for ii in file_id.readlines()][0]
            perplexity_search = re.search(r"([0-9\.]+)", result)

            # NOTE: Does not handle error conditions.
            if perplexity_search is None or epoch_search is None:
                print(f"Missing epoch: {checkpoint_name}")
                continue

            perplexity = float(perplexity_search.group(1))
            epoch = int(epoch_search.group(1))
            valid_metrics[epoch] = perplexity

        best_epoch, _ = sorted(valid_metrics.items(), key=lambda x: x[1])[0]
        best_folder = os.path.join(folder_name, f"checkpoint-{best_epoch}")
        print(best_folder)
        print("." * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_checkpoint_folder",
        nargs="+",
        required=True,
        help="List of model checkpoint folders",
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
