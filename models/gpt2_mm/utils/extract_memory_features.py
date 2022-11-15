#! /usr/bin/env python
"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

Extract BUTD features for memories.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import base64
import json
import os
import pickle

import lmdb

import numpy as np
from PIL import Image
import torch
import tqdm


FEATURE_REGISTRY = {}
def register(cls):
    FEATURE_REGISTRY[cls.label] = cls
    return cls


# Extracts top-down bottom-up image features.
@register
class ImageFeatureReader(object):
    label = "butd"
    def __init__(self, feature_path, max_bboxes=-1):
        """Reads BUTD image features.

        Args:
            feature_path: Path to read the image features.
            max_bboxes: Maximum number of bounding boxes.
        """
        self.reader = lmdb.open(
            feature_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.reader.begin(write=False) as file_ptr:
            self.image_id_list = pickle.loads(file_ptr.get(b"keys"))
        self.num_bboxes = max_bboxes

    def __getitem__(self, image_id):
        image_id = str(image_id).encode()
        assert image_id in self.image_id_list, "Missing image_id!"

        with self.reader.begin(write=False) as file_ptr:
            item = pickle.loads(file_ptr.get(image_id))
            num_boxes = int(item["num_boxes"])
            features = np.frombuffer(
                base64.b64decode(item["features"]), dtype=np.float32
            ).reshape(num_boxes, 2048)
            boxes = np.frombuffer(
                base64.b64decode(item["boxes"]), dtype=np.float32
            ).reshape(num_boxes, 4)
            class_probs = np.frombuffer(
                base64.b64decode(item["cls_prob"]), dtype=np.float32
            ).reshape(num_boxes, 1601)
            features_dict = {
                "features": features,
                "bboxes": boxes,
                "class_probs": class_probs,
                "num_boxes": num_boxes,
                "image_w": int(item["image_w"]),
                "image_h": int(item["image_h"]),
            }
            if self.num_bboxes > 0:
                features_dict = self.trim_butd_features(features_dict)
            return features_dict

    def trim_butd_features(self, features_dict):
        """Trim BUTD features based on class probability.

        Args:
            feature_dict: BUTD features for images
        """
        # Get top class in each bbox and pick ones with highest class probability.
        top_class_prob = np.max(features_dict["class_probs"], axis=1)
        top_bboxes = np.argsort(-top_class_prob)[: self.num_bboxes]
        # Modify the elements.
        features_dict["bboxes"] = features_dict["bboxes"][top_bboxes]
        features_dict["features"] = features_dict["features"][top_bboxes]
        features_dict["num_boxes"] = self.num_bboxes
        del features_dict["class_probs"]
        return self.augment_butd_features(features_dict)

    def augment_butd_features(self, features_dict):
        """Augment BUTD feature with spatial location relative to height x width."""
        # Aliases.
        image_w = features_dict["image_w"]
        image_h = features_dict["image_h"]
        location = np.zeros((features_dict["num_boxes"], 5), dtype=np.float32)
        location[:, :4] = features_dict["bboxes"]
        location[:, 4] = (
            (location[:, 3] - location[:, 1])
            * (location[:, 2] - location[:, 0])
            / (float(image_w) * float(image_h))
        )
        location[:, 0] = location[:, 0] / float(image_w)
        location[:, 1] = location[:, 1] / float(image_h)
        location[:, 2] = location[:, 2] / float(image_w)
        location[:, 3] = location[:, 3] / float(image_h)
        features = np.concatenate([features_dict["features"], location], axis=-1)
        features_dict["features"] = features
        return features_dict


# Extracts clip features.
@register
class CLIPFeatureExtractor(object):
    """Extracts visual features using CLIP architecture."""
    label = "clip"
    def __init__(self, image_folder):
        """Initializes the feature extractor.

        Args:
            image_folder: Path to the raw COCO images.
        """
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._preprocess = clip.load("ViT-B/32", device=self._device)
        self._image_folder = image_folder

    def __getitem__(self, image_id):
        """Extracts image features for a given image_id.

        Args:
            image_id: Corresponding MSCOCO image_id
        """
        image_path = os.path.join(
            self._image_folder, f"COCO_train2014_{image_id:012d}.jpg"
        )
        image = (
            self._preprocess(PIL.Image.open(image_path)).unsqueeze(0).to(self._device)
        )
        with torch.no_grad():
            image_feature = self._model.encode_image(image)
        return {
            "features": image_feature.cpu().numpy(),
        }


# Extracts clip features.
@register
class SWINFeatureExtractor(object):
    """Extracts visual features using SWIN architecture."""
    label = "swin"
    def __init__(self, image_folder):
        """Initializes the feature extractor.

        Args:
            image_folder: Path to the raw COCO images.
        """
        self._use_gpu = torch.cuda.is_available()
        self._model = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            num_classes=0,
        )
        self._image_folder = image_folder

    def _prepare_image(self, image_path):
        """Given image path, load and prepare the image.

        Args:
            image_path: Path to the image to load

        Returns:
            image: Loaded image adjusted to the size
        """
        image = Image.open(image_path)
        image = np.array(image.resize((224, 224)), dtype=np.float32)
        if image.ndim != 3:
            image = np.stack([image, image, image], axis=2)
        image = torch.as_tensor(image).transpose(2, 0)[None]
        return image

    def __getitem__(self, image_id):
        """Extracts image features for a given image_id.

        Args:
            image_id: Corresponding MSCOCO image_id
        """
        image_path = os.path.join(
            self._image_folder, f"COCO_train2014_{image_id:012d}.jpg"
        )
        image = self._prepare_image(image_path)
        with torch.no_grad():
            image_feature = self._model(image)
        return {
            "features": image_feature.cpu().numpy(),
        }


def main(args):
    memory_graphs = {}
    for file_path in args["input_memory_json"]:
        # print(f"Reading: {file_path}")
        with open(file_path, "r") as file_id:
            graph_data = json.load(file_id)
            for datum in graph_data:
                if datum["memory_graph_id"] in memory_graphs:
                    print("Multiple memory graph ids exist!")
                else:
                    memory_graphs[datum["memory_graph_id"]] = datum
    print(f"# memory dialogs: {len(memory_graphs)}")

    memory_dialogs = {}
    for file_path in args["input_dialog_json"]:
        # print(f"Reading: {file_path}")
        with open(file_path, "r") as file_id:
            dialog_data = json.load(file_id)
        for datum in dialog_data["dialogue_data"]:
            dialog_id = datum["dialogue_idx"]
            memory_dialogs[dialog_id] = datum
    print(f"# dialogs: {len(memory_dialogs)}")

    # Load image features and trim if necessary.
    if args["feature_type"] == "butd":
        feature_extractor = ImageFeatureReader(
            args["input_feature_path"], args["max_bboxes"]
        )
    elif args["feature_type"] == "clip":
        feature_extractor = CLIPFeatureExtractor(args["input_image_folder"])
    elif args["feature_type"] == "swin":
        feature_extractor = SWINFeatureExtractor(args["input_image_folder"])
    else:
        raise NotImplementedError(f"""Invalid type: {args["feature_type"]}!""")

    progress_bar = tqdm.tqdm(memory_dialogs.items(), desc="Getting relevant images")
    relevant_image_ids = set()
    for dialog_id, datum in progress_bar:
        assert datum["memory_graph_id"] in memory_graphs, "Memory graph missing!"
        graph = memory_graphs[datum["memory_graph_id"]]
        sample_memories = {}
        for ii in graph["memories"]:
            if ii["memory_id"] in datum["mentioned_memory_ids"]:
                sample_memories[ii["memory_id"]] = ii
        for mem_id, mem_datum in sample_memories.items():
            relevant_image_ids.add((mem_id, mem_datum["media"][0]["media_id"]))

    progress_bar = tqdm.tqdm(relevant_image_ids, desc="Extracting features")
    for memory_id, image_id in progress_bar:
        feature_save_path = os.path.join(
            args["feature_save_path"],
            f"""mscoco_{args["feature_type"]}_{memory_id}.npy""",
        )
        np.save(feature_save_path, feature_extractor[image_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dialog_json", nargs="+", required=True, help="Input memories JSON"
    )
    parser.add_argument(
        "--input_memory_json", nargs="+", required=True, help="Input memories metadata"
    )
    parser.add_argument(
        "--feature_save_path", required=True, help="Folder to save memory features"
    )
    parser.add_argument(
        "--input_feature_path",
        default=None,
        help="Path to image features",
    )
    parser.add_argument(
        "--input_image_folder",
        default=None,
        help="Path to raw input images",
    )
    parser.add_argument(
        "--feature_type",
        choices=["butd", "clip", "swin"],
        required=True,
        help="Type of visual features to extract",
    )
    parser.add_argument(
        "--max_bboxes", default=-1, type=int, help="Maximum bounding boxes to retain"
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    # Conditional imports.
    if parsed_args["feature_type"] == "clip":
        import clip
    if parsed_args["feature_type"] == "swin":
        import timm

    main(parsed_args)
