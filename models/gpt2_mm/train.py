# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
import json
import logging
import math
import os
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from pprint import pformat

import torch
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import (
    OptimizerParamsHandler,
    OutputHandler,
    TensorboardLogger,
)
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from transformers import *
from VideoGPT2 import *
import pickle as pkl

from dataset_memory import collate_fn, get_dataset, MemoryDialogDataset, padding

MODEL_INPUTS = ["input_ids", "token_type_ids", "lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids", "lm_labels"]

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation."""
    if args.local_rank == -1:
        return scalar
    scalar_t = (
        torch.tensor(scalar, dtype=torch.float, device=args.device)
        / torch.distributed.get_world_size()
    )
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def get_data_loaders_new(args, tokenizer):
    train_data = get_dataset(
        tokenizer,
        args.train_path,
        args.feature_path,
        args.visual_feature_width,
    )
    # with open("train_data_gpt2.pkl", "rb") as f:
    #    train_data = pkl.load(f)
    # pkl.dump(train_data, f)
    valid_data = get_dataset(
        tokenizer,
        args.valid_path,
        args.feature_path,
        args.visual_feature_width,
    )
    # with open("valid_data_gpt2.pkl", "rb") as f:
    #    valid_data = pkl.load(f)
    # pkl.dump(valid_data, f)
    train_dataset = MemoryDialogDataset(
        train_data[0],
        tokenizer,
        (train_data[1], valid_data[1]),
        drop_rate=0,
        train=True,
    )
    valid_dataset = MemoryDialogDataset(
        valid_data[0],
        tokenizer,
        (valid_data[1], train_data[1]),
        drop_rate=0,
        train=False,
    )
    # for ii in range(len(train_dataset)):
    #    train_dataset[ii]
    # batch = [train_dataset[ii] for ii in range(3)]
    # features = True
    # collate_fn(batch, tokenizer.pad_token_id, features=features)
    # NOTE: FIX this later.
    # features = None if args.video_agnostic else True
    features = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=1,
        shuffle=(not args.distributed),
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=features),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=1,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=features),
    )
    return train_loader, valid_loader


def read_commandline_options():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train_set4DSTC7-AVSD.json",
        help="Path of the trainset",
    )
    parser.add_argument(
        "--feature_path", type=str, default="data/", help="Path to features"
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default="data/valid_set4DSTC7-AVSD.json",
        help="Path of the validset",
    )
    parser.add_argument(
        "--special_tokens_path",
        type=str,
        required=True,
        help="Path to the special tokens for training",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="gpt2",
        help="Path, url or short name of the model",
    )
    parser.add_argument(
        "--max_history",
        type=int,
        default=3,
        help="Number of previous exchanges to keep in history",
    )
    parser.add_argument(
        "--visual_feature_width",
        type=int,
        default=10,
        help="Feature width for each image; 10 - BUTD; 1 - others"
    )
    parser.add_argument(
        "--visual_feature_size",
        type=int,
        default=2053,
        help="Feature size for each image; 2053 - BUTD; 512 - CLIP",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=4, help="Batch size for validation"
    )
    parser.add_argument(
        "--drop_rate", type=float, default=0.5, help="drop rate for caption"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Accumulate gradients on several steps",
    )
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Clipping gradient norm"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=8, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_before_start",
        action="store_true",
        help="If true start with a first evaluation before training",
    )
    parser.add_argument(
        "--dataloader_dry_run",
        action="store_true",
        help="Flag to set only dataloader components",
    )
    parser.add_argument(
        "--video_agnostic",
        action="store_true",
        help="Ignore video features",
    )
    parser.add_argument(
        "--predict_belief_state", action="store_true", help="Predict belief state"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--fp16",
        type=str,
        default="",
        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (-1: not distributed)",
    )
    parser.add_argument("--log_path", type=str, default="log/", help="Log path")
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args, parser, unknown


def train(args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.warning(
        "Running process %d", args.local_rank
    )  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = args.local_rank != -1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    logger.info(
        "Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning"
    )
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    # Read special tokens from the file.
    with open(args.special_tokens_path, "r") as file_id:
        special_tokens_dict = json.load(file_id)
    tokenizer.add_special_tokens(special_tokens_dict)

    if not args.dataloader_dry_run:
        model_class = VideoGPT2LMHeadModel
        model = model_class.from_pretrained(args.model_checkpoint)
        # model_config = model_class.config_class.from_pretrained(args.model_checkpoint)
        # model = model_class(
        #     model_config,
        #     custom_args={"visual_feature_size": args.visual_feature_size},
        # )
        model.resize_token_embeddings(len(tokenizer))
        model.to(args.device)
        optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare model for FP16 and distributed training if needed
    # (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    logger.info("Prepare datasets")
    train_loader, val_loader = get_data_loaders_new(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        # Process the input_tokens for the batch.
        input_embeds = []
        for datum in batch[0]:
            instance_embeds = []
            for datum_input in datum:
                if isinstance(datum_input, dict):
                    datum_output = model.video_ff(
                        torch.Tensor(datum_input["features"]).to(args.device)
                    )
                else:
                    datum_output = model.transformer.wte(datum_input.to(args.device))
                instance_embeds.append(datum_output)
            input_embeds.append(torch.cat(instance_embeds))
        input_embeds, _ = padding(input_embeds, tokenizer.pad_token_id)
        token_type_ids = batch[1].to(args.device)
        lm_labels = batch[2].to(args.device)
        input_mask = batch[3].to(args.device)
        reply_mask = torch.zeros(
            input_mask.size(), dtype=input_mask.dtype, device=input_mask.device
        )
        reply_loss = model(
            input_embeds,
            token_type_ids=token_type_ids,
            labels=(lm_labels, None),
            attention_mask=[reply_mask, input_mask],
            mode="reply",
        )[0]
        loss = reply_loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            # Process the input_tokens for the batch.
            input_embeds = []
            for datum in batch[0]:
                instance_embeds = []
                for datum_input in datum:
                    if isinstance(datum_input, dict):
                        datum_output = model.video_ff(
                            torch.Tensor(datum_input["features"]).to(args.device)
                        )
                    else:
                        datum_output = model.transformer.wte(
                            datum_input.to(args.device)
                        )
                    instance_embeds.append(datum_output)
                input_embeds.append(torch.cat(instance_embeds))
            input_embeds, _ = padding(input_embeds, tokenizer.pad_token_id)
            token_type_ids = batch[1].to(args.device)
            lm_labels = batch[2].to(args.device)
            input_mask = batch[3].to(args.device)
            reply_mask = torch.zeros(
                input_mask.size(), dtype=input_mask.dtype, device=input_mask.device
            )
            model_outputs = model(
                input_embeds,
                token_type_ids=token_type_ids,
                attention_mask=[reply_mask, input_mask],
                mode="reply",
            )[0]

            lm_logits = model_outputs  # So we can also use GPT2 outputs
            lm_logits_flat_shifted = (
                lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            )
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader)
    )
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(
        optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)]
    )
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {
        "nll": Loss(
            torch.nn.CrossEntropyLoss(ignore_index=-1),
            output_transform=lambda x: (x[0], x[1]),
        )
    }
    metrics.update(
        {"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)}
    )
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and 
    # save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(
            Events.COMPLETED,
            lambda _: pbar.log_message(
                "Validation: %s" % pformat(evaluator.state.metrics)
            ),
        )

        tb_logger = TensorboardLogger(log_dir="./tb_logs")
        tb_logger.attach(
            trainer,
            log_handler=OutputHandler(tag="training", metric_names=["loss"]),
            event_name=Events.ITERATION_COMPLETED,
        )
        tb_logger.attach(
            trainer,
            log_handler=OptimizerParamsHandler(optimizer),
            event_name=Events.ITERATION_STARTED,
        )
        tb_logger.attach(
            evaluator,
            log_handler=OutputHandler(
                tag="validation",
                metric_names=list(metrics.keys()),
                another_engine=trainer,
            ),
            event_name=Events.EPOCH_COMPLETED,
        )

        checkpoint_handler = ModelCheckpoint(
            args.log_path, "checkpoint", save_interval=1, n_saved=args.n_epochs, require_empty=False
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            checkpoint_handler,
            {"mymodel": getattr(model, "module", model)},
        )  # "getattr" take care of distributed encapsulation

        torch.save(args, args.log_path + "model_training_args.bin")
        getattr(model, "module", model).config.to_json_file(
            os.path.join(args.log_path, CONFIG_NAME)
        )
        tokenizer.save_vocabulary(args.log_path)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last 
    # checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(
            checkpoint_handler._saved[-1][1][-1],
            os.path.join(args.log_path, WEIGHTS_NAME),
        )  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    args, _, _ = read_commandline_options()
    train(args)
