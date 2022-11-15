# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Train (multi-modal)
python3 -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/model_run2 \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/mem_special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/mem_dials_train_target.txt \
    --do_eval --eval_all_checkpoints \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/mem_dials_val_target.txt \
    --num_train_epochs=10 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4
