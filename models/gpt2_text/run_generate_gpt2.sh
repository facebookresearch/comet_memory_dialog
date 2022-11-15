# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, multi-modal)
python3 -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/model_run0/checkpoint-23000 \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/mem_dials_test_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/mem_dials_test_predicted_run0.txt
