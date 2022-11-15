# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    PATH_DATA_DIR=$(realpath ../dialog_simulator/final_data)
else
    PATH_DIR=$(realpath "$1")
    PATH_DATA_DIR=$(realpath "$2")
fi

# Train split
python3 -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/mem_dials_train_v2.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/mem_dials_train_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/mem_dials_train_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/mem_special_tokens.json

# --use_multimodal_contexts=1 \
# Dev split
python3 -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/mem_dials_val_v2.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/mem_dials_val_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/mem_dials_val_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/mem_special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/mem_special_tokens.json \

# Devtest split
python3 -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/mem_dials_test_v2.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/mem_dials_test_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/mem_dials_test_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/mem_special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/mem_special_tokens.json \

# Test split
# python3 -m gpt2_dst.scripts.preprocess_input \
#     --input_path_json="${PATH_DATA_DIR}"/mem_dials_test.json \
#     --output_path_predict="${PATH_DIR}"/gpt2_dst/data/mem_dials_test_predict.txt \
#     --output_path_target="${PATH_DIR}"/gpt2_dst/data/mem_dials_test_target.txt \
#     --len_context=2 \
#     --use_multimodal_contexts=1 \
#     --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/mem_special_tokens.json \
#     --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/mem_special_tokens.json \

# Mini split
# python3 -m gpt2_dst.scripts.preprocess_input \
#     --input_path_json="${PATH_DATA_DIR}"/mem_dials_mini.json \
#     --output_path_predict="${PATH_DIR}"/gpt2_dst/data/mem_dials_mini_predict.txt \
#     --output_path_target="${PATH_DIR}"/gpt2_dst/data/mem_dials_mini_target.txt \
#     --len_context=2 \
#     --use_multimodal_contexts=1 \
#     --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/mem_special_tokens.json \
#     --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/mem_special_tokens.json \
