# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi


python -m gpt2_dst.scripts.evaluate_dst_response \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/mem_dials_test_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/mem_dials_test_predicted.txt \
    --compute_bert_score
