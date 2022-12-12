# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

#Test flags.
LOG_PATH="logs/"
MODEL_EPOCH=6

python3 visualize_model_outputs.py --memory_gt_json "data/mem_dials_test.json" \
--model_dst_output_jsons "$LOG_PATH/model_ep${MODEL_EPOCH}_generate_dst_results.json" \
--model_response_output_jsons "$LOG_PATH/model_ep${MODEL_EPOCH}_generate_response_results.json" \
--memory_files "data/mscoco_memory_graphs_1k.json" "data/memory_may21_v1_100graphs.json" \
--visualize_memories
