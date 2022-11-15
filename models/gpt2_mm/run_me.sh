# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
FEATURES="butd"
LOG_PATH="logs/"
MODE="train"
GPU_ID=0

# Test flags.
MODEL_EPOCH=6
OUTPUT_RESULT_FILE="$LOG_PATH/model_ep${MODEL_EPOCH}_generate.json"

# Visual features.
FEATURE_PATH="data/memory_features/butd_10w_features/"
VISUAL_FEATURE_SIZE=2053
VISUAL_FEATURE_WIDTH=10

case $MODE in
"train")
    echo "Training.."
    # Train Memory Dialog Model.
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python train.py --log_path $LOG_PATH \
        --train_path "data/gpt2_data/mem_dials_gpt2_train.json" \
        --valid_path "data/gpt2_data/mem_dials_gpt2_val.json" \
        --special_tokens_path "data/gpt2_data/mem_dials_gpt2_special_tokens.json" \
        --train_batch_size 8 \
        --predict_belief_state \
        --n_epochs 20 \
        --feature_path $FEATURE_PATH \
        --visual_feature_size $VISUAL_FEATURE_SIZE \
        --visual_feature_width $VISUAL_FEATURE_WIDTH
        ;;

"generate")
    # Generate responses from Memory Dialog Model.
    CUDA_VISIBLE_DEVICES=$GPU_ID \
        python generate.py \
            --model_checkpoint $LOG_PATH \
            --model_epoch $MODEL_EPOCH \
            --test_set "data/gpt2_data/mem_dials_gpt2_test.json" \
            --special_tokens_path "data/gpt2_data/mem_dials_gpt2_special_tokens.json" \
            --feature_path $FEATURE_PATH \
            --visual_feature_size $VISUAL_FEATURE_SIZE \
            --visual_feature_width $VISUAL_FEATURE_WIDTH \
            --output $OUTPUT_RESULT_FILE \
            --max_len 100
            ;;

"compile")
    # Compile results and create JSON files to run standard evaluation.
    python utils/create_result_jsons.py \
        --memory_test_json "data/mem_dials_test.json" \
        --model_output_json $OUTPUT_RESULT_FILE
    ;;
esac


FEATURE_PATH="/data/img_feats1.0/visdial_img_feat.lmdb"
# Extracting visual features (BUTD features).
# python utils/extract_memory_features.py \
#     --input_dialog_json data/mem_dials_merged.json \
#     --input_memory_json \
#         data/memory_may21_v1_100graphs.json \
#         data/mscoco_memory_graphs_1k.json \
#     --input_feature_path $FEATURE_PATH \
#     --max_bboxes 10 \
#     --feature_save_path data/memory_features/butd_10w_features/ \
#     --feature_type butd


# Preprocessing the dataset.
# python utils/preprocess_memory_dataset.py \
#     --train_json_path "data/mem_dials_train.json" \
#     --unseen_json_path \
#         "data/mem_dials_val.json" \
#         "data/mem_dials_test.json" \
#     --save_folder "data/gpt2_data/"
