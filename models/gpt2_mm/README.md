# GPT-2 (MM)
This is the code for the GPT-2 model used in [Navigating Connected Memories with a Task-oriented Dialog System][code]. It is based on the AAAI2020-DSTC8-AVSD paper [Bridging Text and Video: A Universal Multimodal Transformer for Video-Audio Scene-Aware Dialog.](<https://arxiv.org/abs/2002.00163>).


## How to Run

**Requirements**

```
Python. 3.6
torch==1.0.1
pytorch-ignite==0.2.1
transformers==2.1.1
tqdm==4.36.1
```

```shell
pip install -r requirements.txt
```

**Data**

Create a soft link to the `../../data` folder in the main folder here as `data/`.
Please see `run_me.sh` for example of how to run the code.




**Step 1: Preprocess the dataset**

```shell
# Preprocessing the dataset.
python utils/preprocess_memory_dataset.py \
	--train_json_path "data/mem_dials_train.json" \
	--unseen_json_path \
		"data/mem_dials_val.json" \
		"data/mem_dials_test.json" \
	--save_folder "data/gpt2_data/"
```

**Step 2: Extracting the image features**

We use this [repository](https://github.com/vmurahari3/visdial-bert#download-preprocessed-data) to download the image features.

```shell
FEATURE_PATH="/data/img_feats1.0/visdial_img_feat.lmdb"
# Extracting visual features (BUTD features).
python utils/extract_memory_features.py \
	--input_dialog_json data/mem_dials_merged.json \
	--input_memory_json \
		data/memory_may21_v1_100graphs.json \
		data/mscoco_memory_graphs_1k.json \
	--input_feature_path $FEATURE_PATH \
	--max_bboxes 10 \
	--feature_save_path data/memory_features/butd_10w_features/ \
	--feature_type butd
```

**Training**

```shell
FEATURES="butd"
LOG_PATH="logs/"
# Visual features.
FEATURE_PATH="data/memory_features/butd_10w_features/"
VISUAL_FEATURE_SIZE=2053
VISUAL_FEATURE_WIDTH=10

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
```

**Evaluation**

```shell
python generate.py \
	--model_checkpoint $LOG_PATH \
	--model_epoch $MODEL_EPOCH \
	--test_set "data/gpt2_data/mem_dials_gpt2_test.json" \
	--special_tokens_path "data/gpt2_data/mem_dials_gpt2_special_tokens.json" \
	--feature_path $FEATURE_PATH \
	--visual_feature_size $VISUAL_FEATURE_SIZE \
	--visual_feature_width $VISUAL_FEATURE_WIDTH \
	--output <output_path> \
	--max_len 100
```
 
**Compiling Results**

```shell
python utils/create_result_jsons.py \
	--memory_test_json "data/mem_dials_test.json" \
	--model_output_json $OUTPUT_RESULT_FILE
```


## Citation

If you use this code in your research, please cite our paper and the original AAAI 2020 DSTC8 workshop 
paper.

```
@inproceedings{moon-kottur-2022-navigating,
    title = "Navigating Connected Memories with a Task-oriented Dialog System",
    author = "Moon, Seungwhan and 
    Kottur, Satwik Kottur and
    Geramifard, Alborz and
    Damavandi, Babak",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Online and Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
}
```

```
@article{li2020bridging,
    title={Bridging Text and Video: A Universal Multimodal Transformer for Video-Audio Scene-Aware Dialog},
    author={Zekang Li and Zongjia Li and Jinchao Zhang and Yang Feng and Cheng Niu and Jie Zhou},
    year={2020},
    eprint={2002.00163},
    archivePrefix={arXiv},
    journal={CoRR},
    primaryClass={cs.CL}
}
```

[code]:https://github.com/facebookresearch/comet_memory_dialog