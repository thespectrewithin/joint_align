#!/bin/bash

set -e

mkdir -p cache
mkdir -p log

GPU=0
MATRIX=europarl-v7.en-es.1k
CACHE=./cache

MODEL=bert-base-multilingual-cased
FEATURE=$MODEL.sum-last-4

BASE_PATH=./
ENG_TRAIN=$BASE_PATH/eng.train
ESP_TESTA=$BASE_PATH/esp.testa
ESP_TESTB=$BASE_PATH/esp.testb
ENG_TRAIN_FEAT_ALIGNED=$ENG_TRAIN.$FEATURE.aligned-to-es.features.pth
ESP_TESTA_FEAT=$ESP_TESTA.$FEATURE.features.pth
ESP_TESTB_FEAT=$ESP_TESTB.$FEATURE.features.pth

# extract and align the features using the provided alignment matrices
CUDA_VISIBLE_DEVICES=$GPU python feature_extraction.py --bert_model $MODEL --cache_dir $CACHE --data $ENG_TRAIN --align_matrix $MATRIX --layers 9:13 --output $ENG_TRAIN_FEAT_ALIGNED
CUDA_VISIBLE_DEVICES=$GPU python feature_extraction.py --bert_model $MODEL --cache_dir $CACHE --data $ESP_TESTA --layers 9:13 --output $ESP_TESTA_FEAT
CUDA_VISIBLE_DEVICES=$GPU python feature_extraction.py --bert_model $MODEL --cache_dir $CACHE --data $ESP_TESTB --layers 9:13 --output $ESP_TESTB_FEAT

# feed the features to a task-specific model for a downstream task
for seed in 1001 2112 3223 4334 5445; do
    CUDA_VISIBLE_DEVICES=$GPU python feature_ner.py \
			--train_data $ENG_TRAIN \
			--train_feat $ENG_TRAIN_FEAT_ALIGNED \
			--dev_data $ESP_TESTA \
			--dev_feat $ESP_TESTA_FEAT \
			--test_data $ESP_TESTB \
			--test_feat $ESP_TESTB_FEAT \
			--bert_model=$MODEL \
			--log_file ./log/eng-esp.$FEATURE.aligned-to-esp.$SEED \
			--seed $seed \
			--log_interval 150
done
