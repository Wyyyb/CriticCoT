#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,7
cd ..
dataset='gsm8k'

HF_MODEL='/mnt/tjena/yubo/leakage/output_models_0707/gsm8k_test_0_0_30/mistral-7b/'
output_file="outputs/mistral-7b-gsm8k_test_0_0_30.jsonl"

python run_open.py \
  --model $HF_MODEL \
  --shots 4 \
  --dataset $dataset \
  --form short \
  --output $output_file
