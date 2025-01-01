#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5
cd ..
dataset='gsm8k'

HF_MODEL='/ML-A100/team/mm/zhangge/leakage_influence/llama_factory/output_models_0707/gsm8k_test_0_0_150/mistral-7b/'
output_file="outputs/mistral-7b-math_test_0_0_150.jsonl"

python run_open.py \
  --model $HF_MODEL \
  --shots 4 \
  --dataset $dataset \
  --form short \
  --output $output_file