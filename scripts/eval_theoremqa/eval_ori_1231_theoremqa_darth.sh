#!/bin/bash
# source /gpfs/public/research/miniconda3/bin/activate
conda activate critic
export CUDA_VISIBLE_DEVICES=2
cd /data/yubowang/CriticCoT/TheoremQA

mkdir -p ../math_eval_result/eval_res_1231_ori/

model_path="/data/yubowang/models/OLMo-2-1124-7B"
output_dir="../math_eval_result/eval_res_1231_ori/ori_OLMo-2-1124-7B"
output="${output_dir}/theorem_qa_step.jsonl"
python run.py --model $model_path --output $output --form "step"

output="${output_dir}/theorem_qa_short.jsonl"
python run.py --model $model_path --output $output --form "short"

model_path="/data/yubowang/models/deepseek-math-7b-base"
output_dir="../math_eval_result/eval_res_1231_ori/ori_deepseek-math-7b-base"
output="${output_dir}/theorem_qa_step.jsonl"
python run.py --model $model_path --output $output --form "step"

output="${output_dir}/theorem_qa_short.jsonl"
python run.py --model $model_path --output $output --form "short"


