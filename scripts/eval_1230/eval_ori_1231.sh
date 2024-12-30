#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness

mkdir -p ../math_eval_result/eval_res_1231_ori/

model_path="/gpfs/public/research/xy/yubowang/models/OLMo-2-1124-7B"
output_dir="../math_eval_result/eval_res_1231_ori/ori_OLMo-2-1124-7B"
bash scripts/run_eval_math.sh cot $model_path $output_dir

model_path="/gpfs/public/research/xy/yubowang/models/deepseek-math-7b-base"
output_dir="../math_eval_result/eval_res_1231_ori/ori_deepseek-math-7b-base"
bash scripts/run_eval_math.sh cot $model_path $output_dir




