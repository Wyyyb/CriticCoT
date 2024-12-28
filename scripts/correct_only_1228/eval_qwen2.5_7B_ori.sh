#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness

model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-7B"
output_dir="../math_eval_result/eval_res_1228/ori_qwen2.5_7B/"
bash scripts/run_eval.sh cot $model_path $output_dir






