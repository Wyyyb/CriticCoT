#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness
bash scripts/run_eval.sh cot /gpfs/public/research/xy/yubowang/models/Qwen2.5-7B






