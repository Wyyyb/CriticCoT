#!/bin/bash
set -ex

# 激活conda环境
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 切换到正确的目录
cd /gpfs/public/research/xy/yubowang/CriticCoT/critique_on_math/

ckpt="checkpoint-80"
model_path="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0114/qwen2.5-math-7B_t2_critic_0114/${ckpt}"
BASE_DIR="/gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/math_eval_result_MAmmoTH-Critique-1_ckpt-80_0116"
input_dir="${BASE_DIR}/qwen_eval_res_0116_multi_result/MAmmoTH-Critique-1-0116-${ckpt}/"

python run_multi_critique_on_vllm.py \
  --model_path $model_path \
  --input_dir $input_dir
