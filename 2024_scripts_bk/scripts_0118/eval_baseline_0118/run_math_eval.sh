set -ex

model_path=$1
output_dir=$2
summary_path=$3
# n_shot=$4

cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
bash eval_math_hard.sh $model_path $output_dir $summary_path

