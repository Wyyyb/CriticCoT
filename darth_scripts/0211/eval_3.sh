set -ex

export CUDA_VISIBLE_DEVICES=7

model_path="/data/yubowang/models/CFT-Webinstruct-0121-ckpt"
output_dir="/data/yubowang/CriticCoT/eval_result_0211/t_0_0/"
summary_path="/data/yubowang/CriticCoT/eval_result_0211/t_0_0/summary.txt"
temp=0.3

cd /data/yubowang/CriticCoT/Qwen2.5-Math-Eval-0203/scripts/
bash eval_math_only.sh $model_path $output_dir $summary_path $temp

