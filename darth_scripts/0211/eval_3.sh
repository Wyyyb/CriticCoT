set -ex

export CUDA_VISIBLE_DEVICES=3,5,6,7

model_path="/data/yubowang/models/CFT-Webinstruct-0121-ckpt"
output_dir="/data/yubowang/CriticCoT/eval_result_0211/cc_4/"
summary_path="/data/yubowang/CriticCoT/eval_result_0211/cc_4/summary.txt"
temp=0

cd /data/yubowang/CriticCoT/Qwen2.5-Math-Eval-0203/scripts/
bash eval_math_only.sh $model_path $output_dir $summary_path $temp

