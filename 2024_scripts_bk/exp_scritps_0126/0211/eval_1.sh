set -ex

export CUDA_VISIBLE_DEVICES=2

model_path="/map-vepfs/yubo/models/CFT-Webinstruct-0121-ckpt"
output_dir="/map-vepfs/yubo/CriticCoT/eval_result_0211/c_2/"
summary_path="/map-vepfs/yubo/CriticCoT/eval_result_0211/c_2/summary.txt"
temp=0

cd /map-vepfs/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts/
bash eval_math_only.sh $model_path $output_dir $summary_path $temp

