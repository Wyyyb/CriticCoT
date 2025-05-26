

summary_path="../../baseline_eval_results_0523/qwen2-5_math_7b_ins/summary.txt"
model_path="/data/yubowang/models/Qwen2.5-Math-7B-Instruct"
output_path="../../baseline_eval_results_0523/qwen2-5_math_7b_ins/"

cd /data/yubowang/CriticCoT/bbeh/bbeh

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=6,7

bash eval_qwen2-5.sh $model_path $output_path $summary_path




