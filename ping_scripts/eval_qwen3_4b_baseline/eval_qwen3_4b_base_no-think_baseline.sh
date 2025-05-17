
# source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

export CUDA_VISIBLE_DEVICES=4,7

summary_path="../baseline_eval_results_0517_no_think/summary.txt"
model_path="/data/yubo/models/Qwen3-4B-Base"
output_path="../baseline_eval_results_0517_no_think/Qwen3-4B-Base_no_think/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen3_no_think.sh $model_path $output_path $summary_path

export CUDA_VISIBLE_DEVICES=4,7

summary_path="../baseline_eval_results_0517_no_think/summary.txt"
model_path="/data/yubo/models/Qwen3-4B"
output_path="../baseline_eval_results_0517_no_think/Qwen3-4B_no_think/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen3_no_think.sh $model_path $output_path $summary_path



