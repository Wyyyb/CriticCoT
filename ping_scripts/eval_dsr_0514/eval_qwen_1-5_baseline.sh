
# source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

export CUDA_VISIBLE_DEVICES=2,3

summary_path="../baseline_eval_results_0515/summary.txt"
model_path="/data/yubo/models/Qwen2.5-Math-1.5B"
output_path="../baseline_eval_results_0515/Qwen2.5-Math-1.5B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path

summary_path="../baseline_eval_results_0515/summary.txt"
model_path="/data/yubo/models/Qwen2.5-Math-1.5B-Instruct"
output_path="../baseline_eval_results_0515/Qwen2.5-Math-1.5B-Instruct/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path

