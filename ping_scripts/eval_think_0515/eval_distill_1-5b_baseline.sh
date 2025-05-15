
# source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

export CUDA_VISIBLE_DEVICES=4,5,6,7

summary_path="../baseline_eval_results_0515_qwen3/summary.txt"
model_path="/data/yubo/models/DeepSeek-R1-Distill-Qwen-1.5B"
output_path="../baseline_eval_results_0515_qwen3/DeepSeek-R1-Distill-Qwen-1,5B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen3.sh $model_path $output_path $summary_path

summary_path="../baseline_eval_results_0515_qwen2-5/summary.txt"
model_path="/data/yubo/models/DeepSeek-R1-Distill-Qwen-1.5B"
output_path="../baseline_eval_results_0515_qwen2-5/DeepSeek-R1-Distill-Qwen-1.5B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path

