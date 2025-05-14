
# source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

export CUDA_VISIBLE_DEVICES=0,1,2,3

summary_path="../baseline_eval_results_0514/summary.txt"
model_path="/data/yubo/models/Qwen2.5-Math-7B"
output_path="../baseline_eval_results_0514/Qwen2.5-Math-7B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_0514/summary.txt"
model_path="/data/yubo/models/Qwen2.5-Math-7B-Instruct"
output_path="../baseline_eval_results_0514/Qwen2.5-Math-7B-Instruct/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_0514/summary.txt"
model_path="/data/yubo/models/Qwen2.5-7B"
output_path="../baseline_eval_results_0514/Qwen2.5-7B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_0514/summary.txt"
model_path="/data/yubo/models/Qwen2.5-7B-Instruct"
output_path="../baseline_eval_results_0514/Qwen2.5-7B-Instruct/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path

