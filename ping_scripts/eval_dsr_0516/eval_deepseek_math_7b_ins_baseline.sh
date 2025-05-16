
# source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

export CUDA_VISIBLE_DEVICES=4,5,6,7

summary_path="../baseline_eval_results_0516/summary.txt"
model_path="/data/yubo/models/deepseek-math-7b-instruct"
output_path="../baseline_eval_results_0516/deepseek-math-7b-instruct/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_deepseek.sh $model_path $output_path $summary_path
