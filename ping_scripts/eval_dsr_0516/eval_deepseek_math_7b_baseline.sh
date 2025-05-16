
# source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

export CUDA_VISIBLE_DEVICES=0,1

summary_path="../baseline_eval_results_0516/summary.txt"
model_path="/data/yubo/models/deepseek-math-7b-base"
output_path="../baseline_eval_results_0516/Llama-3.2-3B-Instruct/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_deepseek.sh $model_path $output_path $summary_path
