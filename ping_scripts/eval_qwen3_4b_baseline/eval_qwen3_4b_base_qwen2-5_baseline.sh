
# source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

export CUDA_VISIBLE_DEVICES=4,7

summary_path="../baseline_eval_results_0517_2-5/summary.txt"
model_path="/data/yubo/models/Qwen3-4B-Base"
output_path="../baseline_eval_results_0517_2-5/Qwen3-4B-Base_qwen2-5/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path
