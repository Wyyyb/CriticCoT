
source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

export CUDA_VISIBLE_DEVICES=6,7

summary_path="../baseline_eval_results_0503/Phi-4-reasoning-plus/summary.txt"
model_path="/data/yubo/models/Phi-4-reasoning-plus"
output_path="../baseline_eval_results_0503/Phi-4-reasoning-plus/"

mkdir -p $output_path

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts


bash evaluate_phi4_8_re.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_0503/Phi-4-reasoning/summary.txt"
model_path="/data/yubo/models/Phi-4-reasoning"
output_path="../baseline_eval_results_0503/Phi-4-reasoning/"

mkdir -p $output_path

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts


bash evaluate_phi4_8_re.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_0503/Phi-4/summary.txt"
model_path="/data/yubo/models/Phi-4"
output_path="../baseline_eval_results_0503/Phi-4/"

mkdir -p $output_path

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts


bash evaluate_phi4_base_8_re.sh $model_path $output_path $summary_path

