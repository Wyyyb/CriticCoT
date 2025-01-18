set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1

cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts_0118/eval_baseline_0118
model_dir="/gpfs/public/research/xy/yubowang/models"

#model_names=("MAmmoTH-Critique-1" "AceMath-7B-Instruct" "deepseek-math-7b-instruct" "Llama-3.1-8B-Instruct" "NuminaMath-72B-CoT" "Qwen2.5-Math-72B-Instruct" "Llama-3.1-70B-Instruct")

model_names=("deepseek-math-7b-instruct" "Llama-3.1-8B-Instruct" "Llama-3.1-70B-Instruct")

for model in "${model_names[@]}"; do
  model_path="${model_dir}/${model}"
  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0118/math_hard_results/${model}"
  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0118/${model}_math_hard_summary.txt"
  mkdir -p $output_dir
  bash run_math_eval.sh $model_path $output_dir $summary_dir

done





