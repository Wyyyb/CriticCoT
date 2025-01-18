set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts_0118/eval_baseline_0118
model_dir="/gpfs/public/research/xy/yubowang/models"

model_names=("MAmmoTH-Critique-1" "AceMath-7B-Instruct" "Eurus-2-7B-SFT" "NuminaMath-72B-CoT" "Qwen2.5-Math-72B-Instruct")


for model in "${model_names[@]}"; do
  model_path="${model_dir}/${model}"
  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0118/math_hard_results/${model}"
  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0118/${model}_math_hard_summary.txt"
  bash run_math_eval.sh $model_path $output_dir $summary_dir

done





