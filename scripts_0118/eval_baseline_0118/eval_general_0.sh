set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts_0118/eval_baseline_0118

model_dir="/gpfs/public/research/xy/yubowang/models"
#model_names=("Abel-7B-002" "deepseek-math-7b-base" "gemma-2-9b" "internlm2-math-7b" "Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Llama-3-8B" "Qwen2.5-7B" "Qwen2.5-Math-7B" "Qwen2.5-Math-7B-Instruct" "Qwen2-7B" "Qwen2-Math-7B" "rho-math-7b-v0.1" "WizardMath-7B-V1.1")
# model_names=("Qwen2.5-7B" "Qwen2.5-Math-7B" "Qwen2.5-Math-7B-Instruct")
model_names=("MAmmoTH-Critique-1")

#output_dir="/data/yubowang/CriticCoT/eval_results_darth/general_results"
#summary_dir="/data/yubowang/CriticCoT/eval_results_darth/general_summary"

#for model in "${model_names[@]}";do
#  echo "${model}"
#done

for model in "${model_names[@]}";do
  model_path="${model_dir}/${model}"
  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0118/general_results/${model}"
  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0118/general_results/${model}_general_summary.txt"
  mkdir -p $output_dir
  bash run_general_reasoning.sh $model_path $output_dir $summary_dir

done





