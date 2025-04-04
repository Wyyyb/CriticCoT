set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo
#source /cpfs/data/shared/public/miniconda3/bin/activate
#conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /gpfs/public/research/xy/yubowang/CriticCoT/exp_scritps_0126/eval_baseline_0127/eval_mmlu-pro/

model_dir="/gpfs/public/research/xy/yubowang/models"
#model_names=("Abel-7B-002" "deepseek-math-7b-base" "gemma-2-9b" "internlm2-math-7b" "Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Llama-3-8B" "Qwen2.5-7B" "Qwen2.5-Math-7B" "Qwen2.5-Math-7B-Instruct" "Qwen2-7B" "Qwen2-Math-7B" "rho-math-7b-v0.1" "WizardMath-7B-V1.1")
#model_names=("deepseek-math-7b-base" "internlm2-math-7b" "Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Qwen2.5-7B" "Qwen2.5-Math-7B" "Qwen2.5-Math-7B-Instruct" "Qwen2-Math-7B" "rho-math-7b-v0.1" "WizardMath-7B-V1.1")
# model_names=("Qwen2.5-7B" "Qwen2.5-Math-7B" "Qwen2.5-Math-7B-Instruct")
#model_names=("MAmmoTH-Critique-1" "AceMath-7B-Instruct" "Eurus-2-7B-SFT" "NuminaMath-72B-CoT" "Qwen2.5-Math-72B-Instruct")
model_names=("Sky-T1-32B-Preview")

#for model in "${model_names}"; do
#  echo ${model}
#done

for model in "${model_names[@]}"; do
  model_path="${model_dir}/${model}"
  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0127/mmlu-pro_results_t1/${model}"
  mkdir -p ${output_dir}
  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0127/${model}_mmlu-pro_summary_t1.txt"
  bash run_mmlu_pro.sh $model_path $output_dir $summary_dir 5

done





