set -ex

#conda activate critic

export CUDA_VISIBLE_DEVICES=4,5
cd ../eval_scripts

model_dir="/data/yubowang/critic_models"
#model_names=("Abel-7B-002" "deepseek-math-7b-base" "gemma-2-9b" "internlm2-math-7b" "Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Llama-3-8B" "Qwen2.5-7B" "Qwen2.5-Math-7B" "Qwen2.5-Math-7B-Instruct" "Qwen2-7B" "Qwen2-Math-7B" "rho-math-7b-v0.1" "WizardMath-7B-V1.1")
#model_names=("deepseek-math-7b-base" "internlm2-math-7b" "Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Qwen2.5-7B" "Qwen2.5-Math-7B" "Qwen2.5-Math-7B-Instruct" "Qwen2-Math-7B" "rho-math-7b-v0.1" "WizardMath-7B-V1.1")
model_names=("deepseek-math-7b-base" "internlm2-math-7b" "Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Qwen2-Math-7B" "rho-math-7b-v0.1" "WizardMath-7B-V1.1" "Qwen2.5-7B" "Qwen2.5-Math-7B" "Qwen2.5-Math-7B-Instruct")

#for model in "${model_names}"; do
#  echo ${model}
#done

for model in "${model_names[@]}"; do
  model_path="${model_dir}/${model}"
  output_dir="/data/yubowang/CriticCoT/eval_results_darth/mmlu-pro_results/${model}"
  summary_dir="/data/yubowang/CriticCoT/eval_results_darth/${model}_mmlu-pro_summary"
  bash eval_mmlu_pro.sh $model_path $output_dir $summary_dir

done





