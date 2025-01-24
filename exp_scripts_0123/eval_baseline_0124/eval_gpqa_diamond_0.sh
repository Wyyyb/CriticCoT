set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts_0118/eval_baseline_0118

model_dir="/gpfs/public/research/xy/yubowang/models"

model_names=("Llama-3.1-8B-Instruct" "AceMath-7B-Instruct" "Eurus-2-7B-SFT" "Llama-3.1-70B-Instruct" "NuminaMath-72B-CoT" "Qwen2.5-Math-72B-Instruct")


for model in "${model_names[@]}";do
  model_path="${model_dir}/${model}"
  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_gpqa_diamond_0119/gpqa_diamond_results/${model}"
  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_gpqa_diamond_0119/gpqa_diamond_results/${model}_summary.txt"
  mkdir -p $output_dir
  bash run_gpqa_diamond.sh $model_path $output_dir $summary_dir

done





