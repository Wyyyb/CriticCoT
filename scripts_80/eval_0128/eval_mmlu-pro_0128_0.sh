set -ex

#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=1,2,4,5
cd /data/yubo/CriticCoT/scripts_80/eval_0128

model_dir="/data/yubo/models"
model_names=("Qwen2.5-32B-Instruct")

#for model in "${model_names[@]}"; do
#  model_path="${model_dir}/${model}"
#  output_dir="/data/yubo/CriticCoT/eval_results_baseline_0128/mmlu-pro_results_0shot/${model}"
#  mkdir -p ${output_dir}
#  summary_dir="/data/yubo/CriticCoT/eval_results_baseline_0128/${model}_mmlu-pro_summary_0shot.txt"
#  bash run_mmlu_pro.sh $model_path $output_dir $summary_dir 0
#done

for model in "${model_names[@]}"; do
  model_path="${model_dir}/${model}"
  output_dir="/data/yubo/CriticCoT/eval_results_baseline_0128/mmlu-pro_results_5shot/${model}"
  mkdir -p ${output_dir}
  summary_dir="/data/yubo/CriticCoT/eval_results_baseline_0128/${model}_mmlu-pro_summary_5shot.txt"
  bash run_mmlu_pro.sh $model_path $output_dir $summary_dir 5
done



