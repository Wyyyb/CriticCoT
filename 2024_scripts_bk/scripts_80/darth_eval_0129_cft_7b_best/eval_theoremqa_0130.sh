set -ex

export CUDA_VISIBLE_DEVICES=7

model_dir="/data/yubowang/models"
model="CFT-Webinstruct-0121-ckpt"
model_path="${model_dir}/${model}"
n_shots=(0 5 1 2 3 4)

for n_shot in "${n_shots[@]}";do
  output_dir="/data/yubowang/CriticCoT/eval_results_baseline_0130/theoremqa_qwen_base_results_${n_shot}_shot_qwen/${model}"
  summary_dir="/data/yubowang/CriticCoT/eval_results_baseline_0130/theoremqa_qwen_base_results_${n_shot}_shot_qwen/${model}_general_summary.txt"
  mkdir -p $output_dir
  bash run_theoremqa_qwen_code_base_0130.sh $model_path $output_dir $summary_dir $n_shot
done

