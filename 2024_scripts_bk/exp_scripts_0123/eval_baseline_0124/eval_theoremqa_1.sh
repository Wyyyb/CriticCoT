set -ex

source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1

cd /cpfs/data/user/yubowang/CriticCoT/exp_scripts_0123/eval_baseline_0124/

model_dir="/cpfs/data/user/yubowang/models"
model="CFT-Webinstruct-0121-ckpt"
model_path="${model_dir}/${model}"
n_shots=(0 1 2 3 4 5)

for n_shot in "${n_shots[@]}";do
  output_dir="/cpfs/data/user/yubowang/CriticCoT/eval_results_baseline_0125/theoremqa_results_${n_shot}_shot_qwen/${model}"
  summary_dir="/cpfs/data/user/yubowang/CriticCoT/eval_results_baseline_0125/theoremqa_results_${n_shot}_shot_qwen/${model}_general_summary.txt"
  mkdir -p $output_dir
  bash run_theoremqa.sh $model_path $output_dir $summary_dir $n_shot "qwen"
done