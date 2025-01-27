set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1

cd /gpfs/public/research/xy/yubowang/CriticCoT/exp_scripts_0123/eval_baseline_0124/

model_dir="/gpfs/public/research/xy/yubowang/models"
model="CFT-32B-Instruct-Webinstruct-0127-ckpt-3"
model_path="${model_dir}/${model}"
n_shots=(0 1 2 3 4 5)

for n_shot in "${n_shots[@]}";do
  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0127/gpqa_results_${n_shot}_shot_gpqa/${model}"
  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0127/gpqa_results_${n_shot}_shot_gpqa/${model}_general_summary.txt"
  mkdir -p $output_dir
  bash run_gpqa.sh $model_path $output_dir $summary_dir $n_shot "gpqa"
done