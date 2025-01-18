set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts_0118/eval_baseline_0118

model_dir="/gpfs/public/research/xy/yubowang/models"

model_names=("MAmmoTH-Critique-1")


for model in "${model_names[@]}";do
  model_path="${model_dir}/${model}"
  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_gpqa_0118/gpqa_results/${model}"
  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_gpqa_0118/gpqa_results/${model}_gpqa_summary.txt"
  mkdir -p $output_dir
  bash run_general_reasoning.sh $model_path $output_dir $summary_dir
done





