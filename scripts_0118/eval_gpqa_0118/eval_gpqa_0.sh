set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
#cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts_0118/eval_gpqa_0118
cd /gpfs/public/research/xy/yubowang/CriticCoT/math_eval_original/

model_dir="/gpfs/public/research/xy/yubowang/models"

model_names=("MAmmoTH-Critique-1")


for model in "${model_names[@]}";do
  model_path="${model_dir}/${model}"
  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_gpqa_0118_cft/gpqa_results/${model}"
  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_gpqa_0118_cft/gpqa_results/${model}_gpqa_summary.txt"
  mkdir -p $output_dir
  bash eval_gpqa.sh $model_path $output_dir $summary_dir 1
  bash eval_gpqa.sh $model_path $output_dir $summary_dir 2
  bash eval_gpqa.sh $model_path $output_dir $summary_dir 3
  bash eval_gpqa.sh $model_path $output_dir $summary_dir 4
  bash eval_gpqa.sh $model_path $output_dir $summary_dir 6
done





