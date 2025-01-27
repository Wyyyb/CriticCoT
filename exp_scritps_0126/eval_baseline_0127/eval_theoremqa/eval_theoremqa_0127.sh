set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1

cd /gpfs/public/research/xy/yubowang/CriticCoT/exp_scritps_0126/eval_baseline_0127/eval_theoremqa/

model_dir="/gpfs/public/research/xy/yubowang/models"
#model="CFT-Webinstruct-0121-ckpt"
#model_path="${model_dir}/${model}"
#n_shots=(0)
#
#for n_shot in "${n_shots[@]}";do
#  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0128/theoremqa_qwen_base_results_${n_shot}_shot_qwen/${model}"
#  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0128/theoremqa_qwen_base_results_${n_shot}_shot_qwen/${model}_general_summary.txt"
#  mkdir -p $output_dir
#  bash run_theoremqa_qwen_code_base_0127.sh $model_path $output_dir $summary_dir $n_shot
#done

model="Sky-T1-32B-Preview"
model_path="${model_dir}/${model}"
n_shots=(0)

for n_shot in "${n_shots[@]}";do
  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0128/theoremqa_qwen_base_results_${n_shot}_shot_qwen/${model}"
  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0128/theoremqa_qwen_base_results_${n_shot}_shot_qwen/${model}_general_summary.txt"
  mkdir -p $output_dir
  bash run_theoremqa_qwen_code_base_0127.sh $model_path $output_dir $summary_dir $n_shot
done


model="CFT-32B-Instruct-Webinstruct-0127-ckpt-3"
model_path="${model_dir}/${model}"
n_shots=(0)

for n_shot in "${n_shots[@]}";do
  output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0128/theoremqa_qwen_base_results_${n_shot}_shot_qwen/${model}"
  summary_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_results_baseline_0128/theoremqa_qwen_base_results_${n_shot}_shot_qwen/${model}_general_summary.txt"
  mkdir -p $output_dir
  bash run_theoremqa_qwen_code_base_0127.sh $model_path $output_dir $summary_dir $n_shot
done



