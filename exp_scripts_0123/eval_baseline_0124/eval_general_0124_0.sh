set -ex

#source /gpfs/public/research/miniconda3/bin/activate
#conda activate lf_yubo
source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /cpfs/data/user/yubowang/CriticCoT/exp_scripts_0123/eval_baseline_0124/

#model_dir="/gpfs/public/research/xy/yubowang/models"
#
#model_names=("MAmmoTH-Critique-1" "Llama-3.1-8B-Instruct" "AceMath-7B-Instruct" "Eurus-2-7B-SFT" "Llama-3.1-70B-Instruct" "NuminaMath-72B-CoT" "Qwen2.5-Math-72B-Instruct")
model_dir="/cpfs/data/user/yubowang/models"
model_names=("CFT-Webinstruct-0121-ckpt")

for model in "${model_names[@]}";do
  model_path="${model_dir}/${model}"
  output_dir="/cpfs/data/user/yubowang/CriticCoT/eval_results_baseline_0124/general_results/${model}"
  summary_dir="/cpfs/data/user/yubowang/CriticCoT/eval_results_baseline_0124/general_results/${model}_general_summary.txt"
  mkdir -p $output_dir
  bash run_general_reasoning.sh $model_path $output_dir $summary_dir 0

done





