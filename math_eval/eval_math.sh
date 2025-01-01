set -ex

model_sub_dir=$1
output_dir=$2

#source /gpfs/public/research/miniconda3/bin/activate
#conda activate lf_yubo

#export CUDA_VISIBLE_DEVICES=0,1
#cd ../../math_eval
summary_path="../math_eval_result_new/summary.txt"
datasets=("math" "gsm8k" "theoremqa" "mmlu_stem" "sat" "svamp")
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory"

# model_sub_dir="output_models/CriticCoT_correct_only_data_1231/checkpoint-723"
model_path="${model_dir}/${model_sub_dir}"
# output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_correct_only_1231_ckpt_723/"

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python run_open.py \
        --model $model_path \
        --shots 5 \
        --dataset $dataset \
        --form short \
        --output_dir $output_dir \
        --summary_path $summary_path







