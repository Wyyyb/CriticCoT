set -ex

model_path=$1
output_dir=$2
summary_path=$3
n_shot=$4
prompt=$5

#cd /gpfs/public/research/xy/yubowang/CriticCoT/MMLU-Pro
#bash mmlu-pro-eval.sh $model_path $output_dir $summary_path
cd /cpfs/data/user/yubowang/CriticCoT/math_eval_original

# datasets=("math" "gsm8k" "theoremqa" "mmlu_stem")
#datasets=("gpqa_diamond" "theoremqa" "mmlu_stem")
datasets=("theoremqa")

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python run_open.py \
        --model $model_path \
        --shots $n_shot \
        --dataset $dataset \
        --form $prompt \
        --output_dir $output_dir \
        --summary_path $summary_path
done

