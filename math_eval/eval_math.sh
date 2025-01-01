set -ex

model_path=$1
output_dir=$2

summary_path="../math_eval_result_new/summary.txt"
datasets=("math" "gsm8k" "theoremqa" "mmlu_stem" "sat" "svamp")

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python run_open.py \
        --model $model_path \
        --shots 5 \
        --dataset $dataset \
        --form short \
        --output_dir $output_dir \
        --summary_path $summary_path
done
