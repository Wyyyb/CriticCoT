set -ex

model_path=$1
output_dir=$2

summary_path="../math_eval_result_new/summary_0105.txt"
# datasets=("math_500" "math" "gsm8k" "theoremqa" "mmlu_stem" "sat")
datasets=("math" "math_500")

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python run_open.py \
        --model $model_path \
        --shots 5 \
        --dataset $dataset \
        --form few-shot \
        --output_dir $output_dir \
        --summary_path $summary_path
done
