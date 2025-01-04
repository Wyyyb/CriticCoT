set -ex

model_path=$1
output_dir=$2

summary_path="../math_eval_result_new/summary_0105_v3.txt"
# datasets=("math_500" "math" "gsm8k" "theoremqa" "mmlu_stem" "sat")
#datasets=("math_500" "math")
# datasets=("aime" "OlympiadBench" "aime_24")
datasets=("aime_24")

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
