set -ex

model_path=$1
output_dir=$2
summary_path=$3

#cd /gpfs/public/research/xy/yubowang/CriticCoT/MMLU-Pro
#bash mmlu-pro-eval.sh $model_path $output_dir $summary_path
#cd /gpfs/public/research/xy/yubowang/CriticCoT/math_eval

# datasets=("math_500" "math" "gsm8k" "theoremqa" "mmlu_stem" "sat" "aime" "aime_24" "OlympiadBench")
datasets=("math_500" "math" "gsm8k")

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python run_open_test_shot_6.py \
        --model $model_path \
        --shots 5 \
        --dataset $dataset \
        --form few-shot \
        --output_dir $output_dir \
        --summary_path $summary_path
done

