set -ex

model_path=$1
output_dir=$2
summary_path=$3
n_shot=$4
dataset=$5

#cd /gpfs/public/research/xy/yubowang/CriticCoT/MMLU-Pro
#bash mmlu-pro-eval.sh $model_path $output_dir $summary_path
#cd /gpfs/public/research/xy/yubowang/CriticCoT/math_eval

datasets=($dataset)
# datasets=("math_500" "math")

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python run_open.py \
        --model $model_path \
        --shots $n_shot \
        --dataset $dataset \
        --form few-shot \
        --output_dir $output_dir \
        --summary_path $summary_path
done

