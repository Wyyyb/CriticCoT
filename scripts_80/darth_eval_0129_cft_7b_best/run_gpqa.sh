set -ex

model_path=$1
output_dir=$2
summary_path=$3
n_shot=$4
prompt=$5

cd /data/yubowang/CriticCoT/math_eval_original

datasets=("gpqa_diamond")

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

