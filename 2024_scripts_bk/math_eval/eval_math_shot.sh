set -ex

model_path=$1
output_dir=$2
summary_path=$3

# 定义shots数组和数据集数组
shots_array=(4 5 6 8)
datasets=("math_500" "math" "gsm8k" "theoremqa")

# 外层循环：不同的数据集
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"

    # 内层循环：不同的shots数量
    for n_shot in "${shots_array[@]}"; do
        echo "Testing $dataset with $n_shot shots"

        python run_open.py \
            --model $model_path \
            --shots $n_shot \
            --dataset $dataset \
            --form few-shot \
            --output_dir $output_dir \
            --summary_path $summary_path
    done
done