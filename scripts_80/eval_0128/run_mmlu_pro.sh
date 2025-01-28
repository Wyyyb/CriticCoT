set -ex

model_path=$1
output_dir=$2
summary_path=$3
n_shot=$4

cd /data/yubo/CriticCoT/MMLU-Pro
bash mmlu-pro-eval.sh $model_path $output_dir $summary_path $n_shot

