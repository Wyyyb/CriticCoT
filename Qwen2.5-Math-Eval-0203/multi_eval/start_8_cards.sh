#!/bin/bash
set -ex

summary_path=$1
model_dir=$2

bash eval_7b_1.sh $summary_path $model_dir &
bash eval_7b_2.sh $summary_path $model_dir &
bash eval_7b_3.sh $summary_path $model_dir &
bash eval_7b_4.sh $summary_path $model_dir &
bash eval_7b_5.sh $summary_path $model_dir &
bash eval_7b_6.sh $summary_path $model_dir &
bash eval_7b_7.sh $summary_path $model_dir &
bash eval_7b_8.sh $summary_path $model_dir



