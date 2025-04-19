#!/bin/bash
set -ex

summary_path=$1
model_dir=$2

#source /cpfs/data/shared/public/miniconda3/bin/activate
#conda activate lf_yubo

bash eval_sinlge_card_1.sh $summary_path $model_dir&
bash eval_sinlge_card_2.sh $summary_path $model_dir&
bash eval_sinlge_card_3.sh $summary_path $model_dir&
bash eval_sinlge_card_4.sh $summary_path $model_dir
