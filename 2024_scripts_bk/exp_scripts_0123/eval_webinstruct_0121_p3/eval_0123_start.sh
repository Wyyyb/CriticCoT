#!/bin/bash
set -ex

source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

bash eval_0123_1.sh &
bash eval_0123_2.sh &
bash eval_0123_3.sh &
bash eval_0123_4.sh &
#bash eval_0123_5.sh &
#bash eval_0123_6.sh &
#bash eval_0123_7.sh &
#bash eval_0123_8.sh &