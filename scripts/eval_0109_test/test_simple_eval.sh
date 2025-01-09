#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd ../../simple-evals-main

python simple_evals.py --list-models