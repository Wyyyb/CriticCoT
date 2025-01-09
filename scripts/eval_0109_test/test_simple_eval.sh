#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd ../../

python -m simple-evals.simple_evals --list-models