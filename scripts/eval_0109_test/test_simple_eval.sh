#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export OPENAI_API_KEY="sk-proj-Smk0AMd6n3LRT3g585BJT3BlbkFJC4lheINPeNTfg6ucK9b0"

cd ../../

python -m simple-evals.simple_evals --list-models