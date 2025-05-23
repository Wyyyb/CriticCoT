set -ex

MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
SUMMARY_PATH=$3


python3 -u run_evaluate.py \
    --model_path ${MODEL_NAME_OR_PATH} \
    --output_dir_path ${OUTPUT_DIR} \
    --summary_path ${SUMMARY_PATH} \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_token 4096




