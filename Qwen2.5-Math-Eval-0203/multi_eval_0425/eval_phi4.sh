set -ex

PROMPT_TYPE="phi4"
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
SUMMARY_PATH=$3
SPLIT="test"
NUM_TEST_SAMPLE=-1

mkdir -p $OUTPUT_DIR
cd ..

DATA_NAME="aime25,aime24,math-500,amc23,minerva_math,olympiadbench"
# DATA_NAME="minerva_math"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --summary_path ${SUMMARY_PATH} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --max_tokens_per_call 16384 \
    --seed 0 \
    --temperature 0.8 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    # --overwrite \

