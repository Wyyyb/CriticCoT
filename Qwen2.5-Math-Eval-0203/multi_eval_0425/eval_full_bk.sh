set -ex

PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
SUMMARY_PATH=$3
SPLIT="test"
NUM_TEST_SAMPLE=-1

mkdir -p $OUTPUT_DIR
cd ..
# English open datasets
# DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"
DATA_NAME="aime25,aime24,amc23,minerva_math,olympiadbench,math,gsm8k,math-500"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --summary_path ${SUMMARY_PATH} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    # --overwrite \
#
## English multiple-choice datasets
#DATA_NAME="aqua,sat_math,mmlu_stem"
#TOKENIZERS_PARALLELISM=false \
#python3 -u math_eval.py \
#    --model_name_or_path ${MODEL_NAME_OR_PATH} \
#    --data_name ${DATA_NAME} \
#    --output_dir ${OUTPUT_DIR} \
#    --split ${SPLIT} \
#    --prompt_type ${PROMPT_TYPE} \
#    --num_test_sample ${NUM_TEST_SAMPLE} \
#    --seed 0 \
#    --temperature 0 \
#    --n_sampling 1 \
#    --top_p 1 \
#    --start 0 \
#    --end -1 \
#    --use_vllm \
#    --save_outputs \
#    --overwrite \
#    --num_shots 5
#
## Chinese gaokao collections
#DATA_NAME="gaokao2024_I,gaokao2024_II,gaokao2024_mix,gaokao_math_cloze,gaokao_math_qa"
#TOKENIZERS_PARALLELISM=false \
#python3 -u math_eval.py \
#    --model_name_or_path ${MODEL_NAME_OR_PATH} \
#    --data_name ${DATA_NAME} \
#    --output_dir ${OUTPUT_DIR} \
#    --split ${SPLIT} \
#    --prompt_type ${PROMPT_TYPE} \
#    --num_test_sample ${NUM_TEST_SAMPLE} \
#    --seed 0 \
#    --temperature 0 \
#    --n_sampling 1 \
#    --top_p 1 \
#    --start 0 \
#    --end -1 \
#    --use_vllm \
#    --save_outputs \
#    --overwrite \
#    --adapt_few_shot
#
## Chinese other datasets
#DATA_NAME="cmath,cn_middle_school"
#TOKENIZERS_PARALLELISM=false \
#python3 -u math_eval.py \
#    --model_name_or_path ${MODEL_NAME_OR_PATH} \
#    --data_name ${DATA_NAME} \
#    --output_dir ${OUTPUT_DIR} \
#    --split ${SPLIT} \
#    --prompt_type ${PROMPT_TYPE} \
#    --num_test_sample ${NUM_TEST_SAMPLE} \
#    --seed 0 \
#    --temperature 0 \
#    --n_sampling 1 \
#    --top_p 1 \
#    --start 0 \
#    --end -1 \
#    --use_vllm \
#    --save_outputs \
#    --overwrite \
#    --adapt_few_shot
#
#
## English competition datasets
#DATA_NAME="aime24,amc23"
#TOKENIZERS_PARALLELISM=false \
#python3 -u math_eval.py \
#    --model_name_or_path ${MODEL_NAME_OR_PATH} \
#    --data_name ${DATA_NAME} \
#    --output_dir ${OUTPUT_DIR} \
#    --split ${SPLIT} \
#    --prompt_type ${PROMPT_TYPE} \
#    --num_test_sample ${NUM_TEST_SAMPLE} \
#    --seed 0 \
#    --temperature 0 \
#    --n_sampling 1 \
#    --top_p 1 \
#    --start 0 \
#    --end -1 \
#    --use_vllm \
#    --save_outputs \
#    --overwrite \
