#!/bin/bash

# Run 8 parallel processes for qwen-2.5-math-7b
# Each GPU processes one chunk

# Model path
MODEL_PATH="/data/minimax-dialogue/feishan/models/Qwen2.5-Math-7B"

# Input and output paths
INPUT_FILE="../cft_data/deepscaler_train.json"
OUTPUT_DIR="../cft_data/solutions_qwen25"
CHUNK_SIZE=1000
NUM_SAMPLES=8

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting 8 parallel processes for qwen-2.5-math-7b"
echo "Model path: $MODEL_PATH"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Chunk size: $CHUNK_SIZE"
echo ""

# Start 8 parallel processes
# GPU 0 - Chunk 0
CUDA_VISIBLE_DEVICES=0 python generate_solutions.py \
  --input_file $INPUT_FILE \
  --model $MODEL_PATH \
  --chunk_id 0 \
  --chunk_size $CHUNK_SIZE \
  --num_samples $NUM_SAMPLES \
  --output_dir $OUTPUT_DIR &

# GPU 1 - Chunk 1
CUDA_VISIBLE_DEVICES=1 python generate_solutions.py \
  --input_file $INPUT_FILE \
  --model $MODEL_PATH \
  --chunk_id 1 \
  --chunk_size $CHUNK_SIZE \
  --num_samples $NUM_SAMPLES \
  --output_dir $OUTPUT_DIR &

# GPU 2 - Chunk 2
CUDA_VISIBLE_DEVICES=2 python generate_solutions.py \
  --input_file $INPUT_FILE \
  --model $MODEL_PATH \
  --chunk_id 2 \
  --chunk_size $CHUNK_SIZE \
  --num_samples $NUM_SAMPLES \
  --output_dir $OUTPUT_DIR &

# GPU 3 - Chunk 3
CUDA_VISIBLE_DEVICES=3 python generate_solutions.py \
  --input_file $INPUT_FILE \
  --model $MODEL_PATH \
  --chunk_id 3 \
  --chunk_size $CHUNK_SIZE \
  --num_samples $NUM_SAMPLES \
  --output_dir $OUTPUT_DIR &

# GPU 4 - Chunk 4
CUDA_VISIBLE_DEVICES=4 python generate_solutions.py \
  --input_file $INPUT_FILE \
  --model $MODEL_PATH \
  --chunk_id 4 \
  --chunk_size $CHUNK_SIZE \
  --num_samples $NUM_SAMPLES \
  --output_dir $OUTPUT_DIR &

# GPU 5 - Chunk 5
CUDA_VISIBLE_DEVICES=5 python generate_solutions.py \
  --input_file $INPUT_FILE \
  --model $MODEL_PATH \
  --chunk_id 5 \
  --chunk_size $CHUNK_SIZE \
  --num_samples $NUM_SAMPLES \
  --output_dir $OUTPUT_DIR &

# GPU 6 - Chunk 6
CUDA_VISIBLE_DEVICES=6 python generate_solutions.py \
  --input_file $INPUT_FILE \
  --model $MODEL_PATH \
  --chunk_id 6 \
  --chunk_size $CHUNK_SIZE \
  --num_samples $NUM_SAMPLES \
  --output_dir $OUTPUT_DIR &

# GPU 7 - Chunk 7
CUDA_VISIBLE_DEVICES=7 python generate_solutions.py \
  --input_file $INPUT_FILE \
  --model $MODEL_PATH \
  --chunk_id 7 \
  --chunk_size $CHUNK_SIZE \
  --num_samples $NUM_SAMPLES \
  --output_dir $OUTPUT_DIR &

echo "All 8 processes started. Waiting for completion..."
wait

echo "All processes completed!"
echo "Results saved in: $OUTPUT_DIR" 