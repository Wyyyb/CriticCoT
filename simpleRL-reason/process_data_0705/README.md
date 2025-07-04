# Chunked Solution Generation

This directory contains scripts for generating candidate solutions using vLLM with chunked processing to handle large datasets efficiently.

## Overview

The process is split into three main steps:
1. **Generate Solutions** - Use vLLM to generate solutions for each chunk
2. **Merge Results** - Combine all chunk results into single files
3. **Create Critique Data** - Generate critique training data (separate script)

## Files

- `generate_solutions.py` - Main script for generating solutions with vLLM
- `generate_commands.py` - Generate shell commands for running chunks on different GPUs
- `merge_solutions.py` - Merge all chunk results into single files
- `requirements.txt` - Python dependencies

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the models available:
   - `qwen-2.5-math-7b`
   - `qwen3-4b-base`

3. Make sure `deepscaler_train.json` exists in `../cft_data/`

## Usage

### Step 1: Generate Commands

First, generate the shell commands to see how many chunks you'll have:

```bash
python generate_commands.py --chunk_size 1000
```

This will show you the total number of chunks and generate commands for each GPU.

### Step 2: Run Solution Generation

#### Option A: Run all chunks sequentially
```bash
# For qwen-2.5-math-7b
python generate_solutions.py --model qwen-2.5-math-7b --chunk_size 1000

# For qwen3-4b-base  
python generate_solutions.py --model qwen3-4b-base --chunk_size 1000
```

#### Option B: Run specific chunk (for distributed processing)
```bash
# Process chunk 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 python generate_solutions.py \
  --model qwen-2.5-math-7b \
  --chunk_id 0 \
  --chunk_size 1000 \
  --output_dir ../cft_data/solutions_qwen25

# Process chunk 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 python generate_solutions.py \
  --model qwen-2.5-math-7b \
  --chunk_id 1 \
  --chunk_size 1000 \
  --output_dir ../cft_data/solutions_qwen25
```

#### Option C: Run multiple chunks in parallel
Create shell scripts using the output from `generate_commands.py`:

```bash
# Generate commands
python generate_commands.py > commands.txt

# Create run script for qwen-2.5-math-7b
cat commands.txt | grep -A 8 "qwen-2.5-math-7b" > run_qwen25.sh
chmod +x run_qwen25.sh

# Run in parallel
./run_qwen25.sh
```

### Step 3: Merge Results

After all chunks are processed, merge them into single files:

```bash
python merge_solutions.py
```

This will create:
- `deepscaler_qwen25_solutions.json` - All qwen-2.5-math-7b solutions
- `deepscaler_qwen3_solutions.json` - All qwen3-4b-base solutions

## Directory Structure

```
cft_data/
├── deepscaler_train.json          # Input data
├── solutions_qwen25/              # qwen-2.5-math-7b chunk files
│   ├── qwen-2.5-math-7b_chunk_000.json
│   ├── qwen-2.5-math-7b_chunk_001.json
│   └── ...
├── solutions_qwen3/               # qwen3-4b-base chunk files
│   ├── qwen3-4b-base_chunk_000.json
│   ├── qwen3-4b-base_chunk_001.json
│   └── ...
├── deepscaler_qwen25_solutions.json  # Merged qwen-2.5-math-7b results
└── deepscaler_qwen3_solutions.json   # Merged qwen3-4b-base results
```

## Parameters

### generate_solutions.py
- `--input_file` - Input deepscaler training data file
- `--output_dir` - Output directory for chunk files
- `--model` - Model to use (qwen-2.5-math-7b or qwen3-4b-base)
- `--chunk_size` - Number of questions per chunk (default: 1000)
- `--chunk_id` - Specific chunk ID to process (if None, process all)
- `--num_samples` - Number of solutions per question (default: 8)

### generate_commands.py
- `--input_file` - Input deepscaler training data file
- `--chunk_size` - Number of questions per chunk (default: 1000)
- `--num_samples` - Number of solutions per question (default: 8)

### merge_solutions.py
- `--qwen25_dir` - Directory containing qwen-2.5-math-7b chunks
- `--qwen3_dir` - Directory containing qwen3-4b-base chunks
- `--output_dir` - Output directory for merged files

## Example for 40K Questions

For 40,000 questions with chunk size 1000:
- Total chunks: 40
- Each chunk: 1000 questions × 8 solutions = 8000 generations
- GPU distribution: 8 GPUs, 5 chunks per GPU

### Quick Start Commands

```bash
# 1. Generate commands
python generate_commands.py --chunk_size 1000 > commands.txt

# 2. Create run scripts
grep -A 8 "qwen-2.5-math-7b" commands.txt > run_qwen25.sh
grep -A 8 "qwen3-4b-base" commands.txt > run_qwen3.sh
chmod +x run_qwen25.sh run_qwen3.sh

# 3. Run in parallel (on different machines or terminals)
./run_qwen25.sh  # On machine with GPUs 0-7
./run_qwen3.sh   # On another machine with GPUs 0-7

# 4. Merge results
python merge_solutions.py
```

## Notes

- Each chunk file contains metadata including chunk_id and chunk_index
- Solutions are generated with temperature=0.7 for diversity
- Batch processing is used to manage memory usage
- Results are automatically sorted by original index when merged
- Make sure you have enough disk space for all chunk files 