# Chunked Solution Generation

This directory contains scripts for generating candidate solutions using vLLM with chunked processing to handle large datasets efficiently.

## Overview

The process is split into six main steps:
1. **Generate Solutions** - Use vLLM to generate solutions for each chunk
2. **Merge Results** - Combine all chunk results into single files
3. **Filter Solutions** - Remove questions where both models answered all correctly or all incorrectly
4. **Generate Critique Data** - Convert solutions to right/wrong critique training data
5. **Format Critique Data** - Format critique data to match original train data structure
6. **Create Critique Data** - Generate critique training data (separate script)

## Files

- `generate_solutions.py` - Main script for generating solutions with vLLM
- `generate_commands.py` - Generate shell commands for running chunks on different GPUs
- `merge_solutions.py` - Merge all chunk results into single files
- `filter_solutions.py` - Filter solutions by model consistency
- `generate_critique_data.py` - Generate critique training data from filtered solutions
- `format_critique_data.py` - Format critique data to match train data structure
- `run_qwen25_parallel.sh` - Shell script to run 8 parallel processes
- `run_full_pipeline.sh` - Complete pipeline script for all steps
- `test_filter.py` - Test script for filtering logic
- `test_critique.py` - Test script for critique data generation
- `test_format.py` - Test script for data formatting
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

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
# Run the complete pipeline with interactive prompts
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

This will:
1. Generate commands and run scripts
2. Check for existing solutions
3. Guide you through solution generation (if needed)
4. Merge, filter, and generate critique data
5. Provide summary statistics

### Option 2: Manual Step-by-Step

#### Step 1: Generate Commands

First, generate the shell commands to see how many chunks you'll have:

```bash
python generate_commands.py --chunk_size 1000
```

This will show you the total number of chunks and generate commands for each GPU.

#### Step 2: Run Solution Generation

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

#### Step 3: Merge Results

After all chunks are processed, merge them into single files:

```bash
python merge_solutions.py
```

This will create:
- `deepscaler_qwen25_solutions.json` - All qwen-2.5-math-7b solutions
- `deepscaler_qwen3_solutions.json` - All qwen3-4b-base solutions

#### Step 4: Filter Solutions

After merging, filter out questions where both models answered all correctly or all incorrectly:

```bash
python filter_solutions.py
```

This will create:
- `deepscaler_train_filter.json` - Filtered training data with solution information

The filtering criteria:
- **Keep**: Questions where at least one model has mixed results (some correct, some incorrect)
- **Remove**: Questions where both models answered all correctly OR both models answered all incorrectly

The filtered data includes:
- All original fields from `deepscaler_train.json`
- `qwen25_solutions` - All 8 solutions from qwen-2.5-math-7b
- `qwen3_solutions` - All 8 solutions from qwen3-4b-base
- Consistency flags for each model

#### Step 5: Generate Critique Data

After filtering, generate critique training data from the filtered solutions:

```bash
python generate_critique_data.py --analyze
```

This will create:
- `deepscaler_critique.json` - Critique training data with right/wrong judgments

The critique data format:
- **Prompt**: "You are a mathematics expert. A student is trying to solve a question. Please explain briefly whether his answer is correct or not. Finally, conclude your judgement with 'Conclusion: right/wrong [END]'"
- **Target**: "right" or "wrong" based on solution correctness
- **Question**: Original math question
- **Solution**: Candidate solution from models
- **Ground Truth**: Correct answer for verification

Features:
- Samples up to 4 solutions per question (configurable)
- Combines solutions from both models
- Provides distribution analysis of right/wrong ratios
- Includes metadata for tracking solution sources

#### Step 6: Format Critique Data

After generating critique data, format it to match the original train data structure:

```bash
python format_critique_data.py --analyze
```

This will create:
- `deepscaler_critique_formatted.json` - Formatted critique data matching train data structure

The formatted data includes:
- **Same structure** as original train data with all required fields
- **Prompt format** with role-based structure
- **Reward model** with ground_truth and style fields
- **Ability field** set to "critique" instead of "math"
- **Extra info** with additional metadata for tracking

Key differences from original train data:
- `answer` and `target` are "right"/"wrong" instead of numerical answers
- `ability` is "critique" instead of "math"
- `data_source` is "deepscaler_critique"
- `extra_info` includes original solution and question ground truth

## Directory Structure

```
cft_data/
├── deepscaler_train.json                  # Input data
├── solutions_qwen25/                      # qwen-2.5-math-7b chunk files
│   ├── qwen-2.5-math-7b_chunk_000.json
│   ├── qwen-2.5-math-7b_chunk_001.json
│   └── ...
├── solutions_qwen3/                       # qwen3-4b-base chunk files
│   ├── qwen3-4b-base_chunk_000.json
│   ├── qwen3-4b-base_chunk_001.json
│   └── ...
├── deepscaler_qwen25_solutions.json       # Merged qwen-2.5-math-7b results
├── deepscaler_qwen3_solutions.json        # Merged qwen3-4b-base results
├── deepscaler_train_filter.json           # Filtered training data
├── deepscaler_critique.json               # Critique training data
└── deepscaler_critique_formatted.json     # Formatted critique data
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

### filter_solutions.py
- `--qwen25_file` - qwen-2.5-math-7b solutions file
- `--qwen3_file` - qwen3-4b-base solutions file  
- `--original_file` - Original training data file
- `--output_file` - Output filtered training data file

### generate_critique_data.py
- `--filtered_file` - Filtered training data file
- `--output_file` - Output critique training data file
- `--max_samples_per_question` - Maximum solutions per question (default: 4)
- `--analyze` - Analyze critique distribution after generation

### format_critique_data.py
- `--critique_file` - Input critique data file
- `--output_file` - Output formatted critique data file
- `--analyze` - Analyze formatted data after generation

## Example for 40K Questions

For 40,000 questions with chunk size 1000:
- Total chunks: 40
- Each chunk: 1000 questions × 8 solutions = 8000 generations
- GPU distribution: 8 GPUs, 5 chunks per GPU

### Quick Start Commands

```bash
# Option 1: Complete pipeline (recommended)
./run_full_pipeline.sh

# Option 2: Manual step-by-step
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

# 5. Filter solutions
python filter_solutions.py

# 6. Generate critique data
python generate_critique_data.py --analyze

# 7. Format critique data
python format_critique_data.py --analyze
```

## Notes

- Each chunk file contains metadata including chunk_id and chunk_index
- Solutions are generated with temperature=0.7 for diversity
- Batch processing is used to manage memory usage
- Results are automatically sorted by original index when merged
- Make sure you have enough disk space for all chunk files 