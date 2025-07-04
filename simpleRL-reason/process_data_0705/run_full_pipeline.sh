#!/bin/bash

# Full pipeline script for processing deepscaler data
# This script runs all steps from solution generation to critique data creation

set -e  # Exit on any error

echo "=========================================="
echo "Deepscaler Data Processing Pipeline"
echo "=========================================="

# Configuration
INPUT_FILE="../cft_data/deepscaler_train.json"
CHUNK_SIZE=1000
NUM_SAMPLES=8
MAX_CRITIQUE_SAMPLES=4

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    print_error "Input file $INPUT_FILE does not exist!"
    exit 1
fi

print_status "Starting pipeline with input file: $INPUT_FILE"

# Step 1: Generate commands
print_status "Step 1: Generating commands..."
python generate_commands.py \
    --input_file "$INPUT_FILE" \
    --chunk_size $CHUNK_SIZE \
    --num_samples $NUM_SAMPLES > commands.txt

print_success "Commands generated and saved to commands.txt"

# Step 2: Create run scripts
print_status "Step 2: Creating run scripts..."
grep -A 8 "qwen-2.5-math-7b" commands.txt > run_qwen25.sh
grep -A 8 "qwen3-4b-base" commands.txt > run_qwen3.sh
chmod +x run_qwen25.sh run_qwen3.sh

print_success "Run scripts created: run_qwen25.sh, run_qwen3.sh"

# Step 3: Check if solutions already exist
print_status "Step 3: Checking existing solutions..."

if [ -f "../cft_data/deepscaler_qwen25_solutions.json" ] && [ -f "../cft_data/deepscaler_qwen3_solutions.json" ]; then
    print_warning "Solution files already exist. Skipping solution generation."
    print_warning "If you want to regenerate, delete the solution files first."
    SKIP_SOLUTIONS=true
else
    print_status "Solution files not found. Will generate solutions."
    SKIP_SOLUTIONS=false
fi

# Step 4: Generate solutions (if needed)
if [ "$SKIP_SOLUTIONS" = false ]; then
    print_status "Step 4: Generating solutions..."
    print_warning "This step requires 8 GPUs and will take a long time."
    print_warning "Make sure you have the models available and enough GPU memory."
    
    read -p "Do you want to continue with solution generation? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting solution generation..."
        print_status "You can run the scripts in parallel on different machines:"
        print_status "  ./run_qwen25.sh  # On machine with GPUs 0-7"
        print_status "  ./run_qwen3.sh   # On another machine with GPUs 0-7"
        
        # For now, just show the commands
        print_status "To run solutions generation, execute:"
        echo "  ./run_qwen25.sh"
        echo "  ./run_qwen3.sh"
    else
        print_warning "Solution generation skipped. Please run the scripts manually."
        print_warning "After solution generation, continue with this script."
    fi
else
    print_success "Solutions already exist, skipping generation."
fi

# Step 5: Merge solutions
print_status "Step 5: Merging solutions..."
if [ -f "../cft_data/deepscaler_qwen25_solutions.json" ] && [ -f "../cft_data/deepscaler_qwen3_solutions.json" ]; then
    python merge_solutions.py
    print_success "Solutions merged successfully."
else
    print_error "Solution files not found. Please run solution generation first."
    exit 1
fi

# Step 6: Filter solutions
print_status "Step 6: Filtering solutions..."
python filter_solutions.py
print_success "Solutions filtered successfully."

# Step 7: Generate critique data
print_status "Step 7: Generating critique data..."
python generate_critique_data.py \
    --max_samples_per_question $MAX_CRITIQUE_SAMPLES \
    --analyze
print_success "Critique data generated successfully."

# Step 8: Format critique data
print_status "Step 8: Formatting critique data..."
python format_critique_data.py --analyze
print_success "Critique data formatted successfully."

# Step 9: Summary
print_status "Step 9: Pipeline summary..."
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="

# Count files and show statistics
if [ -f "../cft_data/deepscaler_train_filter.json" ]; then
    FILTERED_COUNT=$(python -c "import json; print(len(json.load(open('../cft_data/deepscaler_train_filter.json'))))")
    print_success "Filtered data: $FILTERED_COUNT questions"
fi

if [ -f "../cft_data/deepscaler_critique.json" ]; then
    CRITIQUE_COUNT=$(python -c "import json; print(len(json.load(open('../cft_data/deepscaler_critique.json'))))")
    print_success "Critique data: $CRITIQUE_COUNT critiques"
fi

if [ -f "../cft_data/deepscaler_critique_formatted.json" ]; then
    FORMATTED_COUNT=$(python -c "import json; print(len(json.load(open('../cft_data/deepscaler_critique_formatted.json'))))")
    print_success "Formatted critique data: $FORMATTED_COUNT items"
fi

echo ""
print_success "Output files created:"
echo "  - ../cft_data/deepscaler_qwen25_solutions.json"
echo "  - ../cft_data/deepscaler_qwen3_solutions.json"
echo "  - ../cft_data/deepscaler_train_filter.json"
echo "  - ../cft_data/deepscaler_critique.json"
echo "  - ../cft_data/deepscaler_critique_formatted.json"

echo ""
print_status "Next steps:"
echo "  1. Review the generated critique data"
echo "  2. Use the formatted critique data for training your critique model"
echo "  3. Optionally, run test scripts to verify the logic:"
echo "     - python test_filter.py"
echo "     - python test_critique.py"
echo "     - python test_format.py"

print_success "Pipeline completed!" 