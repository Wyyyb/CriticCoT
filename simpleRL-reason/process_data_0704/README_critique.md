# Critique Data Generation

This script generates critique training data using vLLM with two different models.

## Prerequisites

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the following models available:
   - `qwen-2.5-math-7b`
   - `qwen3-4b-base`

## Usage

1. First, ensure you have the `deepscaler_train.json` file in the `../cft_data/` directory.

2. Run the script:
```bash
python generate_critique_data.py
```

## What the script does

1. **Loads data**: Reads `deepscaler_train.json` containing math problems
2. **Generates answers**: Uses vLLM to generate 8 answers per question for each model
3. **Filters data**: Removes questions where both models get all answers correct or all wrong
4. **Extracts answers**: Extracts boxed answers from model responses
5. **Generates conclusions**: Creates "right"/"wrong" conclusions based on model performance
6. **Saves results**: Creates three output files

## Output files

- `deepscaler_filtered_model1.json`: Filtered data with qwen-2.5-math-7b solutions
- `deepscaler_filtered_model2.json`: Filtered data with qwen3-4b-base solutions  
- `deepscaler_critique_data.json`: Critique training data with conclusions

## Critique data format

Each record in `deepscaler_critique_data.json` contains:
- `prompt`: Original question prompt
- `model1_name`: Name of first model
- `model2_name`: Name of second model
- `model1_solutions`: 8 solutions from first model
- `model2_solutions`: 8 solutions from second model
- `ground_truth`: Correct answer
- `conclusion`: "right" or "wrong" based on which model performed better
- `original_index`: Index from original dataset
- `subject`, `level`, `ability`: Metadata

## Notes

- The script processes data in batches to manage memory usage
- It includes robust answer extraction from `\boxed{}` format
- Answer comparison handles LaTeX formatting and numerical evaluation
- Filtering ensures only questions with mixed model performance are kept 