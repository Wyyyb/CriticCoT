#!/usr/bin/env python3
"""
Convert deepscaler.json to training format
"""

import json
import os
from typing import List, Dict, Any

def convert_deepscaler_to_training_format(input_file: str, output_file: str, 
                                         data_source: str = "deepscaler",
                                         subject: str = "Mathematics",
                                         ability: str = "math",
                                         level: int = 5):
    """
    Convert deepscaler.json format to training format
    
    Args:
        input_file: Path to deepscaler.json file
        output_file: Path to output training format file
        data_source: Source identifier for the data
        subject: Subject category
        ability: Ability category
        level: Difficulty level
    """
    
    print(f"Reading deepscaler file: {input_file}")
    
    # Read the deepscaler data
    with open(input_file, 'r', encoding='utf-8') as f:
        deepscaler_data = json.load(f)
    
    print(f"Converting {len(deepscaler_data)} records...")
    
    # Convert to training format
    training_data = []
    
    for i, item in enumerate(deepscaler_data):
        # Extract fields from deepscaler format
        problem = item.get('problem', '')
        answer = item.get('answer', '')
        
        # Create prompt content
        prompt_content = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{problem}
Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""
        
        # Create training format record
        training_record = {
            "answer": answer,
            "gt_answer": answer,  # Ground truth is the same as answer
            "subject": subject,
            "level": level,
            "question": problem,
            "target": answer,
            "data_source": data_source,
            "prompt": [
                {
                    "content": prompt_content,
                    "role": "user"
                }
            ],
            "ability": ability,
            "reward_model": {
                "ground_truth": answer,
                "style": "rule"
            },
            "extra_info": {
                "answer": answer,
                "index": i,
                "level": level,
                "split": "train"
            }
        }
        
        training_data.append(training_record)
    
    # Write to output file
    print(f"Writing {len(training_data)} records to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion completed! Output saved to: {output_file}")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"- Total records: {len(training_data)}")
    print(f"- Data source: {data_source}")
    print(f"- Subject: {subject}")
    print(f"- Ability: {ability}")
    print(f"- Level: {level}")
    
    # Show a sample record
    if training_data:
        print(f"\nSample record:")
        sample = training_data[0]
        print(f"- Question length: {len(sample['question'])} characters")
        print(f"- Answer: {sample['answer']}")
        print(f"- Prompt content length: {len(sample['prompt'][0]['content'])} characters")

def main():
    """Main function to run the conversion"""
    
    # File paths
    input_file = "../cft_data/deepscaler.json"
    output_file = "../cft_data/deepscaler_train.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return
    
    # Run conversion
    print("=== Converting deepscaler.json to training format ===")
    convert_deepscaler_to_training_format(
        input_file=input_file,
        output_file=output_file,
        data_source="deepscaler",
        subject="Mathematics",
        ability="math",
        level=5
    )
    
    print(f"\n=== Conversion completed! ===")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main()
