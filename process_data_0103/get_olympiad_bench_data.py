from datasets import load_dataset
import json

ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP")

output_file = "../math_eval/dataset/OlympiadBench/OlympiadBench.jsonl"

with open(output_file, 'w', encoding='utf-8') as f:
    for item in ds["train"]:
        json_str = json.dumps(item, ensure_ascii=False)
        f.write(json_str + '\n')

