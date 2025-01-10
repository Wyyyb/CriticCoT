import json
from datasets import load_dataset


def load_numina():
    ds = load_dataset("AI-MO/NuminaMath-CoT")
    output_file = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/NuminaMath-CoT/ori_numina_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in ds["train"]:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')


if __name__ == "__main__":
    load_numina()
