import os
import json
import random
from datasets import load_dataset, Dataset, concatenate_datasets
from utils import load_jsonl, lower_keys
from PIL import Image


def clean_text(text):
    if isinstance(text, str):
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    return text


def clean_example(example):
    cleaned = {}
    for key, value in example.items():
        if isinstance(value, dict):
            cleaned[key] = clean_example(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_text(v) if isinstance(v, str)
                          else clean_example(v) if isinstance(v, dict)
                          else None if isinstance(v, Image.Image)  # 直接忽略图像
                          else v
                          for v in value]
        elif isinstance(value, Image.Image):
            cleaned[key] = None  # 直接忽略图像
        else:
            cleaned[key] = clean_text(value) if isinstance(value, str) else value
    return cleaned


def load_data(data_name, split, data_dir='./data'):
    data_name = data_name.replace("_ins", "")
    if data_name in ['minerva_math']:
        data_name = 'math_oai'
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset("competition_math", split=split, name="main", cache_dir=f"{data_dir}/temp")
        elif data_name == "theorem_qa":
            dataset = load_dataset("TIGER-Lab/TheoremQA", split=split)
            print(dataset)
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "gsm_hard":
            dataset = load_dataset("reasoning-machines/gsm_hard", split="train")
        elif data_name == "svamp":
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets([dataset, load_dataset("ChilleD/SVAMP", split="test")])
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(lambda x: ";" not in x['answer'])
        elif data_name == "mawps":
            examples = []
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example['type'] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "finqa":
            dataset = load_dataset("dreamerdeo/finqa", split=split, name="main")
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "tabmwp":
            examples = []
            with open(f"{data_dir}/tabmwp/tabmwp_{split}.json", "r") as f:
                data_dict = json.load(f)
                examples.extend(data_dict.values())
            dataset = Dataset.from_list(examples)
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "mathqa":
            dataset = load_dataset("math_qa", split=split)
            dataset = dataset.rename_column("category", "type")
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "mmlu_stem":
            dataset = load_dataset("hails/mmlu_no_train", 'all', split='test')
            stem_subjects = ['abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
                             'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security',
                             'conceptual_physics', 'electrical_engineering', 'elementary_mathematics',
                             'high_school_biology',
                             'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics',
                             'high_school_physics', 'high_school_statistics', 'machine_learning']
            dataset = dataset.rename_column("subject", "type")
            dataset = dataset.filter(lambda x: x['type'] in stem_subjects)
        elif data_name == "bbh":
            examples = []
            for data_name in ["reasoning_about_colored_objects", "penguins_in_a_table", \
                              "date_understanding", "repeat_copy_logic", "object_counting"]:
                with open(f"{data_dir}/bbh/bbh/{data_name}.json", "r") as f:
                    sub_examples = json.load(f)["examples"]
                    for example in sub_examples:
                        example['type'] = data_name
                    examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "hungarian_exam":
            dataset = load_dataset("json", data_files=f"{data_dir}/hungarian_exam/{split}.jsonl")
        else:
            raise NotImplementedError(data_name)

        # 清理数据并保存
        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        examples = [clean_example(example) for example in examples]

        # 手动保存为 jsonl
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        with open(data_file, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=True)
                f.write('\n')

    # add 'idx' in the first column
    if 'idx' not in examples[0]:
        examples = [{'idx': i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x['idx'])
    return examples


if __name__ == "__main__":
    examples = load_data("theorem_qa", "test")

