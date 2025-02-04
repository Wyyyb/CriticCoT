import json
import os
import random
from datasets import load_dataset


def main():
    educational_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct")
    evol_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "evol_instruct")
    mceval_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "mceval_instruct")
    package_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "package_instruct")

    data = []
    data += format_data(educational_instruct, "educational_instruct")
    data += format_data(evol_instruct, "evol_instruct")
    data += format_data(mceval_instruct, "mceval_instruct")
    data += format_data(package_instruct, "package_instruct")

    random.shuffle(data)
    full_data = data
    sample_200k = data[:200000]
    sample_80k = data[:80000]
    with open(f"../local_data/opc-sft-stage2/opc-sft-stage2_{full_data}.json", "w") as fo:
        fo.write(json.dumps(full_data, indent=4))
    with open(f"../local_data/opc-sft-stage2/opc-sft-stage2_{sample_200k}.json", "w") as fo:
        fo.write(json.dumps(sample_200k, indent=4))
    with open(f"../local_data/opc-sft-stage2/opc-sft-stage2_{sample_80k}.json", "w") as fo:
        fo.write(json.dumps(sample_80k, indent=4))


def format_data(data, data_source):
    res = []
    data = data.tolist()
    for i, each in enumerate(data):
        question = each["instruction"]
        answer = each["output"]
        res.append({"idx": i, "question": question, "answer": answer, "tag": data_source})
    return res


main()

