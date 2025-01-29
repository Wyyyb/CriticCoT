import json
import os
import random


def main():
    input_path = "/map-vepfs/yubo/CriticCoT/LLaMA-Factory/data/webinstruct_cft_80k_0121_p3.json"
    with open(input_path, "r") as fi:
        data = json.load(fi)
    data = sorted(data, key=lambda x: -len(x["output"]))
    data = data[:50000]
    random.shuffle(data)
    output_path = "/map-vepfs/yubo/CriticCoT/LLaMA-Factory/data/webinstruct_cft_50k_0129.json"
    with open(output_path, "w") as fo:
        fo.write(json.dumps(data, indent=4))


main()
