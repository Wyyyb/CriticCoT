import json
import os
import random


def main():
    with open("/data/yubowang/CriticCoT/local_data/WebInstruct-CFT-50K.json", "r") as fi:
        data = json.load(fi)
    random.shuffle(data)
    data_4k = data[:4000]
    with open("/data/yubowang/CriticCoT/local_data/WebInstruct-CFT-4K.json", "w") as fo:
        fo.write(json.dumps(data_4k, indent=4))


main()
