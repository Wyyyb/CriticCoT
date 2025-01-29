import json
import os
import random


def main():
    with open("../local_data/WebInstruct-CFT-50K.json") as fi:
        data = json.load(fi)
    random.shuffle(data)
    data_4k = data[:4000]
    with open("../local_data/WebInstruct-CFT-4K.json") as fo:
        fo.write(json.dumps(data_4k, indent=4))


main()
