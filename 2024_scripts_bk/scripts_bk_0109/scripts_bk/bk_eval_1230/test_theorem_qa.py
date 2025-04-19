from datasets import load_dataset

dataset = load_dataset("TIGER-Lab/TheoremQA")

for d in dataset['test']:
    print(d)


