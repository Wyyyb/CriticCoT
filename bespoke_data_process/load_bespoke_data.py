from datasets import load_dataset
import json

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("bespokelabs/Bespoke-Stratos-17k")
data_list = ds['train'].to_list()

with open("../local_data/bespoke_data/ori_bespoke_data.json", "w") as fo:
    fo.write(json.dumps(data_list))



