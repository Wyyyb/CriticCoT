from openai import OpenAI
client = OpenAI()

output_path = "/cpfs/data/user/yubowang/CriticCoT/local_data/batch_data_0122/batchoutput-1.jsonl"
file_response = client.files.content("file-4Qf5iq6tgTEjS4TjWdz2bo")
# file_response = client.files.content("file-YBxjVob2c4juUpdGJeM4y7")
# print(file_response.text)

with open(output_path, "w") as fo:
    fo.write(file_response.text)


output_path = "/cpfs/data/user/yubowang/CriticCoT/local_data/batch_data_0122/batchoutput-2.jsonl"
# file_response = client.files.content("file-4Qf5iq6tgTEjS4TjWdz2bo")
file_response = client.files.content("file-YBxjVob2c4juUpdGJeM4y7")
# print(file_response.text)

with open(output_path, "w") as fo:
    fo.write(file_response.text)