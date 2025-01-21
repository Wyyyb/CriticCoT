from openai import OpenAI
client = OpenAI()

batch = client.batches.retrieve("file-4qr7yZLDgAZ1yz3Czm2t9f")
print(batch)

