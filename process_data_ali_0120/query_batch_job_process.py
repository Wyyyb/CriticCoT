from openai import OpenAI
client = OpenAI()

# batch = client.batches.retrieve("batch_4Qf5iq6tgTEjS4TjWdz2bo")
# print(batch)
# batch = client.batches.retrieve("batch_YBxjVob2c4juUpdGJeM4y7")
# print(batch)
batch = client.batches.retrieve("batch_4qr7yZLDgAZ1yz3Czm2t9f")
print(batch)

