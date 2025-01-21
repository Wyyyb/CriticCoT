from openai import OpenAI
client = OpenAI()

# batch = client.batches.retrieve("batch_4Qf5iq6tgTEjS4TjWdz2bo")
# print(batch)
batch = client.batches.retrieve("batch_YBxjVob2c4juUpdGJeM4y7")
print(batch)

