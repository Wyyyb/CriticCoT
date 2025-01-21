from openai import OpenAI
client = OpenAI()

batch = client.batches.retrieve("file-4Qf5iq6tgTEjS4TjWdz2bo")
print(batch)

