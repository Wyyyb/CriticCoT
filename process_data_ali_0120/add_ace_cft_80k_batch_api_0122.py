from openai import OpenAI
client = OpenAI()

batch_input_file = client.files.create(
    file=open("/cpfs/data/user/yubowang/CriticCoT/local_data/batch_data_0122/batchinput.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)

batch_input_file_id = batch_input_file.id
client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "ace cft job"
    }
)

