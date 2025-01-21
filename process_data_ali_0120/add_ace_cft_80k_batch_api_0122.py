from openai import OpenAI
client = OpenAI()

batch_input_file = client.files.create(
    file=open("/cpfs/data/user/yubowang/CriticCoT/local_data/batch_data_0122/batchinput-2.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)
batch_input_file_id = batch_input_file.id
print("batch_input_file_id", batch_input_file_id)
client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "ace cft job"
    }
)

'''
FileObject(id='file-4qr7yZLDgAZ1yz3Czm2t9f', bytes=217996954, created_at=1737482842, filename='batchinput.jsonl', object='file', purpose='batch', status='processed', status_details=None)
'''

