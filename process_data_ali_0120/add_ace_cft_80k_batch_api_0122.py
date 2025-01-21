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
FileObject(id='file-4Qf5iq6tgTEjS4TjWdz2bo', bytes=95089356, created_at=1737483899, filename='batchinput-1.jsonl', object='file', purpose='batch', status='processed', status_details=None)
batch_input_file_id file-4Qf5iq6tgTEjS4TjWdz2bo

FileObject(id='file-YBxjVob2c4juUpdGJeM4y7', bytes=95529176, created_at=1737484060, filename='batchinput-2.jsonl', object='file', purpose='batch', status='processed', status_details=None)
batch_input_file_id file-YBxjVob2c4juUpdGJeM4y7
'''

