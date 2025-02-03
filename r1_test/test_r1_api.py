import requests

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "deepseek-ai/DeepSeek-V3",
    "messages": [
        {
            "role": "user",
            "content": "At 2:30 p.m. during a long drive, Bobbi asks her parents, ``Are we there yet?'' Her mother responds, ``We will be there in 7200 seconds.'' If Bobbi's mother is correct, at what time in the afternoon will they arrive at their destination?"
        }
    ],
    "stream": False,
    "max_tokens": 1024,
    "stop": ["null"],
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "response_format": {"type": "text"},
}
headers = {
    "Authorization": "Bearer sk-htkongektpfxqolvifmbozbvfsjpjdosfhzwuseeuxiibvpc",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)

