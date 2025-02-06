import requests
import json

api_key = "fw_3ZYLVpZmp3F84XvWrzcx6Q8J"

url = "https://api.fireworks.ai/inference/v1/chat/completions"

payload = {
  "model": "accounts/fireworks/models/deepseek-r1",
  "max_tokens": 20480,
  "top_p": 1,
  "top_k": 40,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "temperature": 0.6,
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ]
}
headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

print("response", response)


