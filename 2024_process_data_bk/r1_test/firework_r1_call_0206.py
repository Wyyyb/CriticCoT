import requests
import json


def single_request(query):
    api_key_seg_1 = "fw_"
    api_key_seg_2 = "3ZYLVpZmp3F84XvWrzcx6Q8J"
    api_key = api_key_seg_1 + api_key_seg_2
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
                "content": query
            }
        ]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            return None, None, None
        content = json.loads(response.text)["choices"][0]["message"]["content"]
        usage = json.loads(response.text)["usage"]
        cost = usage["prompt_tokens"] * 3.0 / 1000000 + usage["completion_tokens"] * 8.0 / 1000000
        return content, cost, usage["completion_tokens"]
    except Exception as e:
        print("exception", e)
        return None, None, None


