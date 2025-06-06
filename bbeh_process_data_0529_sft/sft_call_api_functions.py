from google import genai
from google.genai import types
import anthropic
from openai import OpenAI


def get_client(model_type, model_name, api_key):
    if model_type == "anthropic-think" or model_type == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
    elif model_type == "gemini":
        client = genai.Client(api_key=api_key)
    elif model_type == "openai" or model_type == "openai-reasoning":
        client = OpenAI()
    elif model_type == "grok":
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
    else:
        print("unsupported model:", model_type, model_name)
        return None
    return client


def send_request(client, model_type, model_name, message):
    if client is None:
        return None
    if model_type == "gemini":
        response = client.models.generate_content(
            model=model_name,
            contents=message,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=16000)
            ),
        )
        return response.text
    elif model_type == "anthropic-think":
        delta_text_in_completion = []
        delta_thinking_in_completion = []
        all_content = []
        with client.messages.stream(
                model=model_name,
                messages=message,
                thinking={"type": "enabled", "budget_tokens": 8000},
                max_tokens=16000
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    pass
                    # print(f"\nStarting {event.content_block.type} block...")
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        # print(f"Thinking: {event.delta.thinking}", end="", flush=True)
                        delta_thinking_in_completion.append(event.delta.thinking)
                        all_content.append(event.delta.thinking)
                    elif event.delta.type == "text_delta":
                        # print(f"Response: {event.delta.text}", end="", flush=True)
                        delta_text_in_completion.append(event.delta.text)
                        all_content.append(event.delta.text)
                elif event.type == "content_block_stop":
                    pass
                    # print("\nBlock complete.")
            completion_thinking = "".join(delta_thinking_in_completion)
            completion_text = "".join(delta_text_in_completion)
        return "<think>\n" + completion_thinking + "\n</think>\n" + completion_text
    elif model_type == "anthropic":
        delta_text_in_completion = []
        delta_thinking_in_completion = []
        all_content = []
        with client.messages.stream(
                model=model_name,
                messages=message,
                max_tokens=8000
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    pass
                    # print(f"\nStarting {event.content_block.type} block...")
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        # print(f"Thinking: {event.delta.thinking}", end="", flush=True)
                        delta_thinking_in_completion.append(event.delta.thinking)
                        all_content.append(event.delta.thinking)
                    elif event.delta.type == "text_delta":
                        # print(f"Response: {event.delta.text}", end="", flush=True)
                        delta_text_in_completion.append(event.delta.text)
                        all_content.append(event.delta.text)
                elif event.type == "content_block_stop":
                    pass
                    # print("\nBlock complete.")
            completion_thinking = "".join(delta_thinking_in_completion)
            completion_text = "".join(delta_text_in_completion)
        return completion_text
    elif model_type == "openai" or model_type == "grok":
        completion = client.chat.completions.create(
            model=model_name,
            messages=message,
            temperature=0.2,
            max_tokens=16000,
            top_p=0.95
        )
        # 保存结果
        return completion.choices[0].message.content
    elif model_type == "openai-reasoning":
        completion = client.chat.completions.create(
            model=model_name,
            messages=message,
            max_completion_tokens=16000,
        )
        # 保存结果
        return completion.choices[0].message.content

    return None


def get_sft_prompt(question, solution):
    question = "Question:\n" + question
    solution = "Student's Solution:\n" + solution
    prompt = f"Solve the following problems step by step with reasoning, putting your final answer within \\boxed{{}}.\n\n{question}"
    return prompt


def get_messages(model_type, question, solution):
    if model_type == "anthropic-think" or model_type == "anthropic" or model_type == "grok":
        messages = [
            {
                "role": "user",
                "content": get_sft_prompt(question, solution)
            }
        ]
        return messages
    elif model_type == "gemini":
        return get_sft_prompt(question, solution)
    elif model_type == "openai" or model_type == "openai-reasoning":
        messages = [{"role": "user",
                     "content": [{"type": "text","text": get_sft_prompt(question, solution)}]}]
        return messages
    return None






