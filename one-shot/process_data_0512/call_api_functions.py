from google import genai
from google.genai import types
import anthropic
from openai import OpenAI


def get_client(model_type, model_name, api_key):
    if model_type == "anthropic-think" or model_type == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
    elif model_type == "gemini":
        client = genai.Client(api_key=api_key)
    elif model_type == "openai":
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
            # max_tokens=16000,
            max_completion_tokens=16000,
            top_p=0.95
        )
        # 保存结果
        return completion.choices[0].message.content

    return None


def get_prompt(question, solution):
    question = "Question:\n" + question
    solution = "Student's Solution:\n" + solution
    prompt = f"You are a mathematics expert. Analyze if the student's solution to the given question is correct. " \
             f"Follow these steps:\n" \
             f"1. Identify the key mathematical concepts and correct approach.\n" \
             f"2. Check each step of the student's solution.\n" \
             f"3. If incorrect, point out errors and provide the correct solution, " \
             f"putting your final answer within \\boxed{{}}.\n" \
             f"Conclude with \"Conclusion: right/wrong [END]\"\n\n{question}\n\n{solution}"
    return prompt


def get_messages(model_type, question, solution):
    if model_type == "anthropic-think" or model_type == "anthropic" or model_type == "grok":
        messages = [
            {
                "role": "user",
                "content": get_prompt(question, solution)
            }
        ]
        return messages
    elif model_type == "gemini":
        return get_prompt(question, solution)
    elif model_type == "openai":
        messages = [{"role": "user",
                     "content": [{"type": "text","text": get_prompt(question, solution)}]}]
        return messages
    return None






