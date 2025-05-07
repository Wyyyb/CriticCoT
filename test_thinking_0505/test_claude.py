import anthropic


def simple_query(api_key: str):
    client = anthropic.Anthropic(api_key=api_key)
    question = "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$"

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    # Query the Claude model
    delta_text_in_completion = []
    delta_thinking_in_completion = []
    all_content = []
    with client.messages.stream(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            thinking={
                "type": "enabled",
                "budget_tokens": 32000,  # Set a reasonable thinking budget
            },
            max_tokens=36000  # Set a reasonable token limit for the response
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
        all_content_text = "".join(all_content)

    return completion_text, completion_thinking, all_content_text


my_api_key_p1 = "sk-ant-api03-x0YUYEWj6wFXu594YHkOEtCraDlMWbvPQUo4Jf-"
my_api_key_p2 = "WmsiI5M6cMjswjzor4uVZmRGOfBu9OsAWq3DDg6dPc-yxXw-ZYliMAAA"
my_api_key = my_api_key_p1 + my_api_key_p2
result, thinking, all_text = simple_query(my_api_key)
print("Answer:", result)
print("Thinking:", thinking)
print("All Text:", all_text)

