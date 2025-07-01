import tiktoken
import json
import os

def count_tokens_tiktoken(text):
    # 使用GPT模型的编码器
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    return len(tokens)


def main():
    input_dir = "../release_1-shot_data/dsr_data"
    # input_dir = "../bbeh_exp_0528/training_data_0528"
    for file in os.listdir(input_dir):
        if not file.endswith(".jsonl"):
            continue
        file_path = os.path.join(input_dir, file)
        input_tokens_total = 0
        output_tokens_total = 0
        count = 0
        with open(file_path, "r") as f:
            for line in f:
                if not line or line == "":
                    continue
                curr_data = json.loads(line)
                input_str = curr_data["messages"][0]["content"]
                output_str = curr_data["messages"][1]["content"]
                input_tokens = count_tokens_tiktoken(input_str)
                output_tokens = count_tokens_tiktoken(output_str)
                input_tokens_total += input_tokens
                output_tokens_total += output_tokens
                count += 1
        print("file name", file)
        print("count", count)
        print("average input tokens: {}".format(input_tokens_total / count))
        print("average output tokens: {}".format(output_tokens_total / count))


if __name__ == "__main__":
    main()