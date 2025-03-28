from vllm import LLM, SamplingParams
from typing import List
import os
import json
import argparse


def load_vllm_model(model_path: str):
    try:
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        # 初始化模型
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))  # 根据GPU数量调整
        )

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            n=1,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
                if "qwen2" in model_path.lower()
                else None
            ),
            top_p=1.0
        )
        return llm, sampling_params
    except Exception as e:
        print("load vllm model failed", e)
        return None, None


def batch_predict(llm, sampling_params, prompts: List[str]) -> List[str]:
    if not llm or not sampling_params:
        print("llm, sampling_params are None")
        return []
    try:
        print("Processing", len(prompts))
        outputs = llm.generate(prompts, sampling_params)
        # 提取生成的文本
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        return results
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return []


def single_file_critique(file_path, model_path):
    model_name = model_path.split("/")[-1].replace(".", "_")
    output_path = file_path.replace(".jsonl", f"_{model_name}.jsonl")
    llm, sampling_params = load_vllm_model(model_path)
    input_data = []
    with open(file_path, 'r', encoding='utf-8') as fi:
        for line in fi.readlines():
            curr = json.loads(line)
            input_data.append(curr)
    prompts = []
    for each in input_data:
        prompt = each["prompt"]
        prompts.append(prompt)
    outputs = batch_predict(llm, sampling_params, prompts)
    if len(outputs) != len(prompts):
        print("inconsistent length of outputs and prompts", len(outputs), len(prompts))
    output_data = []
    for i, each in enumerate(outputs):
        curr = {"prompt": prompts[i], "response": each}
        output_data.append(curr)
    with open(output_path, "w") as fo:
        for each in output_data:
            fo.write(json.dumps(each) + "\n")


def main(model_path, input_path):
    single_file_critique(input_path, model_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process math training files')
    parser.add_argument('--model_path', type=str, required=True,
                        default="/data/yubo/CriticCoT/Qwen2.5-Math-7B-CFT",
                        help='Path to the model')
    parser.add_argument('--input_dir', type=str, required=True,
                        default="/data/yubo/google-research/instruction_following_eval/data/input_data.jsonl",
                        help='Input directory containing math_train folder')

    args = parser.parse_args()
    main(args.model_path, args.input_dir)

