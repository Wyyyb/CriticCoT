from vllm import LLM, SamplingParams
from typing import List
import os
import json
import argparse
import time


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
            max_tokens=16384,
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


def get_prompt(question):
    question = "Question:\n" + question
    prompt = f"<|im_start|>Please reason step by step to find a solution to the following question, " \
             f"and put your final answer within \\boxed{{}}.<|im_end|>\n\n" \
             f"<|im_start|>user\n{question}<|im_end|>\n" \
             f"<|im_start|>assistant\n"
    return prompt


def main():
    input_file = "../local_data/deepmath_cft_data/deepmath_cft_step_1.json"
    output_file = "../local_data/deepmath_cft_data/deepmath_qwen_32b_distill_gen_solution_step_2.json"
    model_path = "/map-vepfs/yubo/models/DeepSeek-R1-Distill-Qwen-32B"
    # model_path = "/map-vepfs/yubo/models/Qwen2.5-32B"

    # 检查是否有中间结果文件存在
    import os
    temp_output_file = output_file + ".temp"
    output_data = []
    processed_count = 0

    if os.path.exists(temp_output_file):
        # 如果存在临时文件，加载已处理的数据
        with open(temp_output_file, "r") as ft:
            output_data = json.load(ft)
        processed_count = len(output_data)
        print(f"Resuming from previous run. Already processed {processed_count} items.")

    llm, sampling_params = load_vllm_model(model_path)
    with open(input_file, "r") as fi:
        deepmath_data = json.load(fi)

    input_data = []
    prompts = []
    idx = 0
    for each in deepmath_data:
        idx += 1
        question = each["question"].replace("Question:\n", "")
        input_data.append({"idx": idx, "question": question})
        idx += 1
        prompts.append(get_prompt(question))

    print("len(prompts)", len(prompts))
    print("prompts[0]", prompts[0])

    # 设置批处理大小
    # batch_size = 1000
    batch_size = 1000

    # 计算剩余需要处理的批次数
    remaining_prompts = prompts[processed_count:]
    remaining_input_data = input_data[processed_count:]
    total_batches = (len(remaining_prompts) + batch_size - 1) // batch_size  # 向上取整

    for batch_idx in range(total_batches):
        start_time = time.time()
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(remaining_prompts))
        batch_prompt = remaining_prompts[batch_start:batch_end]

        print(
            f"Processing batch {batch_idx + 1}/{total_batches}, items {processed_count + batch_start + 1} to {processed_count + batch_end}")
        batch_output = batch_predict(llm, sampling_params, batch_prompt)

        # 添加这批次的结果到输出数据
        for i, output in enumerate(batch_output):
            data_idx = batch_start + i
            curr = remaining_input_data[data_idx].copy()
            curr["solution"] = output
            output_data.append(curr)

        # 每完成一批次就保存临时文件
        with open(temp_output_file, "w") as fo:
            fo.write(json.dumps(output_data, indent=4))

        print(f"Batch {batch_idx + 1} complete. Progress saved. Batch processing time:", time.time() - start_time)

    # 所有批次处理完成后，将临时文件重命名为最终输出文件
    # import shutil
    # shutil.move(temp_output_file, output_file)
    print(f"All processing complete. Total processed items: {len(output_data)}")

    # 检查数据一致性
    if len(output_data) != len(input_data):
        print(f"Warning: Output length ({len(output_data)}) doesn't match input length ({len(input_data)})")


main()
