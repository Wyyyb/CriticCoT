import json
import time
import re
from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import os


def setup_gemini_client(api_key: str):
    """初始化Gemini客户端"""
    return genai.Client(api_key=api_key)


def get_gemini_prompt(question: str) -> str:
    """构建提示词"""
    return f"Please reason step by step to find a solution to the following question, and put your final answer within \\boxed{{}}.\nQuestion:\n{question}"


def extract_boxed_answer(text: str) -> Optional[str]:
    """从输入的字符串中提取\boxed{ANSWER}中的ANSWER部分"""
    if not text or len(text) > 50000:
        print("Text too long or empty:", len(text) if text else 0)
        return None

    pattern = r'\\boxed\{(.*?)\}'
    match = re.search(pattern, text)

    if match:
        return match.group(1)
    else:
        return None


def batch_solve_with_gemini(
        client,
        questions: List[Dict[str, Any]],
        batch_size: int = 10,
        model_name: str = "gemini-2.5-pro-exp-03-25",
        output_file: str = "gemini_solutions.json"
) -> Dict[str, Any]:
    """
    使用Gemini API批量解答数学问题

    Args:
        client: Gemini API客户端
        questions: 包含问题的字典列表
        batch_size: 每批处理的问题数
        model_name: 使用的Gemini模型
        output_file: 输出文件路径

    Returns:
        包含所有问题及其解答的字典
    """
    results = {}
    total_batches = (len(questions) + batch_size - 1) // batch_size

    # 初始化统计数据
    stats = {
        "total": 0,
        "valid_answers": 0,
        "invalid_answers": 0,
        "correct_answers": 0,
        "incorrect_answers": 0
    }

    for batch_idx in tqdm(range(total_batches)):
        start_time = time.time()
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(questions))
        batch_questions = questions[batch_start:batch_end]

        print(f"Processing batch {batch_idx + 1}/{total_batches}, items {batch_start + 1} to {batch_end}")

        batch_stats = {"total": 0, "valid": 0, "invalid": 0, "correct": 0, "incorrect": 0}

        for q_data in batch_questions:
            question_id = q_data.get("id", str(len(results)))
            question_text = q_data["question"]
            gt_answer = q_data.get("gt_answer", None)

            try:
                # 构建提示词
                prompt = get_gemini_prompt(question_text)
                # print("curr prompt:", prompt)
                # 调用Gemini API
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=16384)
                    ),
                )

                answer_text = response.text
                # print("answer_text:", answer_text)
                # 保存结果
                result_item = {
                    "id": question_id,
                    "question": question_text,
                    "gemini_answer": answer_text,
                }

                # 提取答案
                extracted_answer = extract_boxed_answer(answer_text)

                batch_stats["total"] += 1
                stats["total"] += 1

                if extracted_answer is None:
                    result_item["gemini_answer_valid"] = False
                    batch_stats["invalid"] += 1
                    stats["invalid_answers"] += 1
                else:
                    result_item["gemini_answer_valid"] = True
                    result_item["gemini_extracted_answer"] = extracted_answer
                    batch_stats["valid"] += 1
                    stats["valid_answers"] += 1

                    # 如果有参考答案，检查正确性
                    if gt_answer is not None:
                        if extracted_answer == gt_answer:
                            result_item["gemini_answer_correctness"] = True
                            batch_stats["correct"] += 1
                            stats["correct_answers"] += 1
                        else:
                            result_item["gemini_answer_correctness"] = False
                            batch_stats["incorrect"] += 1
                            stats["incorrect_answers"] += 1

                results[question_id] = result_item

            except Exception as e:
                print(f"Error processing question {question_id}: {str(e)}")
                results[question_id] = {
                    "id": question_id,
                    "question": question_text,
                    "error": str(e),
                    "gemini_answer_valid": False
                }
                batch_stats["invalid"] += 1
                stats["invalid_answers"] += 1

            time.sleep(20)
        # 每批次后保存结果
        with open(output_file, "w") as fo:
            json.dump(results, fo, indent=4)

        print(f"Batch {batch_idx + 1} complete. Stats: {batch_stats}")
        print(f"Batch processing time: {time.time() - start_time:.2f}s")

    print(f"Processing complete. Final stats: {stats}")
    return results


def main_gemini_solver(
        input_file: str,
        output_file: str,
        api_key: str,
        model_name: str = "gemini-2.5-pro-exp-03-25",
        batch_size: int = 10
):
    """
    使用Gemini API解答数学问题的主函数

    Args:
        input_file: 包含问题的JSON文件路径
        output_file: 输出文件路径
        api_key: Gemini API密钥
        model_name: 使用的Gemini模型
        batch_size: 每批处理的问题数
    """
    # 初始化客户端
    client = setup_gemini_client(api_key)

    # 加载问题数据
    with open(input_file, "r") as fi:
        input_data = json.load(fi)

    print(f"Loaded {len(input_data)} questions from {input_file}")

    # 检查是否已有部分处理结果
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as fi:
            existing_results = json.load(fi)
        print(f"Loaded {len(existing_results)} existing results from {output_file}")

    # 筛选未处理的问题
    questions_to_process = []
    for q_id, q_data in input_data.items():
        if q_id not in existing_results or not existing_results[q_id].get("gemini_answer_valid", False):
            questions_to_process.append(q_data)

    print(f"Found {len(questions_to_process)} questions to process")

    # 如果有既有结果，先加载它们
    results = existing_results if existing_results else {}

    # 批量解题
    if questions_to_process:
        results = batch_solve_with_gemini(
            client=client,
            questions=questions_to_process,
            batch_size=batch_size,
            model_name=model_name,
            output_file=output_file
        )

    print(f"Processing complete. Results saved to {output_file}")
    return results


# 配置参数
my_api_key_p1 = "AIzaSyC4SgiM-"
my_api_key_p2 = "OXAtW0UjbepH3GBLHV-ShNOF_E"
API_KEY = my_api_key_p1 + my_api_key_p2
INPUT_FILE = "../local_data/cft_data_0506/webinstruct_train.json"  # 包含数学问题的JSON文件
OUTPUT_FILE = "../local_data/cft_data_0506/webinstruct_gemini_add_solutions_0506.json"  # 结果输出文件路径

# 调用主函数
if __name__ == "__main__":
    main_gemini_solver(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        api_key=API_KEY,
        model_name="gemini-2.5-pro-exp-03-25",
        batch_size=2  # 根据API限制和需求调整
    )


