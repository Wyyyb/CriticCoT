import json
import time
import re
import os
import anthropic
from typing import List, Dict, Any, Optional


def setup_claude_client(api_key: str):
    """初始化Claude客户端"""
    return anthropic.Anthropic(api_key=api_key)


def get_claude_prompt(question: str) -> str:
    """构建提示词"""
    return f"Please reason step by step to find a solution to the following question, and put your final answer within \\boxed{{}}.\nQuestion:\n{question}"


def extract_boxed_answer(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return None
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def is_same_answer(str_1, str_2):
    if not str_1 or not str_2:
        return False
    if str_1.replace("dfrac", "frac").lower().replace(",", "").replace(" ", "") \
            == str_2.replace("dfrac", "frac").lower().replace(",", "").replace(" ", ""):
        # if str_1 != str_2:
        #     print("str_1, str_2", str_1, str_2)
        return True
    return str_1 == str_2


def query_claude(client, question: str, model_name: str = "claude-3-7-sonnet-20250219") -> Dict[str, str]:
    """
    向Claude API发送查询并获取回答

    Args:
        client: Claude API客户端
        question: 问题文本
        model_name: 使用的Claude模型名称

    Returns:
        包含答案和思考过程的字典
    """
    print("prompt", get_claude_prompt(question))
    messages = [
        {
            "role": "user",
            "content": get_claude_prompt(question)
        }
    ]

    delta_text_in_completion = []
    delta_thinking_in_completion = []

    try:
        with client.messages.stream(
                model=model_name,
                messages=messages,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
                max_tokens=36000
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        delta_thinking_in_completion.append(event.delta.thinking)
                    elif event.delta.type == "text_delta":
                        delta_text_in_completion.append(event.delta.text)

        completion_thinking = "".join(delta_thinking_in_completion)
        completion_text = "".join(delta_text_in_completion)
        print("completion_text", completion_text)
        print("completion_thinking", completion_thinking)

        return {
            "answer": completion_text,
            "thinking": completion_thinking
        }
    except Exception as e:
        print(f"Error querying Claude: {str(e)}")
        return {
            "answer": "",
            "thinking": "",
            "error": str(e)
        }


def batch_solve_with_claude(
        client,
        questions: List[Dict[str, Any]],
        batch_size: int = 10,
        model_name: str = "claude-3-7-sonnet-20250219",
        output_file: str = "claude_solutions.json",
        existing_results: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    使用Claude API批量解答数学问题

    Args:
        client: Claude API客户端
        questions: 包含问题的字典列表
        batch_size: 每批处理的问题数
        model_name: 使用的Claude模型
        output_file: 输出文件路径
        existing_results: 已有的结果字典，用于断点续传

    Returns:
        包含所有问题及其解答的字典
    """
    # 初始化结果字典，如果有已存在的结果则加载
    results = existing_results if existing_results else {}

    total_batches = (len(questions) + batch_size - 1) // batch_size

    # 初始化统计数据
    stats = {
        "total": 0,
        "valid_answers": 0,
        "invalid_answers": 0,
        "correct_answers": 0,
        "incorrect_answers": 0
    }

    for batch_idx in range(total_batches):
        start_time = time.time()
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(questions))
        batch_questions = questions[batch_start:batch_end]

        print(f"Processing batch {batch_idx + 1}/{total_batches}, items {batch_start + 1} to {batch_end}")

        batch_stats = {"total": 0, "valid": 0, "invalid": 0, "correct": 0, "incorrect": 0}

        for q_data in batch_questions:
            question_id = str(q_data.get("id", len(results)))  # 确保ID是字符串
            question_text = q_data["question"]
            gt_answer = q_data.get("gt_answer", None)

            # 检查是否已经处理过且有效
            if question_id in results and results[question_id].get("claude_answer_valid", False):
                print(f"Question {question_id} already processed with valid answer, skipping")
                continue

            try:
                # print(f"Processing question {question_id}...")
                # 调用Claude API
                claude_response = query_claude(
                    client=client,
                    question=question_text,
                    model_name=model_name
                )

                answer_text = claude_response.get("answer", "")
                thinking_text = claude_response.get("thinking", "")

                # 保存结果
                if question_id not in results:
                    results[question_id] = {
                        "id": question_id,
                        "question": question_text
                    }
                    if gt_answer:
                        results[question_id]["gt_answer"] = gt_answer

                results[question_id]["claude_answer"] = answer_text
                results[question_id]["claude_thinking"] = thinking_text

                # 提取答案
                extracted_answer = extract_boxed_answer(answer_text)

                batch_stats["total"] += 1
                stats["total"] += 1

                if extracted_answer is None:
                    results[question_id]["claude_answer_valid"] = False
                    batch_stats["invalid"] += 1
                    stats["invalid_answers"] += 1
                else:
                    results[question_id]["claude_answer_valid"] = True
                    results[question_id]["claude_extracted_answer"] = extracted_answer
                    batch_stats["valid"] += 1
                    stats["valid_answers"] += 1

                    # 检查答案正确性
                    if gt_answer is not None:
                        if is_same_answer(extracted_answer, gt_answer):
                            results[question_id]["claude_answer_correctness"] = True
                            batch_stats["correct"] += 1
                            stats["correct_answers"] += 1
                        else:
                            results[question_id]["claude_answer_correctness"] = False
                            batch_stats["incorrect"] += 1
                            stats["incorrect_answers"] += 1

                # 每个问题处理完后立即保存，确保断点续传
                with open(output_file, "w") as fo:
                    json.dump(results, fo, indent=4)

            except Exception as e:
                print(f"Error processing question {question_id}: {str(e)}")
                if question_id not in results:
                    results[question_id] = {
                        "id": question_id,
                        "question": question_text
                    }

                results[question_id]["error"] = str(e)
                results[question_id]["claude_answer_valid"] = False
                batch_stats["invalid"] += 1
                stats["invalid_answers"] += 1

                # 错误后也立即保存，确保断点续传
                with open(output_file, "w") as fo:
                    json.dump(results, fo, indent=4)

        # 每批次后保存结果并打印统计信息
        print(f"Batch {batch_idx + 1} complete. Stats: {batch_stats}")

        # 添加10秒等待时间
        wait_time = 10
        print(f"Waiting for {wait_time} seconds before next batch...")
        time.sleep(wait_time)

        print(f"Batch processing time (including wait): {time.time() - start_time:.2f}s")

    print(f"Processing complete. Final stats: {stats}")
    return results


def get_questions_to_process(input_data, existing_results, start_idx=0, end_idx=100000000):
    """
    获取需要处理的问题列表

    Args:
        input_data: 输入数据（所有问题）
        existing_results: 已处理的结果
        start_idx: 开始索引
        end_idx: 结束索引

    Returns:
        需要处理的问题列表和统计信息
    """
    questions_to_process = []
    stats = {
        "total_input": len(input_data),
        "already_processed": 0,
        "to_process": 0
    }

    # 处理所有输入问题
    for q_id, q_data in input_data.items():
        # 只处理指定范围内的问题
        if True:
            # 跳过已经成功处理的问题
            if q_id in existing_results and existing_results[q_id].get("claude_answer_valid", False):
                stats["already_processed"] += 1
                continue

            # 添加到待处理列表
            questions_to_process.append({
                "id": q_id,
                "question": q_data["question"],
                "gt_answer": q_data.get("gt_answer", None)
            })
            stats["to_process"] += 1

    print(
        f"统计信息: 总问题数={stats['total_input']}, 已处理={stats['already_processed']}, 待处理={stats['to_process']}")
    return questions_to_process, stats


def main_claude_solver(
        input_file: str,
        output_file: str,
        api_key: str,
        model_name: str = "claude-3-7-sonnet-20250219",
        batch_size: int = 5,
        start_idx: int = 0,
        end_idx: int = 100000
):
    """
    使用Claude API解答数学问题的主函数

    Args:
        input_file: 包含问题的JSON文件路径
        output_file: 输出文件路径
        api_key: Claude API密钥
        model_name: 使用的Claude模型
        batch_size: 每批处理的问题数
        start_idx: 开始处理的索引
        end_idx: 结束处理的索引
    """
    # 初始化客户端
    client = setup_claude_client(api_key)

    # 首先加载所有输入数据
    with open(input_file, "r") as fi:
        input_data = json.load(fi)
    print(f"从 {input_file} 加载了 {len(input_data)} 个问题")

    # 检查是否已有部分处理结果
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as fi:
            existing_results = json.load(fi)
        print(f"已加载 {len(existing_results)} 条存在的结果从 {output_file}")

    # 确定需要处理的问题列表（包括未处理和处理失败的）
    questions_to_process, _ = get_questions_to_process(
        input_data,
        existing_results,
        start_idx,
        end_idx
    )

    print(f"需要处理的问题数量: {len(questions_to_process)}")

    # 如果有问题需要处理，进行批量处理
    if questions_to_process:
        # 批量解题
        results = batch_solve_with_claude(
            client=client,
            questions=questions_to_process,
            batch_size=batch_size,
            model_name=model_name,
            output_file=output_file,
            existing_results=existing_results
        )

        # 再次检查是否有未完成的问题
        questions_to_process, _ = get_questions_to_process(
            input_data,
            results,
            start_idx,
            end_idx
        )

        # 如果还有未处理数据，继续处理
        if len(questions_to_process) > 0:
            print(f"第一轮处理后仍有 {len(questions_to_process)} 个问题需要处理，继续下一轮...")
            main_claude_solver(input_file, output_file, api_key, model_name, batch_size, start_idx, end_idx)
        else:
            print("所有问题都已处理完毕！")
    else:
        print("没有需要处理的问题！")

    print(f"处理完成。结果保存至 {output_file}")


# 示例用法
if __name__ == "__main__":
    API_KEY = os.getenv("CLAUDE_API_KEY")
    INPUT_FILE = "../local_data/cft_data_0506/webinstruct_train.json"  # 包含数学问题的JSON文件
    OUTPUT_FILE = "../local_data/cft_data_0506/webinstruct_claude_add_solutions_0506_process_2.json"  # 结果输出文件路径

    main_claude_solver(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        api_key=API_KEY,
        model_name="claude-3-7-sonnet-20250219",
        batch_size=1  # Claude API通常限制并发请求，建议使用较小的批量
    )




