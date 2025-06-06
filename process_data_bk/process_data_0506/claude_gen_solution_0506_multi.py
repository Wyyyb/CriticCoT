import json
import time
import re
import os
import anthropic
import multiprocessing
from typing import List, Dict, Any, Optional
from functools import partial
import glob


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
                    "budget_tokens": 10000,
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


def process_batch(process_id, batch_questions, api_key, model_name, output_dir, main_output_file):
    """
    处理一批问题并将结果保存到进程特定的文件中

    Args:
        process_id: 进程ID
        batch_questions: 要处理的问题列表
        api_key: Claude API密钥
        model_name: 使用的Claude模型
        output_dir: 输出目录
        main_output_file: 主输出文件名（用于生成进程特定的文件名）

    Returns:
        处理的统计信息
    """
    # 初始化客户端
    client = setup_claude_client(api_key)

    # 为每个进程创建独立的输出文件
    base_filename = os.path.basename(main_output_file)
    filename_without_ext, ext = os.path.splitext(base_filename)
    process_output_file = os.path.join(output_dir, f"{filename_without_ext}_process_{process_id}{ext}")

    # 初始化统计数据和结果字典
    stats = {
        "total": 0,
        "valid_answers": 0,
        "invalid_answers": 0,
        "correct_answers": 0,
        "incorrect_answers": 0
    }

    results = {}

    # 如果进程的输出文件已存在，加载它
    if os.path.exists(process_output_file):
        with open(process_output_file, "r") as fi:
            results = json.load(fi)
        print(f"进程 {process_id} 已加载 {len(results)} 条结果从 {process_output_file}")

    # 处理批次中的每个问题
    for q_data in batch_questions:
        question_id = str(q_data.get("id"))
        question_text = q_data["question"]
        gt_answer = q_data.get("gt_answer", None)

        # 检查是否已经处理过且有效
        if question_id in results and results[question_id].get("claude_answer_valid", False):
            print(f"进程 {process_id}: 问题 {question_id} 已处理且有效，跳过")
            continue

        print(f"进程 {process_id}: 处理问题 {question_id}...")

        try:
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

            stats["total"] += 1

            if extracted_answer is None:
                results[question_id]["claude_answer_valid"] = False
                stats["invalid_answers"] += 1
            else:
                results[question_id]["claude_answer_valid"] = True
                results[question_id]["claude_extracted_answer"] = extracted_answer
                stats["valid_answers"] += 1

                # 检查答案正确性
                if gt_answer is not None:
                    if is_same_answer(extracted_answer, gt_answer):
                        results[question_id]["claude_answer_correctness"] = True
                        stats["correct_answers"] += 1
                    else:
                        results[question_id]["claude_answer_correctness"] = False
                        stats["incorrect_answers"] += 1

            # 每个问题处理后立即保存到进程特定的文件
            with open(process_output_file, "w") as fo:
                json.dump(results, fo, indent=4)

        except Exception as e:
            print(f"进程 {process_id}: 处理问题 {question_id} 出错: {str(e)}")
            if question_id not in results:
                results[question_id] = {
                    "id": question_id,
                    "question": question_text
                }

            results[question_id]["error"] = str(e)
            results[question_id]["claude_answer_valid"] = False
            stats["invalid_answers"] += 1

            # 错误后也立即保存
            with open(process_output_file, "w") as fo:
                json.dump(results, fo, indent=4)

        # 短暂等待，避免API限制
        time.sleep(1)

    print(f"进程 {process_id} 完成。统计: {stats}")
    return stats


def merge_result_files(output_dir, main_output_file):
    """
    合并所有进程的输出文件到主输出文件

    Args:
        output_dir: 输出目录
        main_output_file: 主输出文件路径

    Returns:
        合并后的结果字典
    """
    # 获取进程特定输出文件的模式
    base_filename = os.path.basename(main_output_file)
    filename_without_ext, ext = os.path.splitext(base_filename)
    file_pattern = os.path.join(output_dir, f"{filename_without_ext}_process_*{ext}")

    # 合并结果
    merged_results = {}

    # 如果主输出文件存在，首先加载它
    if os.path.exists(main_output_file):
        with open(main_output_file, "r") as fi:
            merged_results = json.load(fi)

    # 加载所有进程文件并合并
    for process_file in glob.glob(file_pattern):
        if os.path.exists(process_file):
            with open(process_file, "r") as fi:
                process_results = json.load(fi)

            # 合并到主结果中
            for q_id, result in process_results.items():
                # 只更新未处理或无效的结果
                if q_id not in merged_results or not merged_results[q_id].get("claude_answer_valid", False):
                    merged_results[q_id] = result

    # 保存合并结果
    with open(main_output_file, "w") as fo:
        json.dump(merged_results, fo, indent=4)

    print(f"已合并所有结果到 {main_output_file}，共 {len(merged_results)} 条记录")
    return merged_results


def load_existing_results(output_dir, main_output_file):
    """
    加载所有现有结果（主文件和进程文件）

    Args:
        output_dir: 输出目录
        main_output_file: 主输出文件路径

    Returns:
        合并后的现有结果字典
    """
    existing_results = {}

    # 1. 加载主输出文件
    if os.path.exists(main_output_file):
        with open(main_output_file, "r") as fi:
            main_results = json.load(fi)
            existing_results.update(main_results)

    # 2. 加载所有进程特定的文件
    base_filename = os.path.basename(main_output_file)
    filename_without_ext, ext = os.path.splitext(base_filename)
    file_pattern = os.path.join(output_dir, f"{filename_without_ext}_process_*{ext}")

    for process_file in glob.glob(file_pattern):
        if os.path.exists(process_file):
            with open(process_file, "r") as fi:
                process_results = json.load(fi)

            # 合并到结果中，优先保留有效的结果
            for q_id, result in process_results.items():
                if q_id not in existing_results or (
                        result.get("claude_answer_valid", False) and
                        not existing_results[q_id].get("claude_answer_valid", False)
                ):
                    existing_results[q_id] = result

    print(f"已加载所有现有结果，共 {len(existing_results)} 条记录")
    return existing_results


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


def distribute_questions(questions_to_process, num_processes):
    """
    将问题分配给各个进程

    Args:
        questions_to_process: 要处理的问题列表
        num_processes: 进程数量

    Returns:
        分配给每个进程的问题列表
    """
    batches = [[] for _ in range(num_processes)]

    # 循环分配，确保工作负载均匀
    for i, question in enumerate(questions_to_process):
        process_id = i % num_processes
        batches[process_id].append(question)

    for i, batch in enumerate(batches):
        print(f"进程 {i} 分配了 {len(batch)} 个问题")

    return batches


def main_claude_solver_multiprocessing(
        input_file: str,
        output_file: str,
        api_key: str,
        model_name: str = "claude-3-7-sonnet-20250219",
        num_processes: int = 1
):
    """
    使用Claude API解答数学问题的主函数 (多进程版本，每个进程使用独立文件)

    Args:
        input_file: 包含问题的JSON文件路径
        output_file: 输出文件路径
        api_key: Claude API密钥
        model_name: 使用的Claude模型
        num_processes: 进程数量
    """
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 首先加载所有输入数据
    with open(input_file, "r") as fi:
        input_data = json.load(fi)
    print(f"从 {input_file} 加载了 {len(input_data)} 个问题")

    # 加载所有现有结果（主文件和所有进程文件）
    existing_results = load_existing_results(output_dir, output_file)

    # 确定需要处理的问题列表
    questions_to_process, _ = get_questions_to_process(input_data, existing_results)
    print(f"需要处理的问题数量: {len(questions_to_process)}")

    # 如果有问题需要处理
    if questions_to_process:
        # 将问题分配给各个进程
        batches = distribute_questions(questions_to_process, num_processes)

        # 创建进程池
        pool = multiprocessing.Pool(processes=num_processes)

        # 准备任务参数
        tasks = []
        for i in range(num_processes):
            if batches[i]:  # 只处理有问题的批次
                tasks.append((i, batches[i], api_key, model_name, output_dir, output_file))

        # 并行处理批次
        pool.starmap(process_batch, tasks)

        # 关闭进程池
        pool.close()
        pool.join()

        # 合并所有结果文件
        merge_result_files(output_dir, output_file)

        # 检查是否有未完成的问题
        with open(output_file, "r") as fi:
            final_results = json.load(fi)

        questions_to_process, _ = get_questions_to_process(input_data, final_results)

        # 如果还有未处理的问题，继续处理
        if questions_to_process:
            print(f"处理后仍有 {len(questions_to_process)} 个问题需要处理，继续下一轮...")
            main_claude_solver_multiprocessing(
                input_file, output_file, api_key, model_name, num_processes
            )
        else:
            print("所有问题都已处理完毕！")
    else:
        print("没有需要处理的问题！")

    print(f"处理完成。结果保存至 {output_file}")


# 示例用法
if __name__ == "__main__":
    API_KEY = os.getenv("CLAUDE_API_KEY")
    INPUT_FILE = "../local_data/cft_data_0506/webinstruct_train.json"  # 包含数学问题的JSON文件
    OUTPUT_FILE = "../local_data/cft_data_0506/webinstruct_claude_add_solutions_0510.json"  # 结果输出文件路径
    NUM_PROCESSES = 2  # 默认进程数，可以根据需要调整

    main_claude_solver_multiprocessing(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        api_key=API_KEY,
        model_name="claude-3-7-sonnet-20250219",
        num_processes=NUM_PROCESSES
    )