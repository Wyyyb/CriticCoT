import json
import time
import re
import os
import multiprocessing
from typing import List, Dict, Any, Optional
from functools import partial
import glob
from call_api_functions import *


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


def extract_con(critique):
    # if "Conclusion" not in text:
    #     print("Conclusion not found", text)
    #     return None
    segs = critique.lower().split("conclusion")
    res_seg = segs[-1].lower()
    if "wrong" in res_seg or "incorrect" in res_seg:
        return False
    elif "right" in res_seg or "correct" in res_seg:
        return True
    else:
        # print("*******************recognize failed\n", res_seg, text)
        return None


def is_same_answer(str_1, str_2):
    if not str_1 or not str_2:
        return False
    if str_1.replace("dfrac", "frac").lower().replace(",", "").replace(" ", "") \
            == str_2.replace("dfrac", "frac").lower().replace(",", "").replace(" ", ""):
        return True
    return str_1 == str_2


def process_batch(process_id, batch_questions, api_key, model_name, output_dir, main_output_file):
    # 初始化客户端
    model_info_map = {"claude-3-7-sonnet-20250219": "anthropic-think",
                      "claude-3-5-sonnet-20241022": "anthropic",
                      "gemini-2.5-pro-preview-05-06": "gemini",
                      "gemini-2.5-pro-exp-03-25": "gemini",
                      "gpt-4o-2024-11-20": "openai",
                      "gpt-4o-2024-08-06": "openai",
                      "gpt-4.1-2025-04-14": "openai",
                      "gpt-4.1-mini-2025-04-14": "openai",
                      # "o3-2025-04-16": "openai-reasoning",
                      "o4-mini-2025-04-16": "openai-reasoning",
                      "o3-mini-2025-01-31": "openai-reasoning",
                      "o1-2024-12-17": "openai-reasoning",
                      "grok-3": "grok",
                      "deepseek-r1": "deepseek"}
    model_type = model_info_map.get(model_name, None)
    if not model_type:
        print("unsupported model:", model_name)
        return None
    client = get_client(model_type, model_name, api_key)

    # 为每个进程创建独立的输出文件
    base_filename = os.path.basename(main_output_file)
    filename_without_ext, ext = os.path.splitext(base_filename)
    process_output_file = os.path.join(output_dir, f"{filename_without_ext}_process_{process_id}{ext}")

    # 初始化统计数据和结果字典
    stats = {
        "total": 0,
        "valid_cft_answers": 0,
        "invalid_cft_answers": 0,
        "valid_cft_conclusion": 0,
        "invalid_cft_conclusion": 0
    }

    results = {}

    # 如果进程的输出文件已存在，加载它
    if os.path.exists(process_output_file):
        with open(process_output_file, "r") as fi:
            results = json.load(fi)
        print(f"进程 {process_id} 已加载 {len(results)} 条结果从 {process_output_file}")

    # 处理批次中的每个问题
    for q_data in batch_questions:
        question_id = str(q_data.get("critique_id"))
        results[question_id] = q_data
        question_text = q_data["question"]
        solution = q_data["student_solution"]["solution"]
        solution = remove_thinking(solution)

        # 检查是否已经处理过且有效
        if question_id in results and results[question_id].get("critique_extracted_answer", None) is not None\
                and results[question_id].get("critique_extracted_conclusion", None) is not None:
            print(f"进程 {process_id}: 问题 {question_id} 已处理且有效，跳过")
            continue

        print(f"进程 {process_id}: 处理问题 {question_id}...")

        try:
            # 调用API
            message = get_messages(model_type, question_text, solution)
            answer_text = send_request(client, model_type, model_name, message)
            results[question_id]["critique"] = answer_text
            # 提取答案
            extracted_answer = extract_boxed_answer(answer_text)
            stats["total"] += 1

            if extracted_answer is None:
                results[question_id]["critique_extracted_answer"] = extracted_answer
                stats["invalid_cft_answers"] += 1
            else:
                results[question_id]["critique_extracted_answer"] = extracted_answer
                stats["valid_cft_answers"] += 1

            # 提取结论
            extracted_con = extract_con(answer_text)
            results[question_id]["critique_extracted_conclusion"] = extracted_con
            if extracted_con is None:
                stats["invalid_cft_conclusion"] += 1
            else:
                stats["valid_cft_conclusion"] += 1

            # 每个问题处理后立即保存到进程特定的文件
            results[question_id]["teacher_model"] = model_name
            with open(process_output_file, "w") as fo:
                json.dump(results, fo, indent=4)

        except Exception as e:
            print(f"进程 {process_id}: 处理问题 {question_id} 出错: {str(e)}")
            if question_id not in results:
                results[question_id] = {
                    "critique_id": question_id,
                    "question": question_text
                }

            results[question_id]["error"] = str(e)
            stats["invalid_cft_answers"] += 1

            # 错误后也立即保存
            with open(process_output_file, "w") as fo:
                json.dump(results, fo, indent=4)

        # 短暂等待，避免API限制
        time.sleep(5)

    print(f"进程 {process_id} 完成。统计: {stats}")
    return stats


def remove_thinking(solution):
    if "</think>" in solution:
        solution = solution.split("</think>")[-1].strip()
    return solution


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
                if q_id not in merged_results or not merged_results[q_id].get("claude_cft_answer_valid", False):
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
                        result.get("claude_cft_answer_valid", False) and
                        not existing_results[q_id].get("claude_cft_answer_valid", False)
                ):
                    existing_results[q_id] = result

    print(f"已加载所有现有结果，共 {len(existing_results)} 条记录")
    return existing_results


def get_questions_to_process(input_data, existing_results, start_idx=0, end_idx=20000):
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
            if int(q_data["critique_id"]) < start_idx or int(q_data["critique_id"]) >= end_idx:
                continue
            if q_id in existing_results and existing_results[q_id].get("claude_answer_valid", False):
                stats["already_processed"] += 1
                continue

            # 添加到待处理列表
            questions_to_process.append(q_data)
            stats["to_process"] += 1

    print(
        f"统计信息: 总问题数={stats['total_input']}, 已处理={stats['already_processed']}, 待处理={stats['to_process']}")
    return questions_to_process, stats


def distribute_questions(questions_to_process, num_processes):

    batches = [[] for _ in range(num_processes)]

    # 循环分配，确保工作负载均匀
    for i, question in enumerate(questions_to_process):
        process_id = i % num_processes
        batches[process_id].append(question)

    for i, batch in enumerate(batches):
        print(f"进程 {i} 分配了 {len(batch)} 个问题")

    return batches


def main_solver_multiprocessing(
        input_file: str,
        output_file: str,
        api_key: str,
        model_name: str,
        num_processes: int = 1,
        start_idx: int = 0,
        end_idx: int = 20000
):

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
    questions_to_process, _ = get_questions_to_process(input_data, existing_results, start_idx, end_idx)
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

        questions_to_process, _ = get_questions_to_process(input_data, final_results, start_idx, end_idx)

        # 如果还有未处理的问题，继续处理
        if questions_to_process:
            print(f"处理后仍有 {len(questions_to_process)} 个问题需要处理，继续下一轮...")
            # main_solver_multiprocessing(
            #     input_file, output_file, api_key, model_name, num_processes, start_idx, end_idx
            # )
        else:
            print("所有问题都已处理完毕！")
    else:
        print("没有需要处理的问题！")

    print(f"处理完成。结果保存至 {output_file}")


# 示例用法
if __name__ == "__main__":
    API_KEY = os.getenv("OPENAI_API_KEY")
    # teacher_model_name可以从下列列表中选
    teacher_models = [
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "gpt-4.1-mini-2025-04-14"
        "gpt-4.1-2025-04-14",
        "gpt-4o-2024-11-20",
        # "gpt-4o-2024-08-06",
        # "o3-2025-04-16",
        # "o4-mini-2025-04-16",
        "o3-mini-2025-01-31",
        "o1-2024-12-17"
    ]
    teacher_model_name = "o1-2024-12-17"
    INPUT_FILE = "dsr_p0_100_to_critique_0518_qwen_math_7b.json"
    OUTPUT_FILE = f"dsr_p0_100_with_critique_{teacher_model_name}_qwen_math_7b_0518.json"  # 结果输出文件路径
    NUM_PROCESSES = 20  # 进程数

    main_solver_multiprocessing(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        api_key=API_KEY,
        model_name=teacher_model_name,
        num_processes=NUM_PROCESSES,
        start_idx=0,
        end_idx=8000
    )