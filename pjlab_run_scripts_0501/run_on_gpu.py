import torch
import time
import os
import sys
import datetime


def occupy_gpu(duration_seconds=10):
    # 打印当前系统时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current system time: {current_time}")

    # 检查可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    if num_gpus == 0:
        print("No GPUs available. Exiting...")
        return

    # 为每个GPU创建一个大矩阵
    matrices = []
    for i in range(num_gpus):
        # 计算需要的矩阵大小以占用48GB显存
        # 使用float16类型可以占用更多的元素
        matrix_size = 60000  # 约占用48GB显存，根据GPU型号可能需要调整

        print(f"Creating matrix on GPU {i}...")
        with torch.cuda.device(i):
            # 创建两个大矩阵
            matrix_a = torch.rand(matrix_size, matrix_size, dtype=torch.float16, device=f'cuda:{i}')
            matrix_b = torch.rand(matrix_size, matrix_size, dtype=torch.float16, device=f'cuda:{i}')

            # 执行矩阵加法以确保GPU实际工作
            result = matrix_a + matrix_b
            matrices.append((matrix_a, matrix_b, result))

            # 打印当前GPU的显存使用情况
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")

    print("Matrices created and addition performed on all GPUs.")
    print(f"Holding GPU memory for {duration_seconds} seconds...")

    # 等待指定的时间
    time.sleep(duration_seconds)

    # 打印结束时的系统时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finishing at system time: {current_time}")

    # 释放内存
    for i, (a, b, c) in enumerate(matrices):
        del a, b, c
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
        print(f"Released memory on GPU {i}")

    print("All GPU memory released.")


if __name__ == "__main__":
    # 如果提供了参数，使用第一个参数作为持续时间（秒）
    duration = 10
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print(f"Invalid duration: {sys.argv[1]}. Using default: 10 seconds.")

    occupy_gpu(duration)