export CUDA_VISIBLE_DEVICES=0,1,2,3

conda activate cft

cd /data/yubo/CriticCoT/process_data_0506
# 运行您的Python脚本，并将标准输出和标准错误重定向到指定文件
python qwen2_5_math_7b_gen_wv_solution_0506.py
