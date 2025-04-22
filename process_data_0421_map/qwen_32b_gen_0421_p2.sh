
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

cd /map-vepfs/yubo/CriticCoT/process_data_0421_map

export CUDA_VISIBLE_DEVICES=4,5,6,7
python qwen_32b_gen_solution_0421_p2.py

