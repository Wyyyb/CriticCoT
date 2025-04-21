
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

cd /map-vepfs/yubo/CriticCoT/process_data_0419_map

export CUDA_VISIBLE_DEVICES=0,1
python qwen_32b_distill_gen_deepmath_solution_0420_p2.py
