
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

summary_path="/map-vepfs/yubo/CriticCoT/0206_eval_results_qwen2.5-32B-Instruct-webinstruct_cft_80k_o1_mini_long_0204_0/summary.txt"
model_dir="/map-vepfs/yubo/CriticCoT/LLaMA-Factory/output_models_0206/qwen2.5-32B-Instruct-webinstruct_cft_80k_o1_mini_long_0204"

cd /map-vepfs/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/multi_eval_32b

bash start_4_cards.sh $summary_path $model_dir


