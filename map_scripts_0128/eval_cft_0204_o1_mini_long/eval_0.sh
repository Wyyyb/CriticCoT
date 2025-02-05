
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

summary_path="/map-vepfs/yubo/CriticCoT/0204_eval_results_qwen2.5-math-7B-webinstruct_cft_80k_o1_mini_long_0204_0/summary.txt"
model_dir="/map-vepfs/yubo/CriticCoT/LLaMA-Factory/output_models_0204/qwen2.5-math-7B-webinstruct_cft_80k_o1_mini_long_0204"

cd /map-vepfs/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/multi_eval

bash start_8_cards.sh $summary_path $model_dir


