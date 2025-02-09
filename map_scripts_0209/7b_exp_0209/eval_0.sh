
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

summary_path="/map-vepfs/yubo/CriticCoT/0209_eval_results_qwen2.5-math-7b_webinstruct_cft_80k_0121_p3_exp0/summary.txt"
model_dir="/map-vepfs/yubo/CriticCoT/LLaMA-Factory/output_models_0208/qwen2.5-math-7b_webinstruct_cft_80k_0121_p3_exp0"
ckpt_start=0
ckpt_end=52

cd /map-vepfs/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/multi_eval_0209_dense

bash start_8_cards.sh $summary_path $model_dir $ckpt_start $ckpt_end


