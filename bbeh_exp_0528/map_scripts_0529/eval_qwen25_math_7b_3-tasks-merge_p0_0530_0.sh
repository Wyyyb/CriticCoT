
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_eval

summary_path="/map-vepfs/yubo/CriticCoT/eval_results_0530/summary.txt"
model_path="/map-vepfs/yubo/cft_models/bbeh_qwen25_math_7b_1-shot_cft_causal_understanding_ckpt40"
output_path="/map-vepfs/yubo/CriticCoT/eval_results_0530/bbeh_qwen25_math_7b_1-shot_cft_causal_understanding_ckpt40/"
task_list="bbeh_causal_understanding,bbeh_disambiguation_qa,bbeh_time_arithmetic,bbeh_zz_mini"

cd /map-vepfs/yubo/CriticCoT/bbeh/bbeh

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash eval_qwen2-5_task.sh $model_path $output_path $summary_path $task_list


