
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_eval

summary_path="/map-vepfs/yubo/CriticCoT/baseline_eval_results_0528/Qwen2.5-Math-7B/summary.txt"
model_path="/map-vepfs/yubo/models/Qwen2.5-Math-7B"
output_path="/map-vepfs/yubo/CriticCoT/baseline_eval_results_0528/Qwen2.5-Math-7B/"
task_list="bbeh_causal_understanding,bbeh_disambiguation_qa,bbeh_boolean_expressions,bbeh_time_arithmetic,bbeh_buggy_tables,bbeh_object_counting,bbeh_zz_mini"

cd /map-vepfs/yubo/CriticCoT/bbeh/bbeh

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=4,5,6,7

bash eval_qwen2-5_task.sh $model_path $output_path $summary_path $task_list

cd /map-vepfs/yubo/CriticCoT/bbeh_exp_0528/map_scripts_0528/

bash eval_qwen25_math_easy_causal_understanding_p1.sh
