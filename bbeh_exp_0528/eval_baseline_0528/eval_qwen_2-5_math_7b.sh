summary_path="../../baseline_eval_results_0528/Qwen2.5-Math-7B/summary.txt"
model_path="/data/yubo/models/Qwen2.5-Math-7B"
output_path="../../baseline_eval_results_0528/Qwen2.5-Math-7B/"

cd /data/yubo/CriticCoT/bbeh/bbeh

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=4,5,6,7

bash eval_qwen2-5.sh $model_path $output_path $summary_path
