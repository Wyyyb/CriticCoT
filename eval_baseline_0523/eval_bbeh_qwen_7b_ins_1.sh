

summary_path="../../baseline_eval_results_0523/qwen2-5_7b_ins_1/summary.txt"
model_path="/data/yubo/models/Qwen2.5-7B-Instruct"
output_path="../../baseline_eval_results_0523/qwen2-5_7b_ins_1/"

cd /data/yubo/CriticCoT/bbeh/bbeh

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=4,5,6,7

bash eval_qwen2-5_test.sh $model_path $output_path $summary_path




