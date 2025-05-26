

summary_path="../../baseline_eval_results_0523/qwen3_4b/summary.txt"
model_path="/data/yubowang/models/Qwen3-4B"
output_path="../../baseline_eval_results_0523/qwen3_4b/"

cd /data/yubowang/CriticCoT/bbeh/bbeh

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash eval_qwen2-5.sh $model_path $output_path $summary_path




