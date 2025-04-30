
#source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
#conda activate cft

summary_path="../baseline_eval_results_0428/qwen3_32b/summary.txt"
model_path="/data/yubowang/models/DeepSeek-R1-Distill-Qwen-32B"
output_path="../baseline_eval_results_0428/qwen3_32b/"

mkdir -p $output_path

cd /data/yubowang/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash evaluate_distilled_qwen.sh $model_path $output_path $summary_path





