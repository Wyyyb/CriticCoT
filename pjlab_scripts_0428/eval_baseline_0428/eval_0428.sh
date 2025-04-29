
source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate cft

summary_path="../baseline_eval_results_0428/qwen_32b_distill/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/DeepSeek-R1-Distill-Qwen-32B"
output_path="../baseline_eval_results_0428/qwen_32b_distill/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=0,1,6,7

bash evaluate_single.sh $model_path $output_path $summary_path





