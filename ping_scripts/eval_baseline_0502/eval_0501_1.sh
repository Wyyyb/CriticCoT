
source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft
#
#summary_path="../baseline_eval_results_0430/Qwen3-8B-Base/summary.txt"
#model_path="/data/yubo/models/Qwen3-8B-Base"
#output_path="../baseline_eval_results_0430/Qwen3-8B-Base/"
#
#mkdir -p $output_path
#
#cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
#
#export CUDA_VISIBLE_DEVICES=6,7
#
#bash evaluate_distilled_qwen.sh $model_path $output_path $summary_path

#
#summary_path="../baseline_eval_results_0430/Qwen3-8B/summary.txt"
#model_path="/data/yubo/models/Qwen3-8B"
#output_path="../baseline_eval_results_0430/Qwen3-8B/"
#
#mkdir -p $output_path
#
#cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
#
#export CUDA_VISIBLE_DEVICES=6,7
#
#bash evaluate_distilled_qwen.sh $model_path $output_path $summary_path
#
#
#source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
#conda activate cft
#
#summary_path="../baseline_eval_results_0430/Qwen3-14B-Base/summary.txt"
#model_path="/data/yubo/models/Qwen3-14B-Base"
#output_path="../baseline_eval_results_0430/Qwen3-14B-Base/"
#
#mkdir -p $output_path
#
#cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
#
#export CUDA_VISIBLE_DEVICES=6,7
#
#bash evaluate_distilled_qwen.sh $model_path $output_path $summary_path
#

summary_path="../baseline_eval_results_0430/Qwen3-14B/summary.txt"
model_path="/data/yubo/models/Qwen3-14B"
output_path="../baseline_eval_results_0430/Qwen3-14B/"

mkdir -p $output_path

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=6,7

bash evaluate_distilled_qwen.sh $model_path $output_path $summary_path





