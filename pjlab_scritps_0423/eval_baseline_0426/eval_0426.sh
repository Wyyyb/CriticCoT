
source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate cft

summary_path="../../baseline_eval_results_0426/qwen_32b_instruct/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/Qwen2.5-32B-Instruct"
output_path="../../baseline_eval_results_0426/qwen_32b_instruct/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=6,7

bash evaluate_qwen_test.sh $model_path $output_path $summary_path


summary_path="../../baseline_eval_results_0426/qwen_32b/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/Qwen2.5-32B"
output_path="../../baseline_eval_results_0426/qwen_32b/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=6,7

bash evaluate_qwen_test.sh $model_path $output_path $summary_path



summary_path="../../baseline_eval_results_0426/qwen_32b_distill/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/DeepSeek-R1-Distill-Qwen-32B"
output_path="../../baseline_eval_results_0426/qwen_32b_distill/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=6,7

bash evaluate_qwen_test.sh $model_path $output_path $summary_path





