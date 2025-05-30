
source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate cft

summary_path="../baseline_eval_results_0430/Phi-4-reasoning/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/Phi-4-reasoning"
output_path="../baseline_eval_results_0430/Phi-4-reasoning/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash evaluate_phi4.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_0430/Phi-4/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/Phi-4"
output_path="../baseline_eval_results_0430/Phi-4/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash evaluate_phi4.sh $model_path $output_path $summary_path

summary_path="../baseline_eval_results_0430/Phi-4-reasoning-plus/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/Phi-4-reasoning-plus"
output_path="../baseline_eval_results_0430/Phi-4-reasoning-plus/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash evaluate_phi4.sh $model_path $output_path $summary_path



