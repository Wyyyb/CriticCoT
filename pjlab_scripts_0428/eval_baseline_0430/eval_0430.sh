
source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate cft

summary_path="../baseline_eval_results_0430/MiMo-7B-SFT/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/MiMo-7B-SFT"
output_path="../baseline_eval_results_0430/MiMo-7B-SFT/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash evaluate_distilled_qwen.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_0430/MiMo-7B-Base/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/MiMo-7B-Base"
output_path="../baseline_eval_results_0430/MiMo-7B-Base/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash evaluate_distilled_qwen.sh $model_path $output_path $summary_path



