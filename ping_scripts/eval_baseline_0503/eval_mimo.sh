
source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

summary_path="../baseline_eval_results_0503/MiMo-7B-Base/summary.txt"
model_path="/data/yubo/models/MiMo-7B-Base"
output_path="../baseline_eval_results_0503/MiMo-7B-Base/"

mkdir -p $output_path

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=4

bash evaluate_qwen3_8_re.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_0503/MiMo-7B-SFT/summary.txt"
model_path="/data/yubo/models/MiMo-7B-SFT"
output_path="../baseline_eval_results_0503/MiMo-7B-SFT/"

mkdir -p $output_path

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=4

bash evaluate_qwen3_8_re.sh $model_path $output_path $summary_path



