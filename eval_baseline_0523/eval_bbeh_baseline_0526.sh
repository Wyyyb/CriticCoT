
summary_path="../../baseline_eval_results_0523/qwen3_4b/summary.txt"
model_path="/data/yubowang/models/Qwen3-4B"
output_path="../../baseline_eval_results_0523/qwen3_4b/"

cd /data/yubowang/CriticCoT/bbeh/bbeh

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash eval_qwen3.sh $model_path $output_path $summary_path


summary_path="../../baseline_eval_results_0523/Qwen2.5-Math-7B/summary.txt"
model_path="/data/yubowang/models/Qwen2.5-Math-7B"
output_path="../../baseline_eval_results_0523/Qwen2.5-Math-7B/"

cd /data/yubowang/CriticCoT/bbeh/bbeh

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash eval_qwen2-5.sh $model_path $output_path $summary_path


summary_path="../../baseline_eval_results_0523/Qwen2.5-7B/summary.txt"
model_path="/data/yubowang/models/Qwen2.5-7B"
output_path="../../baseline_eval_results_0523/Qwen2.5-7B/"

cd /data/yubowang/CriticCoT/bbeh/bbeh

mkdir -p $output_path


bash eval_qwen2-5.sh $model_path $output_path $summary_path



summary_path="../../baseline_eval_results_0523/Qwen2.5-14B/summary.txt"
model_path="/data/yubowang/models/Qwen2.5-14B"
output_path="../../baseline_eval_results_0523/Qwen2.5-14B/"

cd /data/yubowang/CriticCoT/bbeh/bbeh

mkdir -p $output_path


bash eval_qwen2-5.sh $model_path $output_path $summary_path


summary_path="../../baseline_eval_results_0523/Qwen2.5-14B-Instruct/summary.txt"
model_path="/data/yubowang/models/Qwen2.5-14B-Instruct"
output_path="../../baseline_eval_results_0523/Qwen2.5-14B-Instruct/"

cd /data/yubowang/CriticCoT/bbeh/bbeh

mkdir -p $output_path

bash eval_qwen2-5.sh $model_path $output_path $summary_path


summary_path="../../baseline_eval_results_0523/Qwen3-4B-Base/summary.txt"
model_path="/data/yubowang/models/Qwen3-4B-Base"
output_path="../../baseline_eval_results_0523/Qwen3-4B-Base/"

cd /data/yubowang/CriticCoT/bbeh/bbeh

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash eval_qwen3.sh $model_path $output_path $summary_path


summary_path="../../baseline_eval_results_0523/Qwen3-8B-Base/summary.txt"
model_path="/data/yubowang/models/Qwen3-8B-Base"
output_path="../../baseline_eval_results_0523/Qwen3-8B-Base/"

cd /data/yubowang/CriticCoT/bbeh/bbeh

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash eval_qwen3.sh $model_path $output_path $summary_path



