
# source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

export CUDA_VISIBLE_DEVICES=4,5

summary_path="../baseline_eval_results_rl_0516/summary.txt"
model_path="/data/yubo/CriticCoT/one_shot_rlvr/checkpoints/verl_few_shot/Qwen2.5-Math-1.5B-pi1_r128/global_step_120/actor/"
output_path="../baseline_eval_results_rl_0516/Qwen2.5-Math-1.5B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_rl_0516/summary.txt"
model_path="/data/yubo/CriticCoT/one_shot_rlvr/checkpoints/verl_few_shot/Qwen2.5-Math-1.5B-pi1_r128/global_step_100/actor/"
output_path="../baseline_eval_results_rl_0516/Qwen2.5-Math-1.5B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path



summary_path="../baseline_eval_results_rl_0516/summary.txt"
model_path="/data/yubo/CriticCoT/one_shot_rlvr/checkpoints/verl_few_shot/Qwen2.5-Math-1.5B-pi1_r128/global_step_80/actor/"
output_path="../baseline_eval_results_rl_0516/Qwen2.5-Math-1.5B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path

summary_path="../baseline_eval_results_rl_0516/summary.txt"
model_path="/data/yubo/CriticCoT/one_shot_rlvr/checkpoints/verl_few_shot/Qwen2.5-Math-1.5B-pi1_r128/global_step_60/actor/"
output_path="../baseline_eval_results_rl_0516/Qwen2.5-Math-1.5B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_rl_0516/summary.txt"
model_path="/data/yubo/CriticCoT/one_shot_rlvr/checkpoints/verl_few_shot/Qwen2.5-Math-1.5B-pi1_r128/global_step_40/actor/"
output_path="../baseline_eval_results_rl_0516/Qwen2.5-Math-1.5B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path


summary_path="../baseline_eval_results_rl_0516/summary.txt"
model_path="/data/yubo/CriticCoT/one_shot_rlvr/checkpoints/verl_few_shot/Qwen2.5-Math-1.5B-pi1_r128/global_step_20/actor/"
output_path="../baseline_eval_results_rl_0516/Qwen2.5-Math-1.5B/"

cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

bash evaluate_qwen.sh $model_path $output_path $summary_path
