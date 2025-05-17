# source /data/yubo/miniconda3/etc/profile.d/conda.sh
conda activate cft

export CUDA_VISIBLE_DEVICES=0,1,2,3

summary_path="../baseline_eval_results_rl_0516/summary.txt"
base_model_path="/data/yubo/CriticCoT/one_shot_rlvr/checkpoints/verl_few_shot/Qwen2.5-Math-1.5B-pi1_r128"
scripts_dir="/data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts"

# 创建输出目录
mkdir -p $output_path

# 遍历不同的checkpoint数量
for ckpt_num in 120 100 80 60 40 20; do
    echo "Evaluating checkpoint $ckpt_num"
    model_path="${base_model_path}/global_step_${ckpt_num}/actor/"
    output_path="../baseline_eval_results_rl_0516/Qwen2.5-Math-1.5B_${ckpt_num}/"
    
    cd $scripts_dir
    bash evaluate_qwen.sh $model_path $output_path $summary_path
done