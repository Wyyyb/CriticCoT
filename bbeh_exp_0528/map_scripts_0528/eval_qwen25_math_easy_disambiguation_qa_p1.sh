
cd /map-vepfs/yubo/CriticCoT/bbeh_exp_0528/map_scripts_0528

bash train_qwen_14b_easy_disambiguation_qa_p0_0529.sh

source /map-vepfs/miniconda3/bin/activate
conda activate yubo_eval

base_dir="/map-vepfs/yubo/CriticCoT/ms-swift/output_models_0528_bbeh_qwen_2-5_math_7b_easy_disambiguation_qa_p1_attn/v1-20250529-065550"
output_base_dir="../../eval_results_0528_easy_disambiguation_qa_p1"
model_name="Qwen2.5-Math-7B"
# task_list="bbeh_causal_understanding,bbeh_disambiguation_qa,bbeh_boolean_expressions,bbeh_time_arithmetic,bbeh_buggy_tables,bbeh_object_counting,bbeh_zz_mini"
task_list="bbeh_disambiguation_qa,bbeh_zz_mini"

cd /map-vepfs/yubo/CriticCoT/bbeh/bbeh

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 循环处理checkpoint-2到checkpoint-50，步长为2
for i in $(seq 5 5 50); do
  checkpoint="checkpoint-$i"
  model_path="$base_dir/$checkpoint"

  # 创建对应的输出目录
  output_path="$output_base_dir/$model_name/$checkpoint"
  summary_path="$output_base_dir/$model_name/summary.txt"

  # 确保输出目录存在
  mkdir -p $output_path

  echo "==========================================================="
  echo "Evaluating $checkpoint"
  echo "Model path: $model_path"
  echo "Output path: $output_path"
  echo "Summary path: $summary_path"
  echo "==========================================================="

  # 运行评估脚本
  bash eval_qwen2-5_task.sh $model_path $output_path $summary_path $task_list

  # 可选：添加间隔时间，避免资源冲突
  sleep 5
done
