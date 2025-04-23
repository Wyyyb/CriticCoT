
source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate cft

summary_path="../../eval_results_0423/qwen_32b_aime25/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/Qwen2.5-32B"
output_path="../../eval_results_0423/qwen_32b_aime25/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash evaluate_qwen_test.sh $model_path $output_path $summary_path

