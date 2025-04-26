
source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate cft

summary_path="/mnt/hwfile/opendatalab/yubo/CriticCoT/0425_eval_results_32b_cft_deepmath/summary.txt"
model_dir="/mnt/hwfile/opendatalab/yubo/CriticCoT/360-LLaMA-Factory-sp/output_models_0425/deepmath_qwen_32b_distill_cft_0424"
ckpt_start=0
ckpt_end=150

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/multi_eval_0425

bash start_8_cards.sh $summary_path $model_dir $ckpt_start $ckpt_end

# bash eval_7b.sh $summary_path $model_dir $ckpt_start $ckpt_end
