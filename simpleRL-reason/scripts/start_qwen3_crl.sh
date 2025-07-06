eval "$(/data/minimax-dialogue/feishan/miniconda3/bin/conda shell.bash hook)"

conda activate yb_verl

unset proxy https_proxy http_proxy ftp_proxy no_proxy

cd /data/minimax-dialogue/feishan/CriticCoT/simpleRL-reason

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

echo "ray started"


bash train_grpo_math_tune_ray_qwen3_4b_crl.sh \
  --model_name Qwen3-4B-Base \
  --max_response_length 4096 \
  --train_batch_size 1024 \
  --rollout_n 8 \
  --kl_loss_coef 0.0001 \
  --entropy_coeffient 0.001 \
  --rollout_gpu_memory_util 0.75 \
  --rollout_tp 2 \
  --save_freq 5