cd ..
bash train_grpo_math_tune_ray.sh \
  --model_name  \
  --max_response_length 4096 \
  --train_batch_size 1024 \
  --rollout_n 8 \
  --kl_loss_coef 0.0001 \
  --entropy_coeffient 0.001 \
  --rollout_gpu_memory_util 0.75 \
  --rollout_tp 2 \
  --save_freq 5