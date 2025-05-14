#!/usr/bin/env bash
set -x
# Base execution environment
# export MODELSCOPE_CACHE='/data/yuansheng/cache'
# export HF_HOME='/data/yuansheng/cache'
# export SWIFT_DEBUG_ARGS=True

# wandb
# if [ -z "$RUN_NAME" ]; then
#     RUN_NAME="python_200K_vis_code_lr5e6"
# fi
# export WANDB_PROJECT="qwen2_5_3b_coder_python_200K"
# export WANDB_NAME=$RUN_NAME

MODEL_PATH="/data/yubo/models/Qwen2.5-Math-7B"

DATA_PATH="/data/yubo/CriticCoT/local_data/training_data_0513/full_one-shot_train_data_filtered_38k_0513.jsonl"

OUTPUT_DIR="/data/yubo/CriticCoT/ms-swift/output_models_0513_filtered_full_38k/"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ../ms-swift

torchrun \
    --nproc_per_node 8 \
    --standalone \
    swift/cli/sft.py\
    --use_hf True \
    \
    --model $MODEL_PATH \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    \
    --dataset $DATA_PATH \
    --split_dataset_ratio 0 \
    --dataset_num_proc 8 \
    --streaming False \
    --strict False \
    --deepspeed zero3 \
    --remove_unused_columns False \
    --dataloader_num_workers 8 \
    \
    --truncation_strategy delete \
    \
    --output_dir $OUTPUT_DIR \
    --gradient_checkpointing True \
    --per_device_train_batch_size 4 \
    --weight_decay 0.05 \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --logging_first_step True \
    --logging_steps 1 \
    \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 4 \
    --save_only_model True \
    --warmup_ratio 0.2 \
    --ddp_backend "nccl" \
    \
    --freeze_llm False \
    --freeze_vit False \
    --freeze_aligner False\


    # --save_strategy "epoch" \
    # --save_strategy "steps" \
    # --save_steps 109 \
    # --save_total_limit 5 \
    # --deepspeed zero3 \
    # --max_steps -1 \
    # --device_map auto \
    # --override_exist_file True \
    # --eval_strategy None \
    # --custom_dataset_in