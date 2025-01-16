#!/bin/bash

# 定义变量
REMOTE_HOST="xcs-research-share-4fz5c-master-0.ou-600a79a43b2e47a07dfcc2c984743ee8.pytorch.bash.dev.pod@sshproxy.dh3.ai"
REMOTE_PORT="2222"
REMOTE_FILE="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/CriticCoT_t2_critic_data_0115.json"
LOCAL_DIR="/Users/MyDisk/2025_git/01_data_model_bk_0117"  # 本地保存目录

# 创建本地目录(如果不存在)
mkdir -p "$LOCAL_DIR"

# 使用scp下载文件
scp -P $REMOTE_PORT \
    "$REMOTE_HOST:$REMOTE_FILE" \
    "$LOCAL_DIR/"