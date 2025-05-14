export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export CHECKPOINTS_DIR=./checkpoints/ # your checkpoint path

conda activate rlvr_train
cd ../
bash scripts/train/training_1.5b_pi1_r128.sh


