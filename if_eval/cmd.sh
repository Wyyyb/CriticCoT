cd /data/yubo/CriticCoT/if_eval
export CUDA_VISIBLE_DEVICES=0
python vllm_generate.py --model_path /data/yubo/CriticCoT/models/Qwen2.5-Math-7B-CFT --input_dir /data/yubo/google-research/instruction_following_eval/data/input_data.jsonl

cd /data/yubo/CriticCoT/if_eval
export CUDA_VISIBLE_DEVICES=0
python vllm_generate.py --model_path /data/yubo/CriticCoT/models/Qwen2.5-Math-7B --input_dir /data/yubo/google-research/instruction_following_eval/data/input_data.jsonl


cd /data/yubo/CriticCoT/if_eval
export CUDA_VISIBLE_DEVICES=1
python vllm_generate.py --model_path /data/yubo/CriticCoT/models/Qwen2.5-Math-7B-Instruct --input_dir /data/yubo/google-research/instruction_following_eval/data/input_data.jsonl




cd /data/yubo/google-research
python3 -m instruction_following_eval.evaluation_main \
  --input_data=./instruction_following_eval/data/input_data.jsonl \
  --input_response_data=./instruction_following_eval/data/input_data_Qwen2_5-Math-7B-CFT.jsonl \
  --output_dir=./instruction_following_eval/data/

python3 -m instruction_following_eval.evaluation_main \
  --input_data=./instruction_following_eval/data/input_data.jsonl \
  --input_response_data=./instruction_following_eval/data/input_data_Qwen2_5-Math-7B.jsonl \
  --output_dir=./instruction_following_eval/data/

python3 -m instruction_following_eval.evaluation_main \
  --input_data=./instruction_following_eval/data/input_data.jsonl \
  --input_response_data=./instruction_following_eval/data/input_data_Qwen2_5-Math-7B-Instruct.jsonl \
  --output_dir=./instruction_following_eval/data/


