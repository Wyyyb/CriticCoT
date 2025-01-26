
huggingface-cli upload ubowang/CriticCoT_data_1228 . --repo-type dataset

huggingface-cli download ubowang/CriticCoT_data_1228 --local-dir . --repo-type dataset

huggingface-cli download TIGER-Lab/WebInstruct-QwQ-462K --local-dir . --repo-type dataset


huggingface-cli upload ubowang/CriticCoT_data_1231 . --repo-type dataset

huggingface-cli download ubowang/CriticCoT_data_1231 --local-dir . --repo-type dataset

huggingface-cli download ubowang/critic_training_data_0111 --local-dir . --repo-type dataset

huggingface-cli download meta-math/MetaMathQA --local-dir . --repo-type dataset

huggingface-cli upload ubowang/critic_training_data_0111 . --repo-type dataset

huggingface-cli upload ubowang/CFT_data_0118 . --repo-type dataset

huggingface-cli download ubowang/CFT_data_0118 . --repo-type dataset

huggingface-cli upload ubowang/MAmmoTH-Critique-1 . --repo-type model

pip install modelscope

modelscope upload ubowang/MAmmoTH-Critique-1 . --repo-type model

modelscope upload ubowang/CFT_data_0118 . --repo-type dataset

modelscope download ubowang/CFT_data_0118 . --repo-type dataset

modelscope download --dataset 'ubowang/CFT_data_0118' --local_dir '.'

huggingface-cli download ubowang/on_policy_data_0119 . --repo-type dataset

huggingface-cli upload ubowang/webinstruct_sft_gpt4o_80k_0119_data . --repo-type dataset

huggingface-cli download ubowang/webinstruct_sft_gpt4o_80k_0119_data . --repo-type dataset

huggingface-cli upload ubowang/qwen_math_numina_80k_add_critique_0119 . --repo-type dataset

huggingface-cli upload ubowang/ace_80k_add_critique_0120 . --repo-type dataset

huggingface-cli download ubowang/ace_80k_add_critique_0120 . --repo-type dataset

modelscope download Qwen/Qwen2.5-Math-7B --repo-type model --local_dir /cpfs/data/user/yubowang/models/Qwen2.5-Math-7B

modelscope download --repo-type model deepseek-ai/deepseek-math-7b-base --local_dir /cpfs/data/user/yubowang/models/deepseek-math-7b-base

modelscope download --repo-type model Qwen/Qwen2.5-7B --local_dir /cpfs/data/user/yubowang/models/Qwen2.5-7B

huggingface-cli upload ubowang/cft_data_0120 . --repo-type dataset

huggingface-cli upload ubowang/ace_gpt-4o-1120_cft_data_1123 . --repo-type dataset

huggingface-cli download ubowang/ace_gpt-4o-1120_cft_data_1123 . --repo-type dataset

huggingface-cli upload ubowang/MetaMathQA_sft_80k_data_0118 . --repo-type dataset

huggingface-cli upload ubowang/MetaMath_batch_critique_data_0124 . --repo-type dataset

# /gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/deepseek_math_output_models/CriticCoT_critic_data_0114/checkpoint-70
huggingface-cli upload ubowang/CFT-webinstruct-dpsk-ckpt-0114 . --repo-type model

# /gpfs/public/research/xy/yubowang/CriticCoT/math_eval_result_0117_dense_qwen/qwen2.5-7B_t2_critic_0117-checkpoint-100
huggingface-cli upload ubowang/CFT-webinstruct-qwen_2_5-ckpt-0117 . --repo-type model

huggingface-cli upload ubowang/numina_batch_critique_data_0124 . --repo-type dataset

huggingface-cli upload ubowang/eval_result_01_bk . --repo-type dataset

huggingface-cli upload ubowang/ali_local_batch_data . --repo-type dataset

huggingface-cli upload ubowang/ali_data_0127_bk . --repo-type dataset

huggingface-cli upload ubowang/critique-32b-instruct . --repo-type model

huggingface-cli upload ubowang/critique-32b . --repo-type model

