
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



