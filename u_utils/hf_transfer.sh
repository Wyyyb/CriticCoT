
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