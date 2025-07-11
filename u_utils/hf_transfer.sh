
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

huggingface-cli upload ubowang/critique-32b-instruct_new_61_6-ckpt-3 . --repo-type model

huggingface-cli upload ubowang/critique-32b . --repo-type model

huggingface-cli upload ubowang/webinstruct-batch-data-gpt4o-mini . --repo-type dataset

huggingface-cli upload ubowang/CFT-Webinstruct-0121-ckpt . --repo-type model

huggingface-cli upload ubowang/self_critic_data_0114 . --repo-type dataset

huggingface-cli upload ubowang/qwen2.5-7b-math-cft-gpt-4o-0128 . --repo-type model

huggingface-cli upload TigerLab/WebInstruct-CFT-50k . --repo-type dataset --private

huggingface-cli upload ubowang/WebInstruct-CFT-50k . --repo-type dataset

huggingface-cli upload TIGER-Lab/Qwen2.5-Math-7B-CFT . --repo-type model

huggingface-cli upload TIGER-Lab/Qwen2.5-32B-Instruct-CFT . --repo-type model

huggingface-cli upload ubowang/webinstruct_cft_40k_o1_mini_brief . --repo-type dataset

huggingface-cli upload ubowang/opc-sft-stage2_data_0203 . --repo-type dataset

huggingface-cli upload ubowang/webinstruct_cft_80k_o1_mini_long_0204 . --repo-type dataset

huggingface-cli upload ubowang/arxiv-llm_qwen_extract_titles_0223 . --repo-type dataset

huggingface-cli upload ubowang/scholarcopilot_re_eval_data_0226 . --repo-type dataset

huggingface-cli upload ubowang/sc_generated_data . --repo-type dataset

huggingface-cli upload ubowang/arxiv_plain_latex_data_1028 . --repo-type dataset

huggingface-cli upload ubowang/arxiv-src-latex . --repo-type dataset

huggingface-cli upload TIGER-Lab/ScholarCopilot-Data-v1 . --repo-type dataset

huggingface-cli download ubowang/webinstruct_cft_80k_o1_mini_long_0204 . --repo-type dataset

huggingface-cli upload ubowang/deepmath_integrate_data_0421 deepmath_integrate_data_0421.json --repo-type dataset

huggingface-cli download ubowang/deepmath_integrate_data_0421 --local-dir . --repo-type dataset

huggingface-cli upload ubowang/deepmath_integrate_data_0421 deepmath_integrate_data_0428_add_solution.json --repo-type dataset

huggingface-cli upload ubowang/webinstruct_verified_0509 webinstruct_v_qwen3_32b_sft_data_70k.json --repo-type dataset

huggingface-cli upload ubowang/webinstruct_verified_0509 webinstruct_data_add_solution_0506_ids_merged.json --repo-type dataset

huggingface-cli upload ubowang/one-shot_data_0512 seed_questions_add_solution_0512.json --repo-type dataset

huggingface-cli upload ubowang/one-shot_data_0512 merged_critique_data_50k_0513.jsonl --repo-type dataset

huggingface-cli upload ubowang/one-shot_data_0512 filtered_critique_data_0513.json --repo-type dataset

huggingface-cli upload ubowang/one-shot_data_0512 one-shot_train_data_filtered_think_0514.jsonl --repo-type dataset

huggingface-cli upload ubowang/1-shot_cft_qwen3_4b_base . --repo-type model

huggingface-cli upload ubowang/1-shot_cft_qwen2-5_math_7b . --repo-type model

huggingface-cli upload ubowang/1-shot_cft_qwen2-5_math_1-5b_ckpt92_0-328 . --repo-type model

huggingface-cli upload ubowang/1-shot_cft_llama3-2_3b_instruct . --repo-type model

huggingface-cli upload ubowang/1-shot_cft_qwen2-5_14b_ckpt60_0-36 . --repo-type model

huggingface-cli upload ubowang/arxiv-llm-0520_bk arxiv_plain_latex_data_1028.tar.gz --repo-type dataset

huggingface-cli upload ubowang/arxiv-llm-0520_bk ./ --repo-type dataset

huggingface-cli upload ubowang/bbeh_qwen25_math_7b_1-shot_cft_causal_understanding_ckpt40 . --repo-type model

huggingface-cli upload ubowang/bbeh_qwen25_math_7b_1-shot_cft_disambiguation_qa_ckpt30 . --repo-type model

huggingface-cli upload ubowang/bbeh_qwen25_math_7b_1-shot_cft_time_arithmetic_ckpt30 . --repo-type model

huggingface-cli upload ubowang/1-shot-cft-data ./ --repo-type dataset

huggingface-cli upload ubowang/qwen3_4b_cft_ckpt40 . --repo-type model

huggingface-cli upload ubowang/critique_rl deepscaler_critique_formatted.json --repo-type dataset

huggingface-cli upload ubowang/critique_rl deepscaler_train_filter.json --repo-type dataset

huggingface-cli upload ubowang/critique_rl deepscaler_critique_formatted.jsonl --repo-type dataset

huggingface-cli upload ubowang/critique_rl deepscaler_train_filter.jsonl --repo-type dataset

huggingface-cli upload ubowang/critique_rl train.parquet --repo-type dataset

huggingface-cli upload ubowang/critique_rl deepscaler_train_filter.jsonl --repo-type dataset

