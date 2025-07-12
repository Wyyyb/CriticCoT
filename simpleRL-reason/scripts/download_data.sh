export proxy="http://pac-internal.xaminim.com:3129" && export https_proxy=$proxy && export http_proxy=$proxy && export ftp_proxy=$proxy && export no_proxy="localhost,[127.0.0.1](http://127.0.0.1/),*.xaminim.com,10.0.0.0/8"

cd /data/minimax-dialogue/feishan/CriticCoT

mkdir -p verl_data

cd verl_data

mkdir training_data

cd training_data

mkdir deepscaler_critique_formatted

mkdir deepscaler_train_filter

cd deepscaler_critique_formatted

wget https://huggingface.co/datasets/ubowang/critique_rl/resolve/main/critique_format_train.parquet

mv critique_format_train.parquet train.parquet

cp train.parquet test.parquet

cd ../deepscaler_train_filter

wget https://huggingface.co/datasets/ubowang/critique_rl/resolve/main/ori_format_train.parquet

mv ori_format_train.parquet train.parquet

cp train.parquet test.parquet

mkdir -p /data/minimax-dialogue/feishan/models

huggingface-cli download --repo-type model Qwen/Qwen3-4B-Base --local-dir /data/minimax-dialogue/feishan/models/Qwen3-4B-Base

huggingface-cli download --repo-type model Qwen/Qwen2.5-Math-7B --local-dir /data/minimax-dialogue/feishan/models/Qwen2.5-Math-7B

cd /data/minimax-dialogue/feishan/critique_rl

#mkdir -p verl_cft_data_bk
#
#mv critique_verl/cft_data verl_cft_data_bk/
#
#mv critique_verl/data verl_cft_data_bk/








