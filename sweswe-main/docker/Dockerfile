FROM harbor.xaminim.com/minimax-dialogue/verl_xiancai:v0.5

ENV http_proxy=http://pac-internal.xaminim.com:3129
ENV https_proxy=http://pac-internal.xaminim.com:3129
ENV no_proxy=localhost,127.0.0.1,algeng.xaminim.com,10.0.0.0/8

# RUN pip install wandb
# RUN pip install -U vllm tensordict
# RUN pip install antlr4-python3-runtime==4.9.3 sympy math-verify word2number fastapi uvicorn

WORKDIR /root/code

RUN rm -rf /root/code/verl
COPY . /root/code/verl

RUN cd verl/r2e-gym && pip install -e .