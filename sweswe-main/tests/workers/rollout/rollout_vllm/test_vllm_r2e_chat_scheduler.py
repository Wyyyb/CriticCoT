# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from typing import Any, Tuple

import numpy as np
import pytest
import ray
from omegaconf import DictConfig, OmegaConf
from transformers.utils import get_json_schema

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.utils import hf_tokenizer

from datasets import load_dataset
import pandas as pd

from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
agent_args = AgentArgs.from_yaml(Path('/root/code/verl/r2e-gym/src/r2egym/agenthub/config/edit_non_fn_calling.yaml'))
job_name_path = "/data/minimax-dialogue/users/xiancai/verl/r2e-gym/src/r2egym/agenthub/runtime/job_name.txt"

def load_jsonl(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def init_r2e_env(batch: DataProto, n: int):
    """
    Initialize the environment for the rollout worker.
    """
    batch.non_tensor_batch["envs"] = np.array([[None] * n for _ in range(len(batch.non_tensor_batch["raw_dataset"]))])
    all_ds = batch.non_tensor_batch["raw_dataset"].repeat(n, axis=0)
    with open(job_name_path, "w") as f:
        pass

    with ThreadPoolExecutor(max_workers=len(all_ds)) as executor:
        futures = [executor.submit(_init_r2e_env_single, index, ds) for index, ds in enumerate(all_ds)]
        for future in tqdm(as_completed(futures), desc="Initializing environments"):
            try:
                env, index = future.result()
                batch.non_tensor_batch["envs"][index // n][index % n] = env
            except Exception as e:
                print(f"Error initializing environment: {e}")

def _init_r2e_env_single(index: int, dumped_dataset: str):
    try:
        ds = json.loads(dumped_dataset)
        env_args = EnvArgs(ds=ds)
        env = RepoEnv(env_args)
        with open(job_name_path, "a") as f:
            f.write(env.runtime.job_name + '\n')
        env.add_commands(agent_args.command_files)
        return env, index
    except Exception as e:
        print(f"Failed env {index} : {e}")
        return None, index
    

def _close_r2e_env_single(env: RepoEnv):
    try:
        env.close()
    except Exception as e:
        print(f"Error closing environment: {e}")


def _close_r2e_envs(batch: DataProto):
    all_envs = batch.non_tensor_batch["envs"].flatten()

    with ThreadPoolExecutor(max_workers=len(all_envs)) as executor:
        futures = [executor.submit(_close_r2e_env_single, env) for env in all_envs]
        for future in tqdm(as_completed(futures), desc="Closing environments"):
            try:
                future.result()
            except Exception as e:
                print(f"Error closing environment: {e}")
    

def test_init_r2e_env():
    dataset = load_dataset("/data/minimax-dialogue/users/xiancai/hf_datasets/R2E-Gym/R2E-Gym-Lite", split="train")
    _dataset = [json.dumps(dataset[i]) for i in range(64)]
    rollout_n = 16
    batch = DataProto(
        non_tensor_batch={
            "raw_dataset": np.array(_dataset),
            "envs": np.array([[None] * rollout_n for _ in range(len(_dataset))]),
        },
    )
    init_r2e_env(batch, rollout_n)
    print(batch.non_tensor_batch["envs"])
    # close all envs
    _close_r2e_envs(batch)


from transformers import AutoTokenizer
from typing import List, Dict
import torch
from tensordict import TensorDict

tokenizer = AutoTokenizer.from_pretrained("/data/minimax-dialogue/users/xiancai/hf_models/R2EGym-32B-Agent")

def postprocess(raw_prompts: List[List[Dict[str, str]]], conversations: List[List[Dict[str, str]]]):
    # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
    # prompts: left pad
    # responses: right pad
    # input_ids: prompt + response
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

    # prompts: prompt from input dataset
    prompts = [tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in raw_prompts]
    
    # sequences: prompt + response
    sequences = [tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False) for conversation in conversations]

    # responses: [response
    responses = [sequence[len(prompts[i]) :] for i, sequence in enumerate(sequences)]

    print(prompts)
    print(responses)

    prompts = tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
    responses = tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")

    print(prompts)
    print(responses)

    # response_mask: response mask with tools calling masked out
    response_mask = _mask_out_env_obs_tokens(raw_prompts, conversations, responses["input_ids"], responses["attention_mask"])
    print(response_mask)

    input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
    attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

    batch = TensorDict(
        {
            "prompts": prompts["input_ids"],  # [bsz, prompt_length]
            "responses": responses["input_ids"],  # [bsz, response_length]
            "response_mask": response_mask,  # [bsz, response_length]
            "input_ids": input_ids,  # [bsz, prompt_length + response_length]
            "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
            "position_ids": position_ids,  # [bsz, prompt_length + response_length]
        },
        batch_size=len(input_ids),
    )

    num_turns = np.array([len(conversation) for conversation in conversations], dtype=np.int32)
    return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns})


def _mask_out_env_obs_tokens(
    raw_prompts: List[List[Dict[str, str]]],
    conversations: List[List[Dict[str, str]]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask out tools calling tokens in the responses.

    Args:
        raw_prompts: [prompt] from input dataset
        conversations: [prompt + response]
        input_ids: responses tokens
        attention_mask: responses attention mask

    Returns:
        mask: (batch_size, response_length)
    """
    batch_size = input_ids.size(0)
    assert len(raw_prompts) == batch_size, f"{len(raw_prompts)} != {batch_size}"
    assert len(conversations) == batch_size, f"{len(conversations)} != {batch_size}"

    loss_mask = attention_mask.clone()
    for i in range(batch_size):
        responses = conversations[i][len(raw_prompts[i]) :]
        assert len(responses) > 0, f"responses is empty: {responses}"

        roles = [response["role"] for response in responses]
        eos_indices = input_ids[i].eq(tokenizer.eos_token_id).nonzero().squeeze(1)[: len(roles)]
        for j in range(len(roles)):
            if roles[j] == "user":
                bos = eos_indices[j - 1] + 2 if j > 0 else 0
                eos = eos_indices[j] + 4
                loss_mask[i, bos : eos + 1] = 0

    return loss_mask


def test_postprocess():
    prompts = [
        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ], 
        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the capital of China?"
            }
        ]
    ]
    conversations = [
        prompt + [
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is the capital of China?"},
            {"role": "assistant", "content": "The capital of China is Beijing."}
        ]
        for prompt in prompts
    ]
    print(postprocess(prompts, conversations))
            

def init_config() -> DictConfig:
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "/data/minimax-dialogue/users/xiancai/hf_models/R2EGym-32B-Agent"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
    config.actor_rollout_ref.rollout.prompt_length = 30768
    config.actor_rollout_ref.rollout.response_length = 2000
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.8
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 8
    config.actor_rollout_ref.rollout.max_model_len = 32768
    config.actor_rollout_ref.rollout.max_num_batched_tokens = 32768

    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.rollout.n=1
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    return config


def test_vllm_async_rollout_without_tool_calls(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # =========================== 1. Init rollout manager ===========================
    async_rollout_manager = init_async_rollout_manager(init_config)

    # test sleep and wake_up
    async_rollout_manager.sleep()
    async_rollout_manager.wake_up()

    # =========================== 2. Generate sequences  ===========================
    # load data from parquet file
    _dataset = pd.read_parquet("/data/minimax-dialogue/users/xiancai/verl/data/r2e/0623/test.parquet")
    _dataset = _dataset.iloc[:128]  # 修正：使用iloc来获取前32行，保持DataFrame结构
    rollout_n = init_config.actor_rollout_ref.rollout.n
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(_dataset["prompt"]),
            "raw_dataset": np.array(_dataset["raw_dataset"]),
        },
    )
    # init_r2e_env(batch, rollout_n)
    # print("Initalize environments successfully!")
    result = async_rollout_manager.generate_sequences(prompts=batch)
    # _close_r2e_envs(batch)
    # print("Close environments successfully!")

    # check result
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert len(result) == len(_dataset) * rollout_n
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len

    # check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    rewards = result.non_tensor_batch["__rewards__"]
    # assert np.all(num_turns == 2)
    print(num_turns)
    print(rewards)
    print(sum(rewards))

    print("Test passed!")
    ray.shutdown()


if __name__ == "__main__":
    # test_vllm_async_rollout_without_tool_calls(init_config())
    # test_init_r2e_env()
    test_postprocess()
    # 100 % 32 = 4
    # 32 - 4 = 28

