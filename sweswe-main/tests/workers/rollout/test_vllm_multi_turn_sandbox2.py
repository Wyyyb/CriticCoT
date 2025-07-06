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

import asyncio
import os
from typing import Any, Dict
import copy
import json
import numpy as np

import ray
from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.rollout.async_server import AsyncLLMServerManager

from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from pathlib import Path
from datasets import load_dataset
from tests.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto


def test_vllm_multi_turn():
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "/data/minimax-dialogue/users/xiancai/hf_models/R2EGym-32B-Agent"
    model_name = "/".join(model_path.split("/")[-2:])
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.r2e_chat_scheduler.R2EChatCompletionScheduler"
    config.actor_rollout_ref.rollout.prompt_length = 31768
    config.actor_rollout_ref.rollout.response_length = 1000

    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    # =========================== 1. Create hybrid ActorRollout workers ===========================
    # make openai client happy
    # os.environ["no_proxy"] = ""
    # os.environ["http_proxy"] = ""
    # os.environ["https_proxy"] = ""

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )

    async_rollout_manager = init_async_rollout_manager(config)
    dataset = load_dataset("/data/minimax-dialogue/users/xiancai/hf_datasets/R2E-Gym/R2E-Gym-Lite", split="train")
    _dataset = [dataset[i] for i in range(50)]
    prompts = DataProto(
        non_tensor_batch={
            "raw_dataset": np.array(_dataset)
        }
    )
    print(len(_dataset))
    result = async_rollout_manager.generate_sequences(prompts=prompts)
    assert len(result) == len(_dataset)


if __name__ == "__main__":
    test_vllm_multi_turn()
