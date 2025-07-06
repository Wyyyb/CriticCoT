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

import ray
from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.rollout.async_server import AsyncLLMServerManager
from tests.rollout.async_rollout_utils import init_async_rollout_manager

from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from pathlib import Path
from datasets import load_dataset


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

    worker_groups, async_rollout_manager = init_async_rollout_manager(config)
    async_chat_scheduler = async_rollout_manager.chat_scheduler

    # test sleep and wake_up
    async_rollout_manager.sleep()
    async_rollout_manager.wake_up()

    use_fn_calling = False
    max_steps = 40
    max_steps_absolute = 50
    max_token_limit = 32768
    max_exec_time = 90
    max_total_time = 1200
    max_llm_time = 120
    fout = open("output.txt", "w")
    # =========================== 3. Multi turn rollout  ===========================
    async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
        assert exception is None, f"exception: {exception}"
        messages, round = info["messages"], info["round"]
        message = completions.choices[0].message
        messages.append({"role": message.role, "content": message.content})
        print(f"[round={round}] role: {message.role}, content: {message.content}")

        extra_headers = {"x-request-id": completions.id}
        if round == 0:
            messages.append({"role": "user", "content": "What is your name?"})
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={"messages": messages, "round": 1},
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        elif round == 1:
            messages.append({"role": "user", "content": "What is your favorite color?"})
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={"messages": messages, "round": 2},
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        else:
            print("Done!")

    async def agent_callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
        agent = info["agent"]
        env = info["env"]
        step_count = info["step_count"]
        total_time_traj = info["total_time_traj"]
        done = info["done"]

        # 1. 解析 LLM 输出
        if hasattr(completions, "usage"):
            usage = completions.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
        else:
            completion_tokens = -1
            prompt_tokens = -1
            total_tokens = -1

        assistant_message = completions.choices[0].message.content
        thought, action = agent.parse_response(assistant_message)

        try:
            obs, reward, done_flag, info_env = env.step(action, timeout=max_exec_time)
        except Exception as e:
            obs = str(e)
            done_flag = True
            info_env = {}

        llm_exec_time = getattr(completions, "llm_exec_time", 0)
        env_exec_time = info_env.get("total_time", 0)
        total_step_time = llm_exec_time + env_exec_time
        total_time_traj += total_step_time
        step_count += 1

        assistant_message = f"{thought}\n\n{action.to_xml_string()}"
        agent.history.append({"role": "assistant", "content": assistant_message})
        fout.write(f"ASSISTANT:\n{assistant_message}\n")
        agent.history.append({"role": "user", "content": str(obs)})
        fout.write(f"OBS:\n{obs}\n")

        # 5. trajectory_steps 追加
        trajectory_step = TrajectoryStep(
            step_idx=step_count - 1,
            thought=thought,
            action=action.to_xml_string(),
            observation=str(obs),
            done=done_flag,
            info=info_env,
            token_usage_prompt=prompt_tokens,
            token_usage_completion=completion_tokens,
            token_usage_total=total_tokens,
            llm_exec_time=llm_exec_time,
            env_exec_time=env_exec_time,
            total_step_time=total_step_time,
            total_time_traj=total_time_traj,
            step_count=step_count,
        )
        agent.trajectory_steps.append(trajectory_step)

        # 6. 检查 done/step/token/time 限制
        steps_remaining = max_steps - step_count
        exit_reason = None
        if done_flag:
            if steps_remaining > 0:
                exit_reason = "agent"
            elif steps_remaining == 0:
                exit_reason = "max_step_limit"
            else:
                exit_reason = "agent_max_step_limit"
            done = True
        elif total_tokens >= max_token_limit:
            exit_reason = "token_limit"
            done = True
        elif step_count >= max_steps_absolute:
            exit_reason = "abs_step_limit"
            done = True
        elif total_time_traj >= max_total_time:
            exit_reason = "traj_time_limit"
            done = True

        if not done:
            await async_chat_scheduler.submit_chat_completions(
                callback=agent_callback,
                callback_additional_info={
                    "env": env,
                    "agent": agent,
                    "step_count": step_count,
                    "total_time_traj": total_time_traj,
                    "done": done,
                },
                model=model_name,
                messages=agent.history,
            )
    
    print(f"Loading dataset...")
    ds = load_dataset('/data/minimax-dialogue/users/xiancai/hf_datasets/R2E-Gym/SWE-Bench-Lite')
    split = 'test' # split of the dataset [train, test]
    # load gym environment
    env_index = 100 # index of the environment [0, len(ds)]
    env_args = EnvArgs(ds = ds[split][env_index])
    print("Loading environment...")
    env = RepoEnv(env_args)

    print(f"ENV Done!")

    agent_args = AgentArgs.from_yaml(Path('/data/minimax-dialogue/users/xiancai/work/r2e-gym/src/r2egym/agenthub/config/edit_non_fn_calling.yaml'))
    agent_args.llm_name = model_name
    agent_args.llm_api_key = "sk-x-api-key"
    agent = Agent(name="EditingAgent", args=agent_args)

    # prompt准备
    problem_statement = env.runtime.get_task_instruction()
    system_prompt = agent.system_prompt_template
    user_prompt = agent.instance_prompt_template.format(problem_statement=problem_statement)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    agent.reset()
    env.reset()
    env.add_commands(agent_args.command_files)
    agent.history = messages
    agent.trajectory_steps = []

    async_rollout_manager.submit_chat_completions(
        callback=agent_callback,
        callback_additional_info={
            "env": env,
            "agent": agent,
            "step_count": 0,
            "total_time_traj": 0,
            "done": False,
        },
        model=model_name,
        messages=agent.history,
    )


if __name__ == "__main__":
    test_vllm_multi_turn()
