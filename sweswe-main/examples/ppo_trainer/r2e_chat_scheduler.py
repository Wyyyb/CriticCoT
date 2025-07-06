import os
import re
import copy
import yaml
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import Agent, AgentArgs

logger = get_logger(__name__)  # Logger for this module

import asyncio

import torch
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler

class R2EChatCompletionScheduler(ChatCompletionScheduler):
    """
    A very naive implementation of ChatCompletionScheduler for demo purpose,
    only do single-turn chat completion.
    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
        use_fn_calling: bool = False,
        # step limits TODO: maybe add these limits in the agent args
        max_steps: int = 40,
        max_steps_absolute: int = 50,
        # token limits
        max_token_limit: int = 32768,  # 32k tokens
        # time limits
        max_exec_time: int = 90,  # 5 mins per env execution
        max_total_time: int = 1200,  # 20 minutes overall agent run limit
        max_llm_time: int = 120,  # 2 mins per LLM timeout (note this is per query exlcuding retries | not enforcing hard limit since llm might hit rate limits etc)
        # temperature
        temperature=0,
    ):
        super().__init__(config, model_path, server_addresses, max_cache_size)
        self.use_fn_calling = use_fn_calling
        self.max_steps = max_steps
        self.max_steps_absolute = max_steps_absolute
        self.max_token_limit = max_token_limit
        self.max_total_time = max_total_time
        self.max_exec_time = max_exec_time
        self.max_llm_time = max_llm_time

    def parse_response(self, response_text: str) -> Tuple[str, Action]:
        """
        Extracts:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<function=` up to the first `</function>`
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action = match.group(1)  # The entire <function=...></function> block
            thought = response_text[: match.start()]  # Everything before the block
        else:
            # If no match, treat entire text as "thought"
            thought = response_text
            action = ""

        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action = Action.from_string(action)

        return thought, action

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[R2EChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def agent_callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            env = info["env"]
            step_count = info["step_count"]
            total_time_traj = info["total_time_traj"]
            done = info["done"]
            batch_conversations = info["batch_conversations"]
            batch_index = info["batch_index"]
            f = open(f"verl/output_example/output_{batch_index}.txt", "a")

            if exception is not None:
                f.write(f"exception: {exception}\n")
                f.write(env.runtime.job_name + "\n")
                batch_conversations[batch_index] = batch_conversations[batch_index][:-2]
                reward, test_output = env.runtime._calculate_reward(get_test_output=True)
                rewards[batch_index] = reward
                # Close the environment and runtime
                env.close()
                return

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
            thought, action = self.parse_response(assistant_message)

            try:
                obs, reward, done_flag, info_env = env.step(action, timeout=self.max_exec_time)
            except Exception as e:
                obs = str(e)
                done_flag = True
                info_env = {}

            # 3. 统计时间
            llm_exec_time = getattr(completions, "llm_exec_time", 0)
            env_exec_time = info_env.get("total_time", 0)
            total_step_time = llm_exec_time + env_exec_time
            total_time_traj += total_step_time
            step_count += 1

            assistant_message = f"{thought}\n\n{action.to_xml_string()}"
            batch_conversations[batch_index].append({"role": "assistant", "content": assistant_message})
            batch_conversations[batch_index].append({"role": "user", "content": str(obs)})
            f.write(f"[STEP {step_count}]\n")
            f.write(f"[ASSISTANT]\n{assistant_message}\n")
            f.write(f"[USER]\n{str(obs)}\n\n")
            f.flush()

            steps_remaining = self.max_steps - step_count
            exit_reason = None
            if done_flag:
                if steps_remaining > 0:
                    exit_reason = "agent"
                elif steps_remaining == 0:
                    exit_reason = "max_step_limit"
                else:
                    exit_reason = "agent_max_step_limit"
                done = True
            elif total_tokens >= self.max_token_limit:
                exit_reason = "token_limit"
                done = True
            elif step_count >= self.max_steps_absolute:
                exit_reason = "abs_step_limit"
                done = True
            elif total_time_traj >= self.max_total_time:
                exit_reason = "traj_time_limit"
                done = True

            # 7. 递归 or 终止
            if done:
                # also get the gt outputs
                reward, test_output = env.runtime._calculate_reward(get_test_output=True)
                rewards[batch_index] = reward
                # Close the environment and runtime
                env.close()
                return
            else:
                await self.submit_chat_completions(
                    callback=agent_callback,
                    callback_additional_info={
                        "env": env,
                        "step_count": step_count,
                        "total_time_traj": total_time_traj,
                        "done": done,
                        "batch_conversations": batch_conversations,
                        "batch_index": batch_index,
                    },
                    model=self.model_name,
                    messages=batch_conversations[batch_index],
                    **kwargs,
                )

        # 启动每个 batch item 的异步任务
        tasks, batch_conversations, prompts, rewards = [], [None] * len(batch), [None] * len(batch), [0] * len(batch)
        agent_args = AgentArgs.from_yaml(Path('/root/code/verl/r2e-gym/src/r2egym/agenthub/config/edit_non_fn_calling.yaml'))
        agent_args.llm_name = self.model_name
        agent_args.llm_api_key = "EMPTY"

        async def setup_and_run(batch_index, ds):
            # set env for agent
            ds = json.loads(ds)
            env_args = EnvArgs(ds=ds)
            # env = RepoEnv(env_args)
            env = await asyncio.to_thread(RepoEnv, env_args)
            # agent = await asyncio.to_thread(Agent, name="EditingAgent", args=agent_args)

            # prompt
            problem_statement = env.runtime.get_task_instruction()
            system_prompt = agent_args.system_prompt
            user_prompt = agent_args.instance_prompt.format(
                problem_statement=problem_statement
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompts[batch_index] = messages
            # env.reset()
            env.add_commands(agent_args.command_files)
            batch_conversations[batch_index] = messages

            # print(f"start {batch_index} at {time.time()}")
            # await asyncio.sleep(2)
            # print(f"end {batch_index} at {time.time()}")

            await self.submit_chat_completions(
                callback=agent_callback,
                callback_additional_info={
                    "env": env,
                    "step_count": 0,
                    "total_time_traj": 0,
                    "done": False,
                    "batch_conversations": batch_conversations,
                    "rewards": rewards,
                    "batch_index": batch_index,
                },
                model=self.model_name,
                messages=batch_conversations[batch_index],
                **kwargs,
            )
            del env

        for batch_index, ds in enumerate(batch.non_tensor_batch["raw_dataset"]):
            print(os.getcwd())
            with open(f"verl/output_example/output_{batch_index}.txt", "w") as f:
                pass
            tasks.append(asyncio.create_task(setup_and_run(batch_index, ds)))

        await asyncio.gather(*tasks)
        print("[R2EChatCompletionScheduler] generate_sequences done")

        return self._postprocess(batch, batch_conversations, prompts, rewards, kwargs["n"])

    def _postprocess(
        self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], prompts: List[List[Dict[str, str]]], rewards: List[float], n: int
    ) -> DataProto:
        print(f"[R2EChatCompletionScheduler] _postprocess rewards: {rewards}")
        non_tensor_batch = batch.non_tensor_batch
        non_tensor_batch["r2e_rewards"] = np.array(rewards)
        # prompts: [prompt] from input dataset
        prompts = [
            self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            for prompt in batch.non_tensor_batch["raw_prompt"]
        ]

        # flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts)

        # 1. 生成 sequences，并记录每个 conversation 里环境 user 的 token 区间
        sequences = []
        env_user_token_spans = []  # 记录每个 sequence 里需要 mask 的 token 区间
        for conversation in batch_conversations:
            # 标记第一个 user 之后的所有 user 的内容
            user_indices = [i for i, msg in enumerate(conversation) if msg["role"] == "user"]
            if len(user_indices) > 1:
                # 只保留第一个 user，后面的都是环境反馈
                env_user_indices = user_indices[1:]
            else:
                env_user_indices = []

            # 生成完整文本
            sequence = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
            sequences.append(sequence)

            # 计算需要 mask 的 token 区间
            env_user_token_span = []
            if env_user_indices:
                # 计算每个 env user 在文本中的起止位置
                offset = 0
                for idx in env_user_indices:
                    # 拼接到当前 user 之前的所有内容
                    before = self.tokenizer.apply_chat_template(conversation[:idx], add_generation_prompt=False, tokenize=False)
                    start = len(self.tokenizer(before, return_tensors=None)["input_ids"])
                    # 当前 user 内容
                    user_text = self.tokenizer.apply_chat_template([conversation[idx]], add_generation_prompt=False, tokenize=False)
                    end = start + len(self.tokenizer(user_text, return_tensors=None)["input_ids"])
                    env_user_token_span.append((start, end))
            env_user_token_spans.append(env_user_token_span)

        # responses: [response]
        responses = [sequence[len(prompts[i // n]):] for i, sequence in enumerate(sequences)]

        prompts_enc = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses_enc = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts_enc["input_ids"] = prompts_enc["input_ids"].repeat_interleave(n, dim=0)
            prompts_enc["attention_mask"] = prompts_enc["attention_mask"].repeat_interleave(n, dim=0)

        input_ids = torch.cat([prompts_enc["input_ids"], responses_enc["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts_enc["attention_mask"], responses_enc["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        # 2. 对 attention_mask 进行 mask
        for i, spans in enumerate(env_user_token_spans):
            for start, end in spans:
                # 只 mask response 部分的 token
                prompt_len = prompts_enc["input_ids"].shape[1]
                # response token 在 input_ids 里的起始位置
                mask_start = prompt_len + start
                mask_end = prompt_len + end
                attention_mask[i, mask_start:mask_end] = 0

        batch = TensorDict(
            {
                "prompts": prompts_enc["input_ids"],
                "responses": responses_enc["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)