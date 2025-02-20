# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from multiprocessing import reduction
import time
from transformers import Trainer
import trl
from unittest.mock import patch

from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
    TrainerState,
)
from collections import defaultdict
import math
import os
from typing import Callable, Optional, Union
import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from trl.trainer.utils import pad, selective_log_softmax
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from trl.data_utils import maybe_apply_chat_template, is_conversational
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
from vllm import LLM, SamplingParams

from accelerate import Accelerator
if is_wandb_available():
    import wandb

def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q

# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class FastGRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


class FastGRPOTrainer(Trainer):
    _tag_names = ["trl", "fast_grpo"]
    def __init__(self, 
                model: str, # only accept str for now
                reward_funcs: Union[RewardFunc, list[RewardFunc]],
                args: FastGRPOConfig, 
                train_dataset: Dataset,
                processing_class: Optional[PreTrainedTokenizerBase] = None,
                data_collator: Optional[DataCollatorWithPadding] = None, 
                callbacks: Optional[list[TrainerCallback]] = None,
                optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
                
                ) -> None:
        
        self.args = args
        self.reward_funcs = reward_funcs
        # Reward weights (move this logic to post_init of config?)
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = args.reward_weights
        else:
            self.reward_weights = [1.0] * len(reward_funcs),
        
        
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model_str = model
            model = AutoModelForCausalLM.from_pretrained(model_str, **model_init_kwargs)
            # offload to cpu
            ref_model = AutoModelForCausalLM.from_pretrained(model_str, **model_init_kwargs).to("cpu")
            
        self.model = model
        self.ref_model = ref_model
        

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        self.processing_class = processing_class
        

        
        self.train_dataset = train_dataset
        
        if data_collator is not None:
            raise ValueError("")
        
        def data_collator(features):  # No data collation is needed in GRPO
            return features       
        self.data_collator = data_collator
        
        local_dataloader_batch_size = exact_div(
            args.per_device_train_batch_size * args.gradient_accumulation_steps, 
            args.num_generations, "per_device_train_batch_size * gradient_accumulation_steps must >= num_generations to remain on policy"
        )
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        
        
        self.train_dataset_len = len(self.train_dataset)
        num_total_samples = int(self.args.num_train_epochs * self.train_dataset_len)
        self.total_steps_per_device = num_total_samples // (local_dataloader_batch_size * self.accelerator.num_processes)
        self.create_optimizer_and_scheduler(num_training_steps=self.total_steps_per_device)     
        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)

        # offload model to cpu, TODO: offload optimizer
        self.model = self.model.to("cpu")
        world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
        rank_patch = patch("torch.distributed.get_rank", return_value=0)
        get_backend_patch = patch("torch.distributed.get_backend", return_value=None)
        
        # profiling_patch = patch(
        #     "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
        # )
        with world_size_patch, rank_patch, get_backend_patch:
            self.gen_vllm = LLM(
                    model=model.name_or_path,
                    device=self.accelerator.device,
                    gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                    dtype=self.args.vllm_dtype,
                    enable_prefix_caching=True,
                    max_model_len=self.args.vllm_max_model_len,
                    enable_sleep_mode=True,
                )
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=self.args.max_completion_length,
            n=args.num_generations,
        )

    def print_gpu_memory_usage(self):
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_reserved = torch.cuda.memory_reserved()
            print(f"GPU memory allocated: {gpu_memory_allocated / (1024 ** 3):.2f} GB")
            print(f"GPU memory reserved: {gpu_memory_reserved / (1024 ** 3):.2f} GB")
        else:
            print("CUDA is not available.")

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
    
    @torch.no_grad()
    def prepare_batch(self, batch):
        """
        This will:
        - generate k samples for each problem
        - compute ref logprobs for each generation
        - using internal reward model(s) to get rewards
        """
        device = self.accelerator.device
        prompts = [x["prompt"] for x in batch]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in batch]
        prompt_inputs = self.processing_class(prompts_text)
        
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # add cuda clear cache here and a sleep

        all_outputs = self.gen_vllm.generate(prompts_text, sampling_params=self.sampling_params, use_tqdm=True)
        
        #offload vllm instance
        completion_ids = []
        for outputs in all_outputs:
            for output in outputs.outputs:
                completion_ids.append(output.token_ids)
               
        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt]*self.args.num_generations)
        
        if is_conversational(batch[0]):
            completions = []
            for prompt, completion in zip(repeated_prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text        
        
        rewards = torch.zeros(len(repeated_prompts), len(self.reward_funcs))
        for i, reward_func,  in enumerate(self.reward_funcs):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in batch[0] if key not in ["prompt", "completion"]]
                reward_kwargs = defaultdict(list)
                for example in batch:
                    for key in keys:
                        reward_kwargs[key].extend([example[key]]*self.args.num_generations)
                output_reward_func = reward_func(prompts=repeated_prompts, completions=completions, **reward_kwargs)
                rewards[:, i] = torch.tensor(output_reward_func, dtype=torch.float32) * self.reward_weights[i]
        
        # calculate the advantages, the prompt is all on the same device to no need to gather here
        grouped_rewards = rewards.sum(-1).view(len(prompts), self.args.num_generations)
        EPS = 1e-4
        grouped_advantages = (grouped_rewards - grouped_rewards.mean(-1, keepdim=True)) /  (grouped_rewards.std(-1, keepdim=True) + EPS)
        advantages = grouped_advantages.flatten().tolist()
        
        # build batch as list of dicts
        examples = []
        for i, prompt in enumerate(repeated_prompts):
            example = {
                "prompt": prompt,
                "prompt_ids": prompt_ids[i // self.args.num_generations],
                "completion": completions_text[i],
                "completion_ids": completion_ids[i],
                "advantages": advantages[i], 
                "rewards": rewards[i]
            }
            examples.append(example)
        
        return examples

    def _move_optimizer_to_device(self, optimizer, device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return optimizer

    def _sync_weights_to_vllm(self):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        state_dict =  unwrapped_model.state_dict()
        gen_model = self.gen_vllm.llm_engine.model_executor.driver_worker.model_runner.model
        gen_model.load_weights(state_dict.items())   

    def train(self,         
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ):
        start_step = 1 # todo, set this when we resume + load model, opt state etc
        
        if self.args.logging_steps is not None:
            if self.args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * self.args.logging_steps)
            else:
                self.state.logging_steps = self.args.logging_steps
                
        if self.args.save_steps is not None:
            if self.args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * self.args.save_steps)
            else:
                self.state.save_steps = self.args.save_steps
                
        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        self.state.global_step = 0
        self.state.max_steps = self.total_steps_per_device
        self.state.num_train_epochs = self.args.num_train_epochs
        
        def repeat_generator():
            while True:
                yield from self.dataloader
        iter_dataloader = iter(repeat_generator())
        
        self.model.train()
        
        @torch.no_grad()
        def mini_batch_collator(examples):
            device = self.accelerator.device
            
            prompt_ids = [example["prompt_ids"] for example in examples]
            completion_ids = [example["completion_ids"] for example in examples]
            
            if self.args.max_prompt_length is not None:
                prompt_ids = [p[-self.args.max_prompt_length :] for p in prompt_ids]
            
            pad_token_id = self.processing_class.pad_token_id
            prompt_completion_ids = [torch.LongTensor(p+c) for p,c in zip(prompt_ids, completion_ids)]
            prompt_completion_ids = pad(prompt_completion_ids, padding_value=pad_token_id)
            attention_mask = (prompt_completion_ids != pad_token_id).long() # TODO, this does not work as expected when pad_id == eos_id
        
            advantages = torch.Tensor([example["advantages"] for example in examples])
            
            return {
                "prompt_completion_ids": prompt_completion_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "advantages": advantages.to(device), 
            }
        
        for step in range(start_step, self.total_steps_per_device + 1):
            batch = next(iter_dataloader)
            
            self.ref_model = self.ref_model.to("cpu")
            self.optimizer = self._move_optimizer_to_device(self.optimizer, "cpu")
            
            # sync latest weights, requires both model and vllm instance to be on the same device
            torch.cuda.empty_cache()
            time.sleep(1.0)
            self.gen_vllm.wake_up()
            self._sync_weights_to_vllm()
            
            self.model = self.model.to("cpu")
            
            batch = self.prepare_batch(batch)
            torch.cuda.empty_cache()
            # time.sleep(1.0)
            self.accelerator.wait_for_everyone()
            self.gen_vllm.sleep()
            torch.cuda.empty_cache()
            time.sleep(1.0)
            # TODO: log completions, rewards, etc
            gen_dataset = Dataset.from_list(batch)
            
            # we could add some optimizations here like sorting the dataset by length to improve throughput, but we will keep it simple for now
            mini_batch_dataloader = DataLoader(
                gen_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True, # we technically don#t need to shuffle due to grad acc, but we may move to clipped loss later
                drop_last=True,
                collate_fn=mini_batch_collator
            )
            # optimization
            # stats for logging
            losses = []
            device = self.accelerator.device

             # fix because of interence on vllm.sleep() ?
            
            self.model = self.model.to(device)
            self.optimizer = self._move_optimizer_to_device(self.optimizer, device)
            for mini_batch in mini_batch_dataloader:
                prompt_completion_ids = mini_batch["prompt_completion_ids"]
                attention_mask = mini_batch["attention_mask"][:, 1:] #  TODO, fix padding with the optimization from the original grpo trainer
                logits_to_keep = prompt_completion_ids.size(1) - 1 # TODO, fix padding with the optimization from the original grpo trainer
                torch.cuda.empty_cache()
                
                # get the ref logprobs, this could also be done at the batch prepare step to avoid too much model unloading
                self.ref_model = self.ref_model.to(device)
                with torch.inference_mode():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )   
                self.ref_model = self.ref_model.to("cpu")
                torch.cuda.empty_cache()
                
                with self.accelerator.accumulate(self.model):
                    per_token_logps = self._get_per_token_logps(self.model, prompt_completion_ids, attention_mask, logits_to_keep)
                    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                    
                    advantages = mini_batch["advantages"]
                    per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
                    per_token_loss = per_token_loss + self.args.beta * per_token_kl
                    loss = (per_token_loss * attention_mask).sum() / attention_mask.sum()
                    
                    losses.append(loss.item())
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # TODO: weight sync


            # logging stats
            metrics = {}
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
            metrics["loss"] = self.accelerator.gather_for_metrics(torch.Tensor(losses).to(device)).mean().item()
            self.state.epoch = step / self.total_steps_per_device
            self.log(metrics)
            
            
            self.lr_scheduler.step()
            self.state.global_step += 1
            
            self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(self.model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
                
        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(self.model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
                
