import contextlib
import functools
import gc
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import reduction
from typing import Callable, Optional, Union
from unittest.mock import patch
from confection import ARGS_FIELD_ALIAS
from transformers.utils import is_liger_kernel_available
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

import trl
from accelerate import Accelerator
from accelerate.utils import gather_object
from open_r1.trainers.job_launcher import SGLangSlurmJobLauncher
from open_r1.trainers.remote_model import RemoteModel
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.trainer.utils import pad, selective_log_softmax
from accelerate import Accelerator
from trl import SFTTrainer
from open_r1.trainers.special_dataloader import RemoteGRPODataloader
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState
if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


if is_wandb_available():
    import wandb
def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q

@contextlib.contextmanager
def profiling_context(instance, name):
    """
    A context manager function for profiling a block of code.
    Can also be used as a decorator.
    """
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time

    if "wandb" in instance.args.report_to and wandb.run is not None and instance.accelerator.is_main_process:
        wandb.log({f"profiling/Time taken: {instance.__class__.__name__}.{name}": duration})


def profiling_decorator(func):
    """
    Decorator to profile a function and log execution time using profiling_context.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper
# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class RemoteGRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    
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
    remote_gen_model_url: str = field(
        default="0.0.0.0",
    )
    remote_gen_model_port: str = field(
        default="30010",
    )
    remote_gen_model_n_gpus: str = field(
        default=8,
    )
    use_liger: bool = field(
        default=True,
        metadata={"help": "Whether to use Liger kernel for training."},
    )


class RemoteGRPOTrainer(Trainer):
    def __init__(self, model, 
                 reward_funcs: Union[RewardFunc, list[RewardFunc]], 
                 args: RemoteGRPOConfig, 
                 train_dataset, 
                 processing_class, 
                 callbacks):
        self.args = args
        self.remote_model = RemoteModel(
            self.args.remote_gen_model_url,
            self.args.remote_gen_model_port,
            processing_class.eos_token_id,
        )
        
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)
            
        def data_collator(features):  # No data collation is needed in GRPO
            return features
            
        super().__init__(model, args, train_dataset=train_dataset, processing_class=processing_class, callbacks=callbacks, data_collator=data_collator)
        
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        # if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if self.args.dataloader_num_workers != 0:
            raise ValueError("dataloader_num_workers should not be greater than 0 for remote training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers, #should be 0
            "pin_memory": self.args.dataloader_pin_memory, # should be False ?
            "persistent_workers": self.args.dataloader_persistent_workers, 
            "config": self.args,
            "remote_model": self.remote_model,
            "processing_class": self.processing_class,
            "reward_funcs": self.reward_funcs,
            "config": self.args,
               
        }
        return self.accelerator.prepare(RemoteGRPODataloader(train_dataset,
                                                             **dataloader_params))
        
    def _create_model_from_path(self, model_path: str, args) -> PreTrainedModel:
        """Creates a model from a path or model identifier."""
        model_init_kwargs = args.model_init_kwargs or {}
        # Handle torch dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `SFTConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        if args.gradient_checkpointing:
            model_init_kwargs["use_cache"] = False

        # Create model
        if args.use_liger:
            if not is_liger_kernel_available():
                raise ImportError("Please install Liger-kernel for use_liger=True")
            model = AutoLigerKernelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        return model
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        print("compute_loss")
        pass