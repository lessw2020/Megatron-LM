import os
from functools import partial
from pathlib import Path

import torch

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.tokenizer.tokenizer import _NullTokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader


_SEQUENCE_LENGTH = 64


def initialize_distributed(
    tensor_model_parallel_size=1, pipeline_model_parallel_size=1
):
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()

    # Set device before initializing process group
    torch.cuda.set_device(rank)

    # Initialize process group - NCCL will use the current device
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )
    print(f"torch distributed initialized on device {torch.cuda.current_device()}")

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )


def model_provider():
    """Build the model."""

    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )

    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=_SEQUENCE_LENGTH,
    )

    return gpt_model


def get_train_data_iterator():
    print("Starting get_train_data_iterator()")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        print(f"Rank {rank}: About to compile helpers")

        # Compile helpers on all ranks to avoid barrier synchronization
        compile_helpers()
        print(f"Rank {rank}: Compiled helpers")

        # Add a small delay to ensure all ranks finish compilation
        import time

        time.sleep(1)
        print(f"Rank {rank}: Synchronization delay completed")
    else:
        print("Single process: compiling helpers")
        compile_helpers()

    print("Successfully compiled the helpers")

    print("Creating GPTDatasetConfig")
    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=_SEQUENCE_LENGTH),
        mid_level_dataset_surplus=0.005,
    )

    print("Building BlendedMegatronDataset")

    # Create a custom is_built_on_rank function that avoids distributed barriers
    def is_built_on_rank():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Build on all ranks to avoid barrier synchronization issues
            return True
        return True

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], is_built_on_rank, config
    ).build()

    print("Creating DataLoader")
    train_dataloader = DataLoader(datasets[0], batch_size=8, shuffle=True)

    print("Creating iterator")
    train_iterator = iter(train_dataloader)

    print("get_train_data_iterator() completed successfully")
    return train_iterator


def forward_step_func(data_iterator, model):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {"lm loss": loss}

    data = next(data_iterator)
    tokens = data["tokens"].to(device)
    attention_mask = data["attention_mask"].to(device)
    position_ids = data["position_ids"].to(device)
    labels = data["labels"].to(device)
    loss_mask = data["loss_mask"].to(device)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    dist_checkpointing.save(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )


def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    checkpoint = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
    gpt_model.load_state_dict(checkpoint)
    return gpt_model


if __name__ == "__main__":
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    gpt_model = model_provider()
    device = torch.device("cuda")
    gpt_model.to(device)

    optim = Adam(gpt_model.parameters())

    train_iterator = get_train_data_iterator()

    forward_backward_func = get_forward_backward_func()

    # Running the model for 5 iterations
    for _ in range(5):
        optim.zero_grad()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=8,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False,
        )

        optim.step()

        print(f"Losses reduced :  {losses_reduced}")

    # Saving the model
    ckpt_path = os.getcwd() + "/ckpt"
    Path(ckpt_path).mkdir(exist_ok=True)
    save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)

    # Loading the model
    gpt_model = load_distributed_checkpoint(
        gpt_model=gpt_model, checkpoint_path=ckpt_path
    )
    gpt_model.to(device)
    print("Successfully loaded the model")
