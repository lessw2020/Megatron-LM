# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Any, Dict, Optional

import torch
from megatron.core import parallel_state


class TokenThroughputMonitor:
    """
    A simple token throughput monitor for Megatron training.

    This class tracks tokens per second (TPS) similar to torchtitan's metrics,
    using time.perf_counter() to measure elapsed time and tracking token flow.
    """

    def __init__(self, log_freq: int = 1):
        """
        Initialize the token throughput monitor.

        Args:
            log_freq: How often to log metrics (every N iterations)
        """
        self.log_freq = log_freq
        self.reset()

    def reset(self):
        """Reset all tracking variables."""
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.total_tokens = 0
        self.tokens_since_last_log = 0
        self.iteration = 0

    def update(self, num_tokens: int):
        """
        Update the monitor with the number of tokens processed in this iteration.

        Args:
            num_tokens: Number of tokens processed in this iteration
        """
        self.iteration += 1
        self.total_tokens += num_tokens
        self.tokens_since_last_log += num_tokens

    def should_log(self) -> bool:
        """Check if we should log metrics this iteration."""
        return self.iteration == 1 or self.iteration % self.log_freq == 0

    def get_throughput_metrics(self) -> Dict[str, float]:
        """
        Calculate and return throughput metrics.

        Returns:
            Dictionary containing throughput metrics
        """
        current_time = time.perf_counter()

        # Calculate time deltas
        time_since_start = current_time - self.start_time
        time_since_last_log = current_time - self.last_log_time

        # Calculate throughput metrics
        if time_since_last_log > 0:
            # Tokens per second since last log
            tps_recent = self.tokens_since_last_log / time_since_last_log
        else:
            tps_recent = 0.0

        if time_since_start > 0:
            # Overall tokens per second
            tps_overall = self.total_tokens / time_since_start
        else:
            tps_overall = 0.0

        # Get parallel dimensions for per-device calculations
        world_size = parallel_state.get_data_parallel_world_size() * parallel_state.get_tensor_model_parallel_world_size() * parallel_state.get_pipeline_model_parallel_world_size()

        # Tokens per second per device
        tps_per_device = tps_recent / max(world_size, 1)

        metrics = {
            'tps_recent': tps_recent,
            'tps_overall': tps_overall,
            'tps_per_device': tps_per_device,
            'total_tokens': self.total_tokens,
            'tokens_since_last_log': self.tokens_since_last_log,
            'time_since_start': time_since_start,
            'time_since_last_log': time_since_last_log,
            'iteration': self.iteration,
            'world_size': world_size
        }

        return metrics

    def log_and_reset(self, loss: Optional[float] = None, extra_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Log metrics and reset counters for next logging period.

        Args:
            loss: Optional loss value to include in logging
            extra_metrics: Optional additional metrics to include

        Returns:
            Dictionary of metrics that were logged
        """
        metrics = self.get_throughput_metrics()

        # Format the log message
        log_parts = [
            f"iter {self.iteration:4d}",
            f"tps: {metrics['tps_recent']:8,.0f}",
            f"tps/device: {metrics['tps_per_device']:6,.0f}",
        ]

        if loss is not None:
            log_parts.insert(1, f"loss: {loss:7.4f}")

        if extra_metrics:
            for key, value in extra_metrics.items():
                if isinstance(value, float):
                    log_parts.append(f"{key}: {value:.4f}")
                else:
                    log_parts.append(f"{key}: {value}")

        # Only log from rank 0 to avoid spam
        if parallel_state.get_data_parallel_rank() == 0 and parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.get_pipeline_model_parallel_rank() == 0:
            print(f"[THROUGHPUT] {' | '.join(log_parts)}")

        # Reset counters for next period
        self.last_log_time = time.perf_counter()
        self.tokens_since_last_log = 0

        return metrics


def calculate_tokens_in_batch(args) -> int:
    """
    Calculate the number of tokens in a batch based on Megatron args.

    Args:
        args: Megatron arguments object

    Returns:
        Number of tokens per batch
    """
    # For GPT models: tokens = micro_batch_size * seq_length * gradient_accumulation_steps
    micro_batch_size = getattr(args, 'micro_batch_size', 1)
    seq_length = getattr(args, 'seq_length', 2048)

    # Calculate gradient accumulation steps
    global_batch_size = getattr(args, 'global_batch_size', micro_batch_size)
    data_parallel_size = parallel_state.get_data_parallel_world_size()
    gradient_accumulation_steps = global_batch_size // (micro_batch_size * data_parallel_size)

    tokens_per_batch = micro_batch_size * seq_length * gradient_accumulation_steps

    return tokens_per_batch


# Global monitor instance
_throughput_monitor: Optional[TokenThroughputMonitor] = None


def initialize_throughput_monitor(log_freq: int = 1) -> TokenThroughputMonitor:
    """
    Initialize the global throughput monitor.

    Args:
        log_freq: How often to log metrics (every N iterations)

    Returns:
        The initialized monitor
    """
    global _throughput_monitor
    _throughput_monitor = TokenThroughputMonitor(log_freq)
    return _throughput_monitor


def get_throughput_monitor() -> Optional[TokenThroughputMonitor]:
    """Get the global throughput monitor instance."""
    return _throughput_monitor


def update_throughput(num_tokens: int):
    """Update the global throughput monitor with token count."""
    if _throughput_monitor is not None:
        _throughput_monitor.update(num_tokens)


def log_throughput(loss: Optional[float] = None, extra_metrics: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, float]]:
    """Log throughput metrics if monitor is initialized and should log."""
    if _throughput_monitor is not None and _throughput_monitor.should_log():
        return _throughput_monitor.log_and_reset(loss, extra_metrics)
    return None
