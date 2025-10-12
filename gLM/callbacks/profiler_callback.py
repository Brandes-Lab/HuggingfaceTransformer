"""PyTorch Profiler callback for Hugging Face Trainer."""

import os
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_callback import TrainingArguments
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule


class PyTorchProfilerCallback(TrainerCallback):
    """
    A callback that profiles the training loop using PyTorch Profiler.

    Args:
        output_dir: Directory to save profiler traces
        profile_steps: Number of steps to profile (default: 10)
        wait_steps: Number of steps to skip before profiling (default: 5)
        warmup_steps: Number of warmup steps (default: 2)
        active_steps: Number of active profiling steps (default: 3)
        repeat_times: Number of times to repeat the profiling cycle (default: 1)
        record_shapes: Whether to record tensor shapes (default: True)
        profile_memory: Whether to profile memory usage (default: True)
        with_stack: Whether to record stack traces (default: True)
        with_flops: Whether to estimate FLOPs (default: True)
    """

    def __init__(
        self,
        output_dir: str = "./profiler_traces",
        wait_steps: int = 5,
        warmup_steps: int = 2,
        active_steps: int = 3,
        repeat_times: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
    ):
        self.output_dir = output_dir
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.repeat_times = repeat_times
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.profiler = None

        # Only profile on rank 0 to avoid conflicts
        self.should_profile = int(os.environ.get("LOCAL_RANK", 0)) == 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize profiler at the start of training."""
        if not self.should_profile:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # Define profiler schedule
        profiler_schedule = schedule(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat_times,
        )

        # Determine activities based on available hardware
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # Create profiler
        self.profiler = profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
        )

        self.profiler.__enter__()
        print(
            f"PyTorch Profiler initialized. Traces will be saved to: {self.output_dir}"
        )
        print(
            f"Profiler schedule: wait={self.wait_steps}, warmup={self.warmup_steps}, "
            f"active={self.active_steps}, repeat={self.repeat_times}"
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Step the profiler after each training step."""
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Close profiler at the end of training."""
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            print(f"Profiling complete! Traces saved to: {self.output_dir}")
            print(f"\nTo view traces, run:")
            print(f"  tensorboard --logdir={self.output_dir}")


class SimpleGPUMemoryCallback(TrainerCallback):
    """
    A simple callback that logs GPU memory usage periodically.
    Useful for quick checks without full profiling overhead.
    """

    def __init__(self, log_every_n_steps: int = 10):
        self.log_every_n_steps = log_every_n_steps
        self.is_rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log GPU memory usage."""
        if not self.is_rank0:
            return

        if (
            state.global_step % self.log_every_n_steps == 0
            and torch.cuda.is_available()
        ):
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB

                print(
                    f"[Step {state.global_step}] GPU {i}: "
                    f"Allocated: {allocated:.2f}GB, "
                    f"Reserved: {reserved:.2f}GB, "
                    f"Peak: {max_allocated:.2f}GB"
                )
