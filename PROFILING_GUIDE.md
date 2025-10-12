# Training Profiling Guide

This guide shows you how to profile CPU and GPU usage in your training script.

## Quick Start: Using Command Line Arguments (Easiest!)

Your training script now has built-in profiling support via CLI arguments!

### Basic Usage

```bash
# Enable profiling with default settings
./local_scripts/train_lambda.sh \
    --enable-profiling

# Enable profiling + GPU memory logging
./local_scripts/train_lambda.sh \
    --enable-profiling \
    --enable-memory-logging

# Customize profiler settings
./local_scripts/train_lambda.sh \
    --enable-profiling \
    --profiler-output-dir ./my_traces \
    --profiler-wait-steps 10 \
    --profiler-warmup-steps 2 \
    --profiler-active-steps 5 \
    --profiler-repeat 2 \
    --enable-memory-logging \
    --memory-log-steps 5
```

### Pre-configured Profiling Script

Use the ready-made script for profiling:

```bash
./local_scripts/train_lambda_with_profiling.sh
```

This script runs a short training (50 steps) with profiling enabled and tells you how to view the results.

### Available CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-profiling` | False | Enable PyTorch profiler |
| `--enable-memory-logging` | False | Enable simple GPU memory logging |
| `--profiler-output-dir` | `./profiler_traces` | Directory to save traces |
| `--profiler-wait-steps` | 5 | Steps to skip before profiling |
| `--profiler-warmup-steps` | 2 | Warmup steps before active profiling |
| `--profiler-active-steps` | 3 | Steps to actively profile |
| `--profiler-repeat` | 1 | Times to repeat profiling cycle |
| `--memory-log-steps` | 10 | Log GPU memory every N steps |

## Understanding the Profiler Output

### What Gets Profiled?

The PyTorch Profiler captures:
- **CPU Time**: Time spent on CPU operations
- **GPU Time**: Time spent on CUDA kernels
- **Memory**: GPU and CPU memory allocations
- **Operations**: Detailed breakdown of PyTorch operations
- **FLOPs**: Estimated floating point operations
- **Stack Traces**: Where operations are called from

### Profiler Schedule

The profiler uses a schedule to control when profiling is active:

```
wait (5 steps) -> warmup (2 steps) -> active (3 steps) -> repeat
```

- **Wait**: Skip initial steps (they're often slower due to initialization)
- **Warmup**: Let things stabilize before recording
- **Active**: Actually record profiling data
- **Repeat**: Optionally repeat the cycle to get multiple samples

## Viewing Profiler Results

### Option 1: TensorBoard (Best for Interactive Exploration)

```bash
# Install tensorboard if needed
pip install tensorboard torch-tb-profiler

# Launch TensorBoard
tensorboard --logdir=./profiler_traces --port=6006

# Open in browser: http://localhost:6006
# Navigate to the "PYTORCH_PROFILER" tab
```

TensorBoard shows:
- Timeline view of operations
- GPU utilization over time
- Memory usage over time
- Top operations by time/memory
- Distributed training view (for multi-GPU)

### Option 2: Export to Chrome Trace

The traces are automatically saved as Chrome trace files. You can:

1. Open Chrome browser
2. Go to `chrome://tracing`
3. Click "Load" and select the `.json.gz` trace file from `profiler_traces/`

### Option 3: Programmatic Analysis

```python
import torch
from torch.profiler import profile, ProfilerActivity

# After profiling, you can analyze results programmatically
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Your code here
    pass

# Print table sorted by self CPU time
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

# Print table sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Print table sorted by memory
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# Export to Chrome trace
prof.export_chrome_trace("trace.json")
```

## Alternative Profiling Methods

### 1. nvidia-smi for Real-Time GPU Monitoring

```bash
# Watch GPU usage in real-time (updates every 1 second)
watch -n 1 nvidia-smi

# Or log to file
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu --format=csv -l 1 > gpu_usage.csv
```

### 2. nvtop (Better than nvidia-smi)

```bash
# Install nvtop
sudo apt install nvtop

# Run
nvtop
```

### 3. py-spy for CPU Profiling (Minimal Overhead)

```bash
# Install
pip install py-spy

# Profile your training (run this in parallel with your training)
py-spy record -o profile.svg --pid <PID>

# Or run it directly
py-spy record -o profile.svg -- python python_scripts/train_modernBERT.py ...
```

### 4. Memory Profiler

```python
# Add to your code
import tracemalloc

# At the start of training
tracemalloc.start()

# After some training steps
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024**3:.2f}GB")
print(f"Peak memory: {peak / 1024**3:.2f}GB")

tracemalloc.stop()
```

## Profiling for Short Duration (Recommended for Testing)

Since profiling adds overhead, you might want to:

1. **Profile only a few steps**: Use the callback with `active_steps=3-5`
2. **Profile on a subset of data**: Reduce your dataset temporarily
3. **Profile specific sections**: Use context managers

Example for profiling specific sections:

```python
from torch.profiler import profile, record_function, ProfilerActivity

# In your training loop or model
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("forward_pass"):
        outputs = model(**inputs)
    with record_function("backward_pass"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Tips for Effective Profiling

1. **Start Simple**: Use `SimpleGPUMemoryCallback` first to check memory usage
2. **Profile Later Steps**: Skip the first 10-20 steps (set `wait_steps=10`)
3. **Profile in Isolation**: Disable WandB logging during profiling to reduce noise
4. **Use Smaller Batch**: If memory is an issue, reduce batch size temporarily
5. **Multi-GPU**: Profiling only runs on rank 0 to avoid conflicts
6. **Compare Runs**: Profile with and without features (e.g., gradient checkpointing)

## Common Issues and Solutions

### Issue: Out of Memory During Profiling

**Solution**: Profiling adds memory overhead. Try:
- Reduce `active_steps` to 1-2
- Set `with_stack=False` and `record_shapes=False`
- Temporarily reduce batch size

### Issue: Traces are Too Large

**Solution**: 
- Reduce `active_steps`
- Don't use `repeat_times` > 1
- Profile only specific operations using context managers

### Issue: Can't See Traces in TensorBoard

**Solution**:
- Make sure `torch-tb-profiler` is installed: `pip install torch-tb-profiler`
- Check that trace files exist in the output directory
- Try a different port: `tensorboard --logdir=./profiler_traces --port=6007`

## Example: Complete Modified Training Script

See `python_scripts/train_modernBERT_with_profiling.py` for a complete example.

To run with profiling enabled:

```bash
# Enable profiling with command line flag
./local_scripts/train_lambda.sh --enable-profiling

# Or modify the script directly to add the callback
```

## Performance Metrics to Look For

1. **GPU Utilization**: Should be >80% for efficient training
2. **Memory Usage**: Should be close to capacity but not OOM
3. **Data Loading Time**: Should be minimal (use multiple workers)
4. **Kernel Launch Overhead**: Should be low relative to kernel execution
5. **CPU Bottlenecks**: Check if CPU is limiting GPU utilization

## Next Steps

1. Add the profiler callback to your training script
2. Run for a short duration (e.g., `--max-steps 50`)
3. View results in TensorBoard
4. Identify bottlenecks and optimize
5. Re-profile to verify improvements

