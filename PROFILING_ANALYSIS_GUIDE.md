# PyTorch Profiler Analysis Guide

## What to Look for in Your Profiling Results

This guide helps you interpret PyTorch Profiler output and identify performance bottlenecks in your training.

---

## TensorBoard Profiler Views

### 1. **Overview Tab** (Start Here!)

This gives you a high-level summary of your training performance.

#### Key Metrics:

**GPU Utilization**
- **Target**: >80% for efficient training
- **<70%**: Likely CPU bottleneck (data loading or preprocessing)
- **<50%**: Serious bottleneck - GPU is waiting for work

**Step Time**
- Shows time per training step
- Look for consistency - should be relatively stable
- Large variations indicate:
  - Data loading issues
  - Dynamic batch sizes causing problems
  - Memory fragmentation

**GPU Memory Usage**
- Should be high but not maxing out (risk of OOM)
- **Ideal**: 85-95% of available memory
- **Too low (<50%)**: Could increase batch size
- **Too high (>98%)**: Risk of out-of-memory errors

**Performance Recommendation**
- TensorBoard provides automatic recommendations
- **Look for**: "Data loading is a bottleneck"
- **Look for**: "Kernel efficiency is low"

---

### 2. **Operator View** (Find Slow Operations)

Shows which PyTorch operations consume the most time.

#### What to Look For:

**Top Time Consumers** (Expected for Transformer Training):
1. `aten::linear` or `aten::matmul` - Matrix multiplications (30-50% of time)
2. `aten::layer_norm` - Layer normalization (5-10%)
3. `aten::softmax` - Attention softmax (5-10%)
4. `aten::add` or `aten::add_` - Residual connections
5. `Memcpy` operations - Data transfers

**Red Flags**:
- **High CPU time on `DataLoader` operations**: Data loading bottleneck
  - Fix: Increase `dataloader_num_workers`
  - Fix: Use `pin_memory=True`
  
- **Excessive `aten::copy_` or `aten::to`**: Too many device transfers
  - Fix: Ensure data is on GPU before forward pass
  - Fix: Reduce unnecessary `.cpu()` calls
  
- **High time in `optimizer.step`**: Optimizer bottleneck
  - Consider fused optimizers (e.g., `apex.optimizers.FusedAdam`)
  
- **`cudaLaunchKernel` overhead**: Small operations being launched
  - Consider operator fusion
  - May indicate inefficient gradient checkpointing

#### How to Sort:
- Sort by "Self CUDA Time" to see GPU kernel time
- Sort by "Self CPU Time" to see CPU overhead
- Sort by "# of Calls" to find operations called too frequently

---

### 3. **Kernel View** (GPU Specific)

Shows actual CUDA kernels being executed.

#### What to Look For:

**High-Performance Kernels** (Good):
- `volta_sgemm` / `ampere_sgemm` - Optimized matrix multiplication
- `fused_` operations - Multiple ops combined into one kernel
- Large "Mean Blocks/SM" (>1.0) - Good GPU occupancy

**Low-Performance Indicators** (Bad):
- Many small kernels with <100μs duration
- Low "Mean Blocks/SM" (<0.5) - Poor GPU utilization
- High "Mean Est. Achieved Occupancy" but low throughput - Memory bottleneck

**For Your ModernBERT Training**:
- Flash Attention kernels should dominate attention computation
- Look for: `fmha_` (Flash Attention) or `_attention` kernels
- These should be using Tensor Cores (high throughput)

---

### 4. **Trace View** (Timeline Analysis)

Interactive timeline showing what's happening when.

#### What to Look For:

**Healthy Training Pattern**:
```
[Data Load] [Forward] [Backward] [Optimizer] [Data Load] [Forward] ...
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ GPU Busy
```

**Problem Patterns**:

1. **GPU Idle Gaps**:
```
[Forward] [Backward] [...gap...] [Data Load] [Forward]
                     ^^^^^^^^^^^^ GPU waiting for CPU!
```
- Fix: Increase dataloader workers or prefetch factor

2. **CPU-GPU Transfer Overhead**:
```
[Forward] [H2D Copy] [Forward] [D2H Copy] [Backward]
          ^^^^^^^^^ Data transfer bottleneck
```
- Fix: Keep data on GPU, use pinned memory

3. **Gradient Accumulation Issues**:
```
[Forward] [Backward] [Forward] [Backward] ... [Long Optimizer Step]
                                              ^^^^^^^^^^^^^^^^^^^^^^
```
- This is expected with gradient accumulation
- Optimizer step should be <10% of total time

4. **Memory Operations**:
- Look for frequent `cudaMalloc` or `cudaFree`
- Indicates memory fragmentation
- Fix: Use `torch.cuda.empty_cache()` periodically (but sparingly)

#### Zoom Into a Single Step:
1. Find one complete training step
2. Measure time for:
   - Data loading
   - Forward pass
   - Backward pass
   - Optimizer step
   - Logging/callbacks

**Ideal Ratios** (for your setup):
- Forward: 35-40%
- Backward: 40-45%
- Optimizer: 10-15%
- Data/Other: <10%

---

### 5. **Memory View** (Understanding Memory Usage)

Shows GPU memory allocations over time.

#### What to Look For:

**Memory Timeline**:
- Should show steady pattern per step
- Peak memory during backward pass (stores gradients)
- Drop after optimizer step (gradient buffers freed)

**Red Flags**:
- **Constantly increasing memory**: Memory leak
  - Check for: Detached tensors accumulating
  - Check for: Lists of tensors growing unbounded
  - Check for: Not calling `loss.backward()` with `.item()` or `.detach()`

- **Saw-tooth pattern with high peaks**: Memory fragmentation
  - Large allocations causing fragmentation
  - May benefit from gradient checkpointing

- **Out of Memory (OOM)**: Spikes exceed available memory
  - Reduce batch size
  - Enable gradient checkpointing (you already have this)
  - Use `torch.cuda.amp` for mixed precision (you're using bf16)

**Memory Breakdown**:
Look at the pie chart showing:
- **Activations**: Intermediate forward pass values
  - High? Use gradient checkpointing (you already do)
- **Gradients**: Parameter gradients
  - Should be ~2x parameter size with Adam
- **Parameters**: Model weights
  - Fixed size based on model
- **Optimizer State**: Adam's momentum and variance
  - ~2x parameter size for Adam

---

## Specific Things for Your ModernBERT Training

### 1. Dynamic Batching Performance

Since you're using custom dynamic batching (`--dynamic-batching`):

**Check**:
- Step times should be more consistent than fixed batching
- Memory usage should be more uniform
- Look in Trace View for batch creation time

**If batching is slow**:
- Check time in `DynamicBatchSampler` or bucketing code
- May need to optimize batch collation
- Consider caching sorted indices

### 2. Gradient Accumulation

With `gradient_accumulation_steps=32`:

**Check**:
- 32 forward/backward passes per optimizer step
- Optimizer step time should be <10% of 32 accumulation steps
- Memory should be stable across accumulation steps

### 3. Distributed Training (8 GPUs)

With `WORLD_SIZE=8`:

**Check** (in Trace View):
- AllReduce operations during backward pass
- Should overlap with computation (NCCL should pipeline)
- Look for synchronization barriers

**Red Flags**:
- Large gaps waiting for AllReduce
- Unbalanced work across GPUs
- Frequent synchronization points

**To check per-GPU balance**:
```bash
# Run nvidia-smi in another terminal
watch -n 0.5 nvidia-smi

# All 8 GPUs should show similar:
# - Utilization %
# - Memory usage
# - Temperature
```

### 4. Gradient Checkpointing

You have `model.gradient_checkpointing_enable()`:

**Check**:
- Forward pass should be faster
- Backward pass should be slower (recomputes activations)
- Memory usage should be lower
- Trade-off: ~20% slower but saves memory

---

## Common Bottlenecks & Solutions

### Bottleneck 1: Data Loading (GPU <70% utilized)

**Symptoms**:
- GPU utilization low
- Gaps in Trace View between steps
- High "DataLoader" time in Operator View

**Solutions**:
```python
# Increase workers (you have 4)
dataloader_num_workers: int = field(default=8)  # Try 8-16

# Increase prefetch (you have 2)
dataloader_prefetch_factor: int = field(default=4)

# Ensure persistent workers (you already have this)
dataloader_persistent_workers: bool = field(default=True)

# Pin memory for faster H2D transfers
pin_memory=True  # in DataLoader
```

### Bottleneck 2: Small Batch Size (GPU underutilized)

**Symptoms**:
- Low GPU memory usage (<50%)
- High GPU utilization but low throughput
- Short kernels in Kernel View

**Solutions**:
```bash
# Increase tokens per batch
--max-tokens-per-batch 100000  # You have 50000

# Or increase per-device batch size
--per-device-train-batch-size 2  # You have 1

# With gradient accumulation, effective batch stays same
```

### Bottleneck 3: Inefficient Attention

**Symptoms**:
- High time in `aten::bmm` or attention operations
- Low kernel efficiency for attention

**Solutions**:
- Ensure Flash Attention is being used
- Check for Flash Attention kernels in Kernel View
- Verify sequence length bucketing is working

### Bottleneck 4: Optimizer Overhead

**Symptoms**:
- Long optimizer step time (>15% of iteration)
- High CPU time during optimizer step

**Solutions**:
```python
# Use fused optimizer
from torch.optim import AdamW
# or
from apex.optimizers import FusedAdam  # Faster

# Reduce optimizer frequency (already doing with grad accumulation)
```

### Bottleneck 5: Logging/Callbacks

**Symptoms**:
- High time in WandB or other callbacks
- Gaps after optimizer step

**Solutions**:
```python
# Log less frequently
logging_steps: int = field(default=100)  # You have 32

# Disable WandB during profiling
--disable-wandb
```

---

## Quick Checklist

Go through these in order:

- [ ] **GPU Utilization >80%?**
  - No → Fix data loading (increase workers/prefetch)
  
- [ ] **Step time consistent?**
  - No → Check for dynamic behavior or memory issues
  
- [ ] **Memory usage 85-95%?**
  - Too low → Increase batch size
  - Too high → Decrease batch size or enable grad checkpointing
  
- [ ] **Forward/Backward time dominant?**
  - No → Reduce optimizer/logging overhead
  
- [ ] **Flash Attention being used?**
  - Check Kernel View for fused attention kernels
  
- [ ] **No CPU-GPU transfer gaps?**
  - Use pinned memory and keep data on GPU
  
- [ ] **AllReduce overlapped with computation?**
  - For multi-GPU, NCCL should pipeline communication

---

## Example Analysis: Identifying Issues

### Example 1: Data Loading Bottleneck

**Observed**:
```
GPU Utilization: 45%
Step Time: 2.5s
- Data Loading: 1.0s (40%)  ← Problem!
- Forward: 0.8s
- Backward: 0.7s
```

**Fix**:
```bash
# Increase workers from 4 to 12
--dataloader-num-workers 12
--dataloader-prefetch-factor 4
```

**Expected After Fix**:
```
GPU Utilization: 85%
Step Time: 1.8s
- Data Loading: 0.3s (17%)  ← Fixed!
- Forward: 0.8s
- Backward: 0.7s
```

### Example 2: Small Batch Underutilization

**Observed**:
```
GPU Memory: 12GB / 40GB (30%)  ← Not using enough!
GPU Utilization: 75%
Small kernel times: 50μs average
```

**Fix**:
```bash
# Increase tokens per batch
--max-tokens-per-batch 100000  # from 50000
```

**Expected After Fix**:
```
GPU Memory: 35GB / 40GB (87%)  ← Better!
GPU Utilization: 92%
Kernel times: 150μs average
20% faster overall
```

---

## Tools for Deeper Analysis

### 1. Print Profiler Statistics

Add to your code:
```python
print(prof.key_averages().table(
    sort_by="cuda_time_total", 
    row_limit=20
))

# Or export to file
with open("profile_stats.txt", "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 2. Memory Profiling

```python
# Add to your code
print(torch.cuda.memory_summary())

# Or during training
if step % 100 == 0:
    print(torch.cuda.memory_stats())
```

### 3. NVIDIA Nsight Systems

For detailed GPU profiling:
```bash
nsys profile -o profile.qdrep \
  python python_scripts/train_modernBERT.py \
  --max-steps 10

# View in Nsight Systems GUI
nsys-ui profile.qdrep
```

---

## Benchmarking Your Optimizations

When you make changes, compare:

**Before**:
```
Throughput: 120 samples/sec
GPU Util: 65%
Memory: 18GB / 40GB
Step Time: 2.1s
```

**After** (with optimizations):
```
Throughput: 185 samples/sec  (+54% 🎉)
GPU Util: 88%                (+35%)
Memory: 36GB / 40GB          (better utilization)
Step Time: 1.4s              (-33% faster!)
```

---

## Summary: Quick Reference

| Metric | Target | If Too Low | If Too High |
|--------|--------|------------|-------------|
| GPU Util | >80% | ↑ workers, ↑ batch | - |
| GPU Memory | 85-95% | ↑ batch size | ↓ batch size |
| Step Time Variance | <10% | Fix dynamic issues | - |
| Data Load Time | <10% | ↑ workers/prefetch | - |
| Forward Time | 35-40% | - | Check model efficiency |
| Backward Time | 40-45% | - | Check grad checkpoint |
| Optimizer Time | <15% | - | Use fused optimizer |

---

## Next Steps

1. Run profiling: `./local_scripts/train_lambda_with_profiling.sh`
2. Open TensorBoard: `tensorboard --logdir=./profiler_traces`
3. Start with Overview tab → Check GPU utilization
4. Go to Operator View → Find top time consumers
5. Check Trace View → Look for gaps/patterns
6. Optimize based on findings
7. Re-profile to verify improvements

Good luck optimizing! 🚀



