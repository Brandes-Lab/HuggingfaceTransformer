# Profiling Quick Start

## TL;DR - Just Profile My Training!

### Option 1: Run the Pre-made Script (Easiest)

```bash
./local_scripts/train_lambda_with_profiling.sh
```

Then view results:
```bash
tensorboard --logdir=./profiler_traces --port=6006
```

Open browser: http://localhost:6006 → Go to "PYTORCH_PROFILER" tab

---

### Option 2: Add to Your Existing Script

Edit your training script (`train_lambda.sh`) and add these flags:

```bash
torchrun \
    --nnodes=1 \
    --nproc-per-node=8 \
    python_scripts/train_modernBERT.py \
    --run-name my-training-run \
    # ... your existing arguments ... \
    --enable-profiling \
    --enable-memory-logging
```

---

## Common Profiling Scenarios

### Scenario 1: Quick Memory Check

```bash
# Just want to see memory usage without full profiling overhead
./local_scripts/train_lambda.sh \
    --enable-memory-logging \
    --memory-log-steps 10
```

### Scenario 2: Profile for Longer

```bash
# Profile more steps for better statistics
./local_scripts/train_lambda.sh \
    --enable-profiling \
    --profiler-active-steps 10 \
    --profiler-repeat 3
```

### Scenario 3: Profile Later in Training

```bash
# Skip the first 100 steps, then profile
./local_scripts/train_lambda.sh \
    --enable-profiling \
    --profiler-wait-steps 100 \
    --profiler-active-steps 5
```

### Scenario 4: Full Profiling + Memory Logging

```bash
# Everything enabled
./local_scripts/train_lambda.sh \
    --enable-profiling \
    --enable-memory-logging \
    --profiler-wait-steps 5 \
    --profiler-warmup-steps 2 \
    --profiler-active-steps 5 \
    --profiler-repeat 2 \
    --memory-log-steps 5
```

---

## What to Look for in TensorBoard

1. **GPU Utilization**: Should be >80% ideally
2. **Memory Timeline**: See when OOM might occur
3. **Kernel View**: Which operations take the most time
4. **Data Loading**: Check if CPU is bottleneck
5. **Step Time**: Consistent or varying?

---

## Profiling Best Practices

✅ **DO:**
- Profile with a smaller max-steps (e.g., 50-100) to save time
- Skip the first 5-10 steps (they're slow due to initialization)
- Profile on rank 0 only (automatic)
- Use `--enable-memory-logging` for quick checks

❌ **DON'T:**
- Profile your entire long training run (unnecessary overhead)
- Profile too many steps (3-5 active steps is enough)
- Forget that profiling adds memory overhead

---

## Troubleshooting

**Out of Memory during profiling?**
```bash
# Reduce profiling overhead
--profiler-active-steps 1 \
--profiler-repeat 1
```

**Traces too large?**
```bash
# Less detailed profiling
--profiler-active-steps 2
# Don't repeat
--profiler-repeat 1
```

**Can't access TensorBoard?**
```bash
# Try different port
tensorboard --logdir=./profiler_traces --port=6007

# Or view traces in Chrome
# Open chrome://tracing
# Load .json.gz file from profiler_traces/
```

---

## Example Output

When profiling is enabled, you'll see:

```
================================================================================
PyTorch Profiler ENABLED
Output directory: ./profiler_traces
Schedule: wait=5, warmup=2, active=3, repeat=2
================================================================================
```

And during training:

```
[Step 10] GPU 0: Allocated: 24.52GB, Reserved: 25.12GB, Peak: 24.89GB
[Step 15] GPU 0: Allocated: 24.51GB, Reserved: 25.12GB, Peak: 24.89GB
```

At the end:

```
Profiling complete! Traces saved to: ./profiler_traces

To view traces, run:
  tensorboard --logdir=./profiler_traces
```

---

See `PROFILING_GUIDE.md` for detailed documentation.



