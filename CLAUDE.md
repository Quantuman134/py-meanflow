# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Requirement from Instructions

0. When you execute any following instructions, you need to say which instruction you will execute next in the chat box. If no following instruction is executed, say you do not need to execute them in chat box.

1. After each large update (like add a function, or change the working logic), need to execute the instruction: "Check if this change introduce any logical error. If so, fix it."

2. After each update related to variable type transform, or change the type of variable compared to orginal version. Check if you made any mistake.

## Project Overview

PyTorch + GPU re-implementation of [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) on CIFAR-10. The original paper used JAX+TPU; this port achieves ~2.8–2.9 FID on CIFAR-10 using 8 GPUs. The key mechanism is **JVP (Jacobian-vector product)** for one-step image generation via `torch.func.jvp`.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate meanflow
```

Requires PyTorch 2.7.1+ (uses `torch.compile` and `torch.func.jvp`).

## Common Commands

**Train (8 GPUs, recommended config):**
```bash
cd meanflow/scripts/
bash cifar10_v1.sh           # ~2.9 FID at 16000 epochs, 0.21s/iter on H200s
bash cifar10_v0.sh           # original paper config
```

**Quick sanity check (single-batch run):**
```bash
cd meanflow/
torchrun --standalone --nproc_per_node=1 --master_port=12345 \
    train.py --dataset=cifar10 --test_run --epochs=1
```

**Demo / FID evaluation:**
```bash
jupyter notebook demo.ipynb   # expects <2.9 FID from a pre-trained checkpoint
```

All training is launched from `meanflow/` (not the project root), since imports are relative to that directory.

## Architecture

### Core Training Flow

The training loop in [training/train_loop.py](meanflow/training/train_loop.py) calls a compiled `train_step` → `MeanFlow.forward_with_loss()` → `torch.func.jvp`. Understanding these three layers together is essential:

1. **`MeanFlow.forward_with_loss()`** ([models/meanflow.py](meanflow/models/meanflow.py)): Samples `(t, r)` timestep pairs, constructs the noisy `z = t*x + (1-t)*e` (t=0: noise, t=1: clean), defines `u_func` (a closure over `self.net`), and calls `torch.func.jvp(u_func, ...)` to get both the network prediction `u_pred` and its time-derivative `dudt` in a single pass. The target is `u_tgt = v - (t - r) * dudt` (detached). Loss uses adaptive per-sample weighting.

2. **`train_step`** ([training/train_loop.py](meanflow/training/train_loop.py:70)): A plain function (not a method) wrapping `forward_with_loss` + `loss.backward()`. It is wrapped with `torch.compile()` at startup. The function takes `model_without_ddp` (not the DDP wrapper) because `torch.func.jvp` does not support DDP objects.

3. **DDP + JVP workaround** ([training/train_loop.py](meanflow/training/train_loop.py:24)): Because `torch.func.jvp` bypasses DDP's gradient hooks, `synchronize_gradients(model)` manually all-reduces gradients after each step. A `gradient_sanity_check` verifies norm agreement across ranks every 100 epochs.

### Timestep Sampling

[models/time_sampler.py](meanflow/models/time_sampler.py) samples pairs `(t, r)` with `t <= r` from a logit-normal distribution (t=noisy query, r=clean reference). Two variants exist:
- `v0`: paper version — sort then randomly collapse `r` to `t`
- `v1` (recommended): collapse first, then clamp `t = min(t, r)`

The `--ratio` argument controls the fraction of steps where `t != r`.

### RNG Control

[models/rng.py](meanflow/models/rng.py) mimics JAX's `fold_in` via SHA256 hashing to derive per-(step, rank, purpose) seeds. All stochastic operations inside `train_step` and `augment_pipe` are wrapped with `torch.random.fork_rng` so that the global RNG state is unaffected. This enables reproducibility across distributed ranks without rank-specific code paths in model forward passes.

### EMA Management

`MeanFlow` owns one primary EMA (`net_ema`) and any number of extra EMAs (`net_ema1`, `net_ema2`, ...) configured via `--ema_decays`. All EMA nets are submodules and included in checkpoints. `sample()` uses `net_ema` by default; the eval loop evaluates all EMA variants and the non-EMA net.

### Model Architecture

The backbone is `SongUNet` ([models/unet.py](meanflow/models/unet.py)), ported from the EDM repo. The network takes `(z, (t, h), aug_cond)` where `h = r - t` is the interval length (always >= 0 since r >= t) — this is the only way `r` enters the network. `aug_cond` carries EDM augmentation conditioning (or `None` at inference).

## Key Design Notes

- **No compilation fallback**: If `--compile` is set (default on), the code asserts that at least one compilation occurred. To disable: pass `--no-compile`.
- **Alternative to JVP compilation**: For ops unsupported by `torch.compile`, compute `u_pred` separately under `torch.no_grad()` then call `jvp` for `dudt` only (see README for the pattern).
- **Pixel scaling**: Input images are normalized to `[-1, 1]` in the training loop (`samples * 2.0 - 1.0`) before being passed to the model.
- **Inference is one step**: `MeanFlow.sample()` starts from z_0=noise (t=0), does `z_1 = z_0 + u`, a single network call. No iterative sampling.
- **License**: CC-BY-NC-SA 4.0 — non-commercial use only.
