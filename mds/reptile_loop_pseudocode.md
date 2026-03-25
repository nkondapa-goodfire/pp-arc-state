# Reptile Meta-Learning Loop — Pseudocode

---

## Outer Loop (one step per task batch)

```
for each task_batch  [k mini-batches from the same GRN task]:

    θ_t  ←  snapshot(θ)                 # save current parameters

    inner_opt  ←  Adam(lr=inner_lr, β₁=0)   # fresh optimizer, no momentum carry-over

    ## Inner Loop
    for each mini_batch in task_batch:

        loss  ←  compute_inner_loss(mini_batch)
                 ├── forward pass  →  pred
                 ├── distributional loss  (primary)
                 ├── + batch-token CE loss      [optional]
                 ├── + decoder loss             [optional]
                 ├── + confidence token loss    [optional]
                 └── + L1 regularization        [optional]

        inner_opt.zero_grad()
        backward(loss)          # no DDP all-reduce (no_sync)
        clip_grad_norm(θ)
        inner_opt.step()

    θ̃  ←  θ  (adapted parameters, local to this GPU)

    ## Reptile Gradient
    g  ←  θ_t − θ̃

    ## Multi-GPU Sync (skip if single GPU)
    all_reduce(g, op=AVG)      # each GPU ran a different task; average directions

    ## Outer Update
    θ  ←  θ_t                 # restore snapshot
    θ.grad  ←  g
    clip_grad_norm(θ)
    outer_opt.step()           # θ ← θ_t − outer_lr · g
                               #   = θ_t + outer_lr · (θ̃ − θ_t)  ✓
```

---

## Key Hyperparameters

| Symbol      | Meaning                          | Default |
|-------------|----------------------------------|---------|
| `k`         | inner steps per task             | 10      |
| `inner_lr`  | inner Adam learning rate         | 1e-3    |
| `outer_lr`  | outer Adam learning rate         | 1e-3    |
| `grad_clip` | max gradient norm (inner+outer)  | 1.0     |

---

## Notes

- **Inner optimizer** uses β₁=0 (no momentum) — keeps each task's inner steps independent.
- **Outer optimizer** uses β₁=0.9 — moments accumulate *across tasks* (correct).
- **DDP**: inner backward calls are wrapped in `no_sync()` — zero all-reduces during inner loop.
  Reptile gradients are all-reduced once before the outer step.
- **Interpretation**: outer step moves θ toward the average post-adaptation point across all GPUs' tasks — equivalent to batched Reptile / SimuParallelSGD.
