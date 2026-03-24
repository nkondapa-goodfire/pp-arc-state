"""
reptile_base.py — Reptile outer-loop wrapper for PerturbationModel.

Outer-loop logic
----------------
Each Lightning "batch" is a task batch: a list of k mini-batch dicts, all drawn
from the same GRN task (same grn_type, grn_size, grn_seed). The standard
dataloader yields flat dicts; a task-grouped collator is required upstream.

For each task batch received by training_step:

  1. Snapshot  θ_t = clone of all trainable parameters.
  2. Build a fresh inner Adam (β₁=0, lr=inner_lr). Never reused across tasks.
  3. For each of the k mini-batches (ALL wrapped in no_sync()):
       - forward + loss via compute_inner_loss()
       - manual_backward inside no_sync() — DDP all-reduce suppressed
       - inner_opt.step() / zero_grad()
  4. After k steps, θ̃ = current parameters (local, not yet synced).
  5. Reptile gradient:  g = θ_t − θ̃
     (Outer Adam does θ ← θ − lr·g, which gives θ + lr·(θ̃ − θ_t). ✓)
  6. All-reduce reptile_grads across GPUs (AVG).
     Each GPU contributed a different task's direction; the outer step moves
     toward their average — equivalent to batched Reptile / SimuParallelSGD.
  7. Restore θ_t into the model.
  8. Assign averaged g to param.grad for every trainable parameter.
  9. Clip by norm, then outer_opt.step() / zero_grad().
     Outer Adam accumulates moments across tasks (correct — not reset per task).

Batch format contract
---------------------
training_step receives:  task_batch: List[Dict[str, Tensor]]
  len(task_batch) == k_inner  (enforced by the task-grouped collator)
  Each element is a standard mini-batch dict with the usual keys:
    ctrl_cell_emb, pert_cell_emb, pert_emb, pert_cell_counts (optional), batch, cell_type

DDP note
--------
All inner manual_backward() calls are wrapped in no_sync() — zero inner
all-reduces. Reptile gradients are explicitly all-reduced (AVG) once before
the outer step. This is Option B (batched Reptile): each GPU sees a different
task; the outer update is the average over all tasks on all GPUs.
"""

import logging
from abc import abstractmethod
from contextlib import nullcontext
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn

from .base import PerturbationModel

logger = logging.getLogger(__name__)

TRAIN_BINS = {"bin_0", "bin_1", "bin_2", "bin_3"}  # bin_4 = zero-shot holdout


class ReptilePerturbationModel(PerturbationModel):
    """
    Subclass of PerturbationModel implementing the Reptile outer loop.

    Concrete subclasses must implement:
        compute_inner_loss(batch: Dict[str, Tensor]) -> Tensor
            Scalar loss for one inner-loop mini-batch (no self.log() calls).

    Additional constructor kwargs (consumed here, not forwarded to super):
        inner_lr  (float, default 1e-3)  inner Adam learning rate
        k_inner   (int,   default 10)    inner gradient steps per task
        outer_lr  (float, default 1e-3)  outer Adam learning rate
        grad_clip (float, default 1.0)   max norm for reptile gradient clipping
    """

    # Lightning hands optimizer.step() control to us.
    automatic_optimization = False

    def __init__(self, *args, **kwargs):
        self._inner_lr  = float(kwargs.pop("inner_lr",  1e-3))
        self._k_inner   = int(kwargs.pop("k_inner",     10))
        self._outer_lr  = float(kwargs.pop("outer_lr",  1e-3))
        self._grad_clip = float(kwargs.pop("grad_clip", 1.0))
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Subclasses implement this — no self.log() inside
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_inner_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Scalar training loss for one inner-loop mini-batch."""

    # ------------------------------------------------------------------
    # Reptile training step — one outer update per call
    # ------------------------------------------------------------------

    def training_step(
        self,
        task_batch: List[Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> None:
        outer_opt = self.optimizers()
        k = len(task_batch)

        # 1. Snapshot θ_t -----------------------------------------------
        theta_t: Dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

        # 2. Fresh inner optimizer (β₁=0 keeps inner gradients independent)
        inner_opt = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self._inner_lr,
            betas=(0.0, 0.999),
        )

        # 3. k inner gradient steps — all inside no_sync() ---------------
        #    Suppresses DDP all-reduce for every inner step.
        #    Single-GPU (no DDP): no_sync() is a no-op via nullcontext.
        ddp_model = self.trainer.model
        no_sync = ddp_model.no_sync if hasattr(ddp_model, "no_sync") else nullcontext

        inner_losses: List[torch.Tensor] = []
        for mini_batch in task_batch:
            loss = self.compute_inner_loss(mini_batch)
            inner_losses.append(loss.detach())

            inner_opt.zero_grad()
            with no_sync():
                self.manual_backward(loss)
            inner_opt.step()

        # 4. Reptile gradient: θ_t − θ̃ ---------------------------------
        #    After restore, outer opt does: θ ← θ_t − lr·(θ_t − θ̃)
        #                                     = θ_t + lr·(θ̃ − θ_t)  ✓
        reptile_grads: Dict[str, torch.Tensor] = {
            name: theta_t[name] - param.data
            for name, param in self.named_parameters()
            if param.requires_grad
        }

        # 5. All-reduce reptile_grads across GPUs (AVG) -----------------
        #    Each GPU ran a different task; averaging gives batched Reptile.
        #    Skipped when running on a single GPU (dist not initialized).
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            for g in reptile_grads.values():
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                g.div_(world_size)

        # 6. Restore θ_t ------------------------------------------------
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.data.copy_(theta_t[name])

        # 7. Assign averaged reptile gradient and outer step ------------
        outer_opt.zero_grad()
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.grad = reptile_grads[name]

        grad_norm = nn.utils.clip_grad_norm_(
            [p for p in self.parameters() if p.requires_grad],
            max_norm=self._grad_clip,
        )
        outer_opt.step()

        # 8. Logging ----------------------------------------------------
        self.log("train/inner_loss_mean", torch.stack(inner_losses).mean(), prog_bar=True)
        self.log("train/reptile_grad_norm", grad_norm)
        self.log("train/k_inner", float(k))

    # ------------------------------------------------------------------
    # Outer optimizer — moments accumulate across tasks
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self._outer_lr,
            betas=(0.9, 0.999),
        )
