"""
state_transition_reptile.py — Reptile wrapper for StateTransitionPerturbationModel.

StateTransitionReptileModel inherits from both ReptilePerturbationModel and
StateTransitionPerturbationModel via multiple inheritance.

MRO: StateTransitionReptileModel
       → ReptilePerturbationModel        (outer loop, automatic_optimization=False)
       → StateTransitionPerturbationModel (architecture, forward, _compute_distribution_loss)
       → PerturbationModel               (base Lightning module)

Constructor:
  ReptilePerturbationModel.__init__ pops (inner_lr, k_inner, outer_lr, grad_clip)
  from kwargs before forwarding to StateTransitionPerturbationModel.__init__,
  so all StateTransition kwargs pass through unchanged.

training_step:
  Provided by ReptilePerturbationModel — expects List[Dict] task batches.
  Calls compute_inner_loss() (implemented here) for each inner mini-batch.

compute_inner_loss:
  Replicates StateTransitionPerturbationModel.training_step without any
  self.log() calls. All auxiliary losses (decoder, batch-token CE, confidence
  token, L1 regularization) are included so the inner loop sees the same
  objective as standard training.
"""

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .reptile_base import ReptilePerturbationModel
from .state_transition import StateTransitionPerturbationModel

logger = logging.getLogger(__name__)


class StateTransitionReptileModel(ReptilePerturbationModel, StateTransitionPerturbationModel):
    """
    Reptile meta-learning outer loop wrapping StateTransitionPerturbationModel.

    Pass all StateTransitionPerturbationModel kwargs as usual; additionally:
        inner_lr  (float, default 1e-3)
        k_inner   (int,   default 10)
        outer_lr  (float, default 1e-3)
        grad_clip (float, default 1.0)

    The dataloader must yield List[Dict[str, Tensor]] task batches (one list
    per outer step), where every element is a mini-batch from the same GRN task.
    """

    def compute_inner_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Full StateTransition training objective for one inner-loop mini-batch.
        No self.log() calls — logging is done in the outer training_step.
        """
        # ----------------------------------------------------------------
        # Forward pass
        # ----------------------------------------------------------------
        confidence_pred = None
        if self.confidence_token is not None:
            pred, confidence_pred = self.forward(batch, padded=True)
        else:
            pred = self.forward(batch, padded=True)

        target = batch["pert_cell_emb"]
        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        # ----------------------------------------------------------------
        # Primary distributional loss
        # ----------------------------------------------------------------
        per_set_losses = self._compute_distribution_loss(pred, target)
        total_loss = torch.nanmean(per_set_losses)
        self.log("train_inner_loss", total_loss)

        # ----------------------------------------------------------------
        # Batch-token CE loss (optional)
        # ----------------------------------------------------------------
        if self.use_batch_token and self.batch_classifier is not None and self._batch_token_cache is not None:
            logits = self.batch_classifier(self._batch_token_cache)  # [B, 1, C]
            batch_token_targets = batch["batch"]
            B = logits.shape[0]
            C = logits.size(-1)

            if batch_token_targets.dim() > 1 and batch_token_targets.size(-1) == C:
                target_oh = batch_token_targets.reshape(-1, self.cell_sentence_len, C)
                sentence_batch_labels = target_oh.argmax(-1)
            else:
                sentence_batch_labels = batch_token_targets.reshape(-1, self.cell_sentence_len)

            if sentence_batch_labels.shape[0] != B:
                sentence_batch_labels = sentence_batch_labels.reshape(B, -1)

            if self.basal_mapping_strategy == "batch":
                uniform_mask = sentence_batch_labels.eq(sentence_batch_labels[:, :1]).all(dim=1)
                if not torch.all(uniform_mask):
                    bad_indices = torch.where(~uniform_mask)[0]
                    label_strings = [
                        f"sentence {idx.item()}: {sentence_batch_labels[idx].detach().cpu().tolist()}"
                        for idx in bad_indices
                    ]
                    raise ValueError(
                        "Expected all cells in a sentence to share the same batch when "
                        "basal_mapping_strategy is 'batch'. "
                        f"Found mixed batch labels: {', '.join(label_strings)}"
                    )

            target_idx = sentence_batch_labels[:, 0]
            if target_idx.numel() != B:
                target_idx = target_idx.reshape(-1)[:B]

            ce_loss = F.cross_entropy(logits.reshape(B, -1, C).squeeze(1), target_idx.long())
            total_loss = total_loss + self.batch_token_weight * ce_loss

        # ----------------------------------------------------------------
        # Decoder loss (optional)
        # ----------------------------------------------------------------
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"].reshape(
                -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
            )
            if self.detach_decoder:
                latent_preds = target.reshape_as(pred).detach() if np.random.rand() < 0.1 else pred.detach()
            else:
                latent_preds = pred
            decoder_loss = self._compute_distribution_loss(
                self.gene_decoder(latent_preds), gene_targets
            ).mean()
            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        # ----------------------------------------------------------------
        # Confidence token loss (optional)
        # ----------------------------------------------------------------
        if confidence_pred is not None:
            confidence_pred_vals = confidence_pred
            if confidence_pred_vals.dim() > 1:
                confidence_pred_vals = confidence_pred_vals.squeeze(-1)
            confidence_targets = per_set_losses.detach()
            if self.confidence_target_scale is not None:
                confidence_targets = confidence_targets * self.confidence_target_scale
            confidence_targets = confidence_targets.to(confidence_pred_vals.device)
            confidence_loss = self.confidence_weight * self.confidence_loss_fn(
                confidence_pred_vals, confidence_targets
            )
            total_loss = total_loss + confidence_loss

        # ----------------------------------------------------------------
        # L1 regularization (optional)
        # ----------------------------------------------------------------
        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"].reshape_as(pred)
            total_loss = total_loss + self.regularization * torch.abs(pred - ctrl_cell_emb).mean()

        return total_loss
