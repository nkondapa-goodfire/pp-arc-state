#!/usr/bin/env python3
"""
smoketest_reptile.py

End-to-end smoke test for the Reptile training pipeline using Lightning Trainer:
  ReptileDataModule → TaskGroupedBatchSampler → task_collate_fn
  → StateTransitionReptileModel.training_step (automatic_optimization=False)

Runs MAX_STEPS outer steps via trainer.fit(), printing train/inner_loss_mean
and train/reptile_grad_norm to stdout via a simple callback.

Run from the state repo root:
    uv run python simulate/sergio/scripts/smoketest_reptile.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from state.tx.data.reptile import ReptileDataModule
from state.tx.models.state_transition_reptile import StateTransitionReptileModel

# ── config ────────────────────────────────────────────────────────────────────
TOML       = str(REPO_ROOT / "simulate/sergio/configs/sergio_ppt_train.toml")
PERT_MAP   = str(REPO_ROOT / "simulate/sergio/configs/pert_onehot_map_ppt.pt")
CELL_LEN   = 64
K_INNER    = 3
MAX_STEPS  = 20
HIDDEN_DIM = 128
N_LAYERS   = 2
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── data ──────────────────────────────────────────────────────────────────────
print(f"Setting up ReptileDataModule (device={DEVICE}) ...")
dm = ReptileDataModule(
    toml_config_path=TOML,
    batch_size=K_INNER,
    num_workers=0,
    cell_sentence_len=CELL_LEN,
    k_inner=K_INNER,
    pert_col="gene",
    cell_type_key="cell_type",
    batch_col="gem_group",
    control_pert="non-targeting",
    embed_key="X_hvg",
    output_space="gene",
    perturbation_features_file=PERT_MAP,
    pin_memory=False,
)
dm.setup("fit")
var_dims = dm.get_var_dims()
print(f"  var_dims: { {k: v for k, v in var_dims.items() if not isinstance(v, list)} }")
print(f"  outer steps/epoch: {len(dm.train_dataloader())}\n")

# ── model ─────────────────────────────────────────────────────────────────────
print("Building StateTransitionReptileModel ...")
model = StateTransitionReptileModel(
    input_dim=var_dims["output_dim"],
    hidden_dim=HIDDEN_DIM,
    gene_dim=var_dims["hvg_dim"],
    hvg_dim=var_dims["hvg_dim"],
    output_dim=var_dims["output_dim"],
    pert_dim=var_dims["pert_dim"],
    batch_dim=var_dims["batch_dim"],
    basal_mapping_strategy="random",
    distributional_loss="energy",
    loss="energy",
    cell_set_len=CELL_LEN,
    predict_residual=True,
    softplus=True,
    batch_encoder=True,
    output_space="gene",
    embed_key="X_hvg",
    gene_names=var_dims["gene_names"],
    batch_size=K_INNER,
    control_pert="non-targeting",
    n_encoder_layers=N_LAYERS,
    n_decoder_layers=N_LAYERS,
    transformer_backbone_key="llama",
    transformer_backbone_kwargs=dict(
        bidirectional_attention=True,
        max_position_embeddings=CELL_LEN,
        hidden_size=HIDDEN_DIM,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        use_cache=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        layer_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rotary_dim=0,
        use_rotary_embeddings=False,
    ),
    inner_lr=1e-3,
    k_inner=K_INNER,
    outer_lr=1e-3,
    grad_clip=1.0,
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  trainable params: {n_params:,}\n")


# ── callback: print metrics each step ────────────────────────────────────────
class PrintMetrics(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = {k: f"{v:.4f}" for k, v in trainer.callback_metrics.items()}
        print(f"  step {trainer.global_step:3d}  {metrics}")


# ── trainer ───────────────────────────────────────────────────────────────────
print(f"{'='*70}")
print(f"Reptile smoke test (Lightning)  |  k_inner={K_INNER}  cell_len={CELL_LEN}  device={DEVICE}")
print(f"{'='*70}\n")

trainer = pl.Trainer(
    max_steps=MAX_STEPS,
    accelerator="gpu" if DEVICE == "cuda" else "cpu",
    devices=1,
    enable_checkpointing=False,
    enable_progress_bar=False,
    logger=False,
    callbacks=[PrintMetrics()],
)

trainer.fit(model, datamodule=dm)

print(f"\n{'='*70}")
print("Smoke test complete — no errors.")
