#!/usr/bin/env python3
"""
dryrun_reptile_dataloader.py

Dry-run for ReptileDataModule + TaskGroupedBatchSampler.

Loads the SERGIO PPT dataset and iterates through N_OUTER outer steps.
For each outer step prints:
  - The GRN task identity (task_key = cell_type, e.g. "BA-VM_size010_seed0000")
  - For each of the k inner mini-batches:
      * unique pert names (should vary across inner steps)
      * unique bins / gem_groups
      * n cells
  - Summary: distinct pert and bin counts across k inner steps

Assertions verified:
  - Exactly k_inner inner batches per outer step
  - All inner batches share the same task_key
  - Every cell in every inner batch has cell_type == task_key

Run from the state repo root:
    uv run python simulate/sergio/scripts/dryrun_reptile_dataloader.py
"""

import sys
from collections import Counter
from pathlib import Path

# ── repo root on path ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]   # .../state
sys.path.insert(0, str(REPO_ROOT / "src"))

from state.tx.data.reptile import ReptileDataModule  # noqa: E402

# ── config ───────────────────────────────────────────────────────────────────
TOML     = str(REPO_ROOT / "simulate/sergio/configs/sergio_ppt_train.toml")
K_INNER  = 5     # inner steps per outer update
CELL_LEN = 64    # cells per inner mini-batch (matches n_sc in dataset_ppt.json)
N_OUTER  = 8     # outer steps to print
WORKERS  = 0     # 0 = main process only (easiest for a dry-run)

# ── datamodule setup ─────────────────────────────────────────────────────────
# cell_type now encodes full task identity: "{grn_type}_size{grn_size}_seed{seed}"
# No h5_to_task_prefix_fn needed — cell_type is unambiguous across files.
dm = ReptileDataModule(
    toml_config_path=TOML,
    batch_size=K_INNER,   # used only by val_dataloader; irrelevant here
    num_workers=WORKERS,
    cell_sentence_len=CELL_LEN,
    k_inner=K_INNER,
    pert_col="gene",
    cell_type_key="cell_type",
    batch_col="gem_group",
    control_pert="non-targeting",
    embed_key="X_hvg",
    output_space="gene",
    pin_memory=False,
)
dm.setup("fit")

loader = dm.train_dataloader()

# ── iterate ──────────────────────────────────────────────────────────────────
print(f"{'='*70}")
print(f"ReptileDataModule dry-run")
print(f"  TOML:          {TOML}")
print(f"  k_inner:       {K_INNER}")
print(f"  cell_sent_len: {CELL_LEN}")
print(f"  dataset size:  {len(loader)} outer steps per epoch")
print(f"{'='*70}\n")

for outer_idx, task_batch in enumerate(loader):
    if outer_idx >= N_OUTER:
        break

    # task_batch is List[Dict[str, Tensor | str]] of length k_inner
    assert len(task_batch) == K_INNER, (
        f"Expected {K_INNER} inner batches, got {len(task_batch)}"
    )

    # All inner batches must share the same task_key
    task_keys = [b["task_key"] for b in task_batch]
    assert len(set(task_keys)) == 1, (
        f"Mixed task_keys in one outer step: {set(task_keys)}"
    )
    task_key = task_keys[0]

    # Every cell in every inner batch must belong to this task
    for step_i, mb in enumerate(task_batch):
        wrong = [str(ct) for ct in mb["cell_type"] if str(ct) != task_key]
        assert not wrong, (
            f"Outer {outer_idx} inner {step_i}: cell_type mismatch "
            f"(expected {task_key}): {wrong[:3]}"
        )

    print(f"Outer step {outer_idx:3d}  │  task: {task_key}")
    print(f"{'─'*60}")

    all_perts = []
    all_bins  = []
    for step_i, mb in enumerate(task_batch):
        cell_types   = [str(ct) for ct in mb["cell_type"]]
        pert_names   = [str(p)  for p  in mb["pert_name"]]
        unique_perts = sorted(set(pert_names))
        unique_bins  = sorted({str(b) for b in mb["batch_name"]})
        n = mb["ctrl_cell_emb"].shape[0]

        # Each sentence must be homogeneous: one cell_type and one pert_name.
        assert len(set(cell_types)) == 1, (
            f"Outer {outer_idx} inner {step_i}: mixed cell_types in one sentence: "
            f"{set(cell_types)}"
        )
        assert len(unique_perts) == 1, (
            f"Outer {outer_idx} inner {step_i}: mixed pert_names in one sentence: "
            f"{unique_perts}"
        )

        all_perts.extend(unique_perts)
        all_bins.extend(unique_bins)

        print(
            f"  inner {step_i}  n={n:4d}"
            f"  perts={unique_perts}"
            f"  bins={unique_bins}"
        )

    print(
        f"  ── summary: {len(Counter(all_perts))} distinct pert(s),"
        f" {len(Counter(all_bins))} distinct bin(s) across {K_INNER} inner steps"
    )
    print()

print(f"{'='*70}")
print("Dry-run complete — no assertions failed.")
