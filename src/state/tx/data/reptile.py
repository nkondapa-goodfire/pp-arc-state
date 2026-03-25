"""
reptile.py — Task-grouped sampling and data module for Reptile meta-learning.

TaskGroupedBatchSampler
-----------------------
Standalone batch sampler (does not inherit PerturbationBatchSampler).

At construction it builds _task_to_cells: {task_key: [global_idx, ...]},
mapping each GRN task to every cell index that belongs to it across all
h5 subsets.

Each call to __iter__ yields one flat list of k_inner * cell_sentence_len
indices, freshly sampled without replacement from the task's cell pool
(with replacement only when the pool is smaller than the request).
Sampling is fresh every epoch, so no cell membership is fixed across steps.

The DataLoader delivers this list as k_inner * cell_sentence_len individual
samples, which task_collate_fn splits into k_inner collated mini-batches.

task_collate_fn
---------------
Receives the flat list of k_inner * cell_sentence_len samples from the DataLoader.
Splits into k_inner groups of cell_sentence_len, collates each with
PerturbationDataset.collate_fn, returns List[Dict[str, Tensor]] of length k_inner.
This is the format expected by ReptilePerturbationModel.training_step.

ReptileDataModule
-----------------
Thin subclass of PerturbationDataModule that overrides train_dataloader() to
use TaskGroupedBatchSampler + task_collate_fn instead of the standard sampler.

DDP note: each GPU runs the DataLoader independently, so different GPUs see
different tasks. The outer all-reduce in ReptilePerturbationModel averages the
reptile gradients across GPUs — this is the batched Reptile / SimuParallelSGD
update (correct behavior, no synchronization of samplers needed).
"""

import logging
from collections import defaultdict
from functools import partial
from typing import Callable, Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler, DataLoader

from cell_load.data_modules.perturbation_dataloader import (
    PerturbationDataModule,
    _worker_init_fn,
)
from cell_load.dataset._perturbation import PerturbationDataset
from cell_load.dataset._metadata import MetadataConcatDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class TaskGroupedBatchSampler(Sampler):
    """
    Yields batches of k_inner * cell_sentence_len indices, all from the same
    cell_type (GRN task).  Each group of cell_sentence_len indices becomes one
    inner-loop mini-batch after task_collate_fn processes the flat list.

    At construction, every cell in every h5 subset is assigned to a task key
    (= cell_type string, or h5_prefix/cell_type if h5_to_task_prefix_fn is
    given).  _task_to_cells stores the full pool of global indices per task.

    Each __iter__ call freshly samples k_inner perturbations from the task's
    pert pool, then draws cell_sentence_len cells from each — no fixed
    sentences, no partial-sentence edge cases, different cell combinations
    every epoch.  Each inner batch is homogeneous in perturbation identity,
    and the k_inner batches vary in perturbation — exactly what Reptile needs.

    Args:
        dataset:              MetadataConcatDataset
        k_inner:              Inner-loop mini-batches per outer step.
        cell_sentence_len:    Cells per mini-batch (= model's cell_set_len).
        shuffle:              Shuffle task order each epoch. Default True.
        seed:                 Base RNG seed. Incremented each epoch via set_epoch().
        h5_to_task_prefix_fn: Optional ``(h5_path: str) -> str`` — maps an h5
                              file path to a prefix prepended to cell_type to
                              form the task key.  Default: cell_type alone.
    """

    def __init__(
        self,
        dataset: MetadataConcatDataset,
        k_inner: int,
        cell_sentence_len: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        h5_to_task_prefix_fn: Optional[Callable[[str], str]] = None,
    ):
        self.dataset = dataset
        self.k_inner = k_inner
        self.cell_sentence_len = cell_sentence_len
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._h5_to_task_prefix_fn = h5_to_task_prefix_fn

        # task_key → pert_key → [global_idx, ...]
        self._task_to_pert_cells: dict[str, dict[str, list[int]]] = self._build_task_to_pert_cells()
        self._task_keys: list[str] = list(self._task_to_pert_cells.keys())

        logger.info(
            "TaskGroupedBatchSampler: %d tasks, k_inner=%d, cell_sentence_len=%d",
            len(self._task_keys),
            self.k_inner,
            self.cell_sentence_len,
        )

    # ------------------------------------------------------------------
    # Build cell pool per (task, pert)
    # ------------------------------------------------------------------

    def _build_task_to_pert_cells(self) -> dict[str, dict[str, list[int]]]:
        """Map each (task_key, pert_key) to all global cell indices."""
        task_to_pert_cells: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        global_offset = 0

        for subset in self.dataset.datasets:
            base_dataset = subset.dataset
            cache = base_dataset.metadata_cache
            h5_path = base_dataset.h5_path

            prefix = (
                self._h5_to_task_prefix_fn(h5_path)
                if self._h5_to_task_prefix_fn is not None
                else None
            )

            for local_pos, file_idx in enumerate(subset.indices):
                ct_code = int(cache.cell_type_codes[file_idx])
                ct_name = str(cache.cell_type_categories[ct_code])
                task_key = f"{prefix}/{ct_name}" if prefix is not None else ct_name

                pert_code = int(cache.pert_codes[file_idx])
                pert_key = str(cache.pert_categories[pert_code])

                task_to_pert_cells[task_key][pert_key].append(global_offset + local_pos)

            global_offset += len(subset)

        return {t: dict(p) for t, p in task_to_pert_cells.items()}

    # ------------------------------------------------------------------
    # Iterator — fresh sample every epoch
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        task_keys = list(self._task_keys)
        if self.shuffle:
            rng.shuffle(task_keys)

        for task in task_keys:
            pert_keys = list(self._task_to_pert_cells[task].keys())
            # Sample k_inner perts (with replacement if task has fewer perts than k_inner)
            chosen_perts = rng.choice(
                pert_keys,
                size=self.k_inner,
                replace=len(pert_keys) < self.k_inner,
            )
            flat: list[int] = []
            for pert in chosen_perts:
                pool = self._task_to_pert_cells[task][pert]
                flat.extend(
                    rng.choice(pool, size=self.cell_sentence_len, replace=len(pool) < self.cell_sentence_len).tolist()
                )
            yield flat

    def __len__(self) -> int:
        return len(self._task_keys)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def task_collate_fn(
    samples: List[dict],
    k_inner: int,
    exp_counts: bool = False,
) -> List[dict]:
    """
    Splits a flat list of k_inner * cell_sentence_len samples into k_inner
    collated mini-batch dicts, one per inner-loop step.

    Each returned dict has an extra key ``"task_key"`` (str) — the cell_type
    of the GRN task shared by all k_inner mini-batches.

    Args:
        samples:    Flat list delivered by DataLoader (k * cell_sentence_len).
        k_inner:    Number of inner steps (= number of output mini-batches).
        exp_counts: Passed through to PerturbationDataset.collate_fn.

    Returns:
        List[Dict[str, Tensor]] of length k_inner.
    """
    n = len(samples)
    if n % k_inner != 0:
        raise ValueError(
            f"task_collate_fn: expected {k_inner} * cells_per_batch samples, got {n}."
        )
    cells_per_inner = n // k_inner
    task_key: str = str(samples[0]["cell_type"])

    return [
        {
            **PerturbationDataset.collate_fn(
                samples[i * cells_per_inner : (i + 1) * cells_per_inner],
                exp_counts=exp_counts,
            ),
            "task_key": task_key,
        }
        for i in range(k_inner)
    ]


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class ReptileDataModule(PerturbationDataModule):
    """
    PerturbationDataModule variant whose train_dataloader() yields task batches
    for Reptile: List[Dict[str, Tensor]] of length k_inner, one per outer step.

    All PerturbationDataModule constructor arguments are accepted unchanged.
    Extra arguments:

        k_inner (int, default 10): inner-loop steps per outer update.
        h5_to_task_prefix_fn: optional callable mapping h5 path → task prefix.
    """

    def __init__(
        self,
        *args,
        k_inner: int = 10,
        h5_to_task_prefix_fn: Optional[Callable[[str], str]] = None,
        **kwargs,
    ):
        self.k_inner = k_inner
        self._h5_to_task_prefix_fn = h5_to_task_prefix_fn
        super().__init__(*args, **kwargs)

    def train_dataloader(self, test: bool = False) -> DataLoader:
        if not self.train_datasets:
            raise ValueError("No training datasets. Call setup() first.")

        ds = MetadataConcatDataset(self.train_datasets)

        sampler = TaskGroupedBatchSampler(
            dataset=ds,
            k_inner=self.k_inner,
            cell_sentence_len=self.cell_sentence_len,
            shuffle=not test,
            seed=0,
            h5_to_task_prefix_fn=self._h5_to_task_prefix_fn,
        )

        exp_counts = getattr(self, "exp_counts", False)
        collate = partial(task_collate_fn, k_inner=self.k_inner, exp_counts=exp_counts)

        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=getattr(self, "pin_memory", True),
            prefetch_factor=4 if not test and self.num_workers > 0 else None,
            persistent_workers=bool(self.num_workers > 0 and not test),
            worker_init_fn=_worker_init_fn if self.num_workers > 0 else None,
        )
