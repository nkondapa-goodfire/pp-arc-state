"""
reptile.py — Task-grouped sampling and data module for Reptile meta-learning.

TaskGroupedBatchSampler
-----------------------
Subclasses PerturbationBatchSampler. Reuses its sentence-building logic
(sentences are lists of cell_sentence_len global indices, all from the same
(cell_type, pert) group). Then groups those sentences by cell_type, so that
each outer step yields k * cell_sentence_len indices all from the same GRN task.

Each call to __iter__ yields one flat list of k * cell_sentence_len indices.
The DataLoader delivers this list as k * cell_sentence_len individual samples,
which task_collate_fn splits into k collated mini-batches.

task_collate_fn
---------------
Receives the flat list of k * cell_sentence_len samples from the DataLoader.
Splits into k groups of cell_sentence_len, collates each with
PerturbationDataset.collate_fn, returns List[Dict[str, Tensor]] of length k.
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

import bisect
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np
from torch.utils.data import DataLoader

from cell_load.data_modules.perturbation_dataloader import (
    PerturbationDataModule,
    _worker_init_fn,
)
from cell_load.data_modules.samplers import PerturbationBatchSampler
from cell_load.dataset._perturbation import PerturbationDataset
from cell_load.dataset._metadata import MetadataConcatDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class TaskGroupedBatchSampler(PerturbationBatchSampler):
    """
    Yields batches of k_inner * cell_sentence_len indices, all from the same
    cell_type (GRN task). Each group of cell_sentence_len indices becomes one
    inner-loop mini-batch after task_collate_fn processes the flat list.

    Sentences (groups of cell_sentence_len indices sharing the same
    (cell_type, pert)) are built by the parent class. They are then regrouped
    by task key, so k_inner sentences drawn from the same task vary in
    perturbation identity — exactly the inner-loop variation Reptile needs.

    The task key combines a per-h5-file prefix with the cell_type name.
    By default the prefix is the h5 filename stem (so each file is its own
    task namespace). Pass ``h5_to_task_prefix_fn`` to extract a coarser
    grouping — e.g., for SERGIO, extract (grn_type, grn_size) so that cells
    from the same GRN topology but different noise levels / pert types all
    contribute to the same task pool.

    Args:
        dataset:              MetadataConcatDataset
        k_inner:              Inner-loop mini-batches per outer step.
        cell_sentence_len:    Cells per mini-batch (= model's cell_set_len).
        shuffle:              Shuffle task order each epoch. Default True.
        seed:                 Base RNG seed. Epoch offset added each iteration.
        h5_to_task_prefix_fn: ``(h5_path: str) -> str`` — maps an h5 file path
                              to a string prefix that identifies which group of
                              files belongs to the same task. Files with the same
                              prefix AND the same cell_type are merged into one
                              task. Default: use the full filename stem (each
                              file is its own prefix, giving fine-grained tasks).
        **kwargs:             Forwarded to PerturbationBatchSampler (drop_last,
                              use_batch, downsample_cells, …).
    """

    def __init__(
        self,
        dataset: MetadataConcatDataset,
        k_inner: int,
        cell_sentence_len: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        h5_to_task_prefix_fn: Optional[Callable[[str], str]] = None,
        **kwargs,
    ):
        self.k_inner = k_inner
        self.shuffle = shuffle
        # Default: no prefix — cell_type alone is the task key.
        # Pass a function here when cell_type is ambiguous across files
        # (e.g., when cell_type encodes only a seed and the file name carries
        # graph-type / graph-size information).
        self._h5_to_task_prefix_fn = h5_to_task_prefix_fn

        # Parent builds self.sentences and self.metadata_caches.
        # batch_size is unused in our __iter__ but required by parent __init__.
        super().__init__(
            dataset=dataset,
            batch_size=k_inner,
            cell_sentence_len=cell_sentence_len,
            seed=seed,
            **kwargs,
        )

        # Group sentences by compound task key.
        self._task_to_sentences: dict[str, list[int]] = self._group_sentences_by_task()
        self._task_keys: list[str] = list(self._task_to_sentences.keys())
        logger.info(
            "TaskGroupedBatchSampler: %d tasks, %d total sentences, k_inner=%d",
            len(self._task_keys),
            len(self.sentences),
            self.k_inner,
        )

    # ------------------------------------------------------------------
    # Sentence → task mapping
    # ------------------------------------------------------------------

    def _group_sentences_by_task(self) -> dict[str, list[int]]:
        """Return {task_key: [sentence_idx, ...]} for all sentences."""
        from collections import defaultdict

        task_to_sentences: dict = defaultdict(list)
        for sent_idx, sentence in enumerate(self.sentences):
            task_key = self._task_key_for_global(sentence[0])
            task_to_sentences[task_key].append(sent_idx)
        return dict(task_to_sentences)

    def _task_key_for_global(self, global_idx: int) -> str:
        """
        Resolve a global DataLoader index to a string task key.

        Task key = h5_task_prefix + "/" + cell_type_name, where:
          - h5_task_prefix is derived from the h5 file path via
            self._h5_to_task_prefix_fn (caller-supplied; default = file stem)
          - cell_type_name is the string category for this cell's cell_type code

        Global indices are laid out as:
          subset_0: [0 .. len(subset_0)-1]
          subset_1: [len(subset_0) .. len(subset_0)+len(subset_1)-1]
          ...
        cumulative_sizes[i] == sum(len(subset_j) for j <= i).
        """
        cumulative = self.dataset.cumulative_sizes
        ds_idx = bisect.bisect_right(cumulative, global_idx)
        local_idx = global_idx - (cumulative[ds_idx - 1] if ds_idx > 0 else 0)

        subset = self.dataset.datasets[ds_idx]
        file_idx = int(subset.indices[local_idx])
        h5_path = subset.dataset.h5_path
        cache = self.metadata_caches[h5_path]

        ct_code = int(cache.cell_type_codes[file_idx])
        ct_name = str(cache.cell_type_categories[ct_code])
        if self._h5_to_task_prefix_fn is not None:
            prefix = self._h5_to_task_prefix_fn(h5_path)
            return f"{prefix}/{ct_name}"
        return ct_name

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        task_keys = list(self._task_keys)
        if self.shuffle:
            rng.shuffle(task_keys)

        for task in task_keys:
            sent_indices = self._task_to_sentences[task]
            n_available = len(sent_indices)
            replace = n_available < self.k_inner
            if replace:
                logger.debug(
                    "Task %d has only %d sentences but k_inner=%d; sampling with replacement.",
                    task,
                    n_available,
                    self.k_inner,
                )
            chosen = rng.choice(sent_indices, size=self.k_inner, replace=replace).tolist()
            yield [idx for i in chosen for idx in self.sentences[i]]

    def __len__(self) -> int:
        return len(self._task_keys)


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
    of the GRN task shared by all k_inner mini-batches. This is the same value
    as ``batch["cell_type"][0]`` but promoted to a top-level string for easy
    logging and verification.

    Args:
        samples:    Flat list delivered by DataLoader (k * cell_sentence_len).
        k_inner:    Number of inner steps (= number of output mini-batches).
        exp_counts: Passed through to PerturbationDataset.collate_fn.

    Returns:
        List[Dict[str, Tensor]] of length k_inner. Each dict is a fully
        collated mini-batch with shape (cell_sentence_len, feature_dim) tensors,
        plus ``"task_key": str``.
    """
    n = len(samples)
    if n % k_inner != 0:
        raise ValueError(
            f"task_collate_fn: expected {k_inner} * cells_per_batch samples, got {n}."
        )
    cells_per_inner = n // k_inner

    # task_key = cell_type of the shared GRN task.
    # With the updated SERGIO dataset, cell_type already encodes the full
    # task identity (e.g. "BA-VM_size010_seed0000").
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
    One extra argument:

        k_inner (int, default 10): inner-loop steps per outer update.
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
            drop_last=self.drop_last,
            use_batch=self.basal_mapping_strategy == "batch",
            downsample_cells=getattr(self, "downsample_cells", None),
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
