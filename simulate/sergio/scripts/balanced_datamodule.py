"""
balanced_datamodule.py — PerturbationDataModule subclass with balanced per-perturbation subsampling.

Adds a `cells_per_pert` parameter that, after the standard setup, trims each training
Subset to keep at most N cells per (cell_type, perturbation) group.  Control cells are
kept in full (they are shared across perturbations and trimming them would break basal
mapping).

Usage
-----
from balanced_datamodule import BalancedPerturbationDataModule

dm = BalancedPerturbationDataModule(
    cells_per_pert=50,
    toml_config_path=...,
    ...all other PerturbationDataModule kwargs...
)
dm.setup(stage="fit")
"""

import numpy as np
from cell_load.data_modules.perturbation_dataloader import PerturbationDataModule


class BalancedPerturbationDataModule(PerturbationDataModule):
    """
    Wraps PerturbationDataModule with balanced per-perturbation cell subsampling.

    Args:
        cells_per_pert: Max cells to keep per (cell_type, perturbation) group in the
            training split.  Control cells are never trimmed.  Set to None to disable
            (identical to base class behaviour).
        random_seed: Passed to the base class; also used for reproducible subsampling.
        **kwargs: All other PerturbationDataModule keyword arguments.
    """

    def __init__(self, cells_per_pert: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if cells_per_pert is not None and (
            not isinstance(cells_per_pert, int) or cells_per_pert <= 0
        ):
            raise ValueError("cells_per_pert must be a positive int or None.")
        self.cells_per_pert = cells_per_pert

    def setup(self, stage: str | None = None):
        super().setup(stage=stage)
        if self.cells_per_pert is not None and stage in ("fit", None):
            self._apply_balanced_subsample()

    def _apply_balanced_subsample(self):
        rng = np.random.default_rng(self.random_seed)
        total_before = sum(len(s.indices) for s in self.train_datasets)
        total_after = 0

        for subset in self.train_datasets:
            ds = subset.dataset
            cache = ds.metadata_cache
            indices = np.asarray(subset.indices)

            # Identify control cells — keep them all
            ctrl_code = cache.control_pert_code
            pert_codes = cache.pert_codes[indices]
            is_ctrl = pert_codes == ctrl_code
            ctrl_idx = indices[is_ctrl]
            pert_idx = indices[~is_ctrl]

            if len(pert_idx) == 0:
                total_after += len(subset.indices)
                continue

            # Group perturbed cells by (cell_type, pert) and subsample
            ct_codes  = cache.cell_type_codes[pert_idx]
            pt_codes  = cache.pert_codes[pert_idx]
            kept = []
            for ct in np.unique(ct_codes):
                ct_mask = ct_codes == ct
                for pt in np.unique(pt_codes[ct_mask]):
                    pt_mask = ct_mask & (pt_codes == pt)
                    group = pert_idx[pt_mask]
                    if len(group) > self.cells_per_pert:
                        group = rng.choice(group, size=self.cells_per_pert, replace=False)
                    kept.append(group)

            kept_pert = np.concatenate(kept) if kept else np.array([], dtype=np.int64)
            new_indices = np.concatenate([kept_pert, ctrl_idx])
            subset.indices = new_indices
            total_after += len(new_indices)

        print(
            f"[BalancedPerturbationDataModule] cells_per_pert={self.cells_per_pert}: "
            f"{total_before:,} → {total_after:,} train cells "
            f"({100 * total_after / total_before:.1f}%)"
        )
