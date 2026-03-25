"""
balanced.py — BalancedPerturbationDataModule for sample-efficiency experiments.

Subclasses PerturbationDataModule and trims training subsets to at most
`cells_per_pert` perturbed cells per (cell_type, perturbation) group after
the standard setup.  Control cells are kept in full.

Register in DATA_MODULE_DICT as "BalancedPerturbationDataModule" so that
`state tx train data.name=BalancedPerturbationDataModule data.kwargs.cells_per_pert=N`
works out of the box.
"""

import numpy as np
from cell_load.data_modules.perturbation_dataloader import PerturbationDataModule


class BalancedPerturbationDataModule(PerturbationDataModule):
    """
    PerturbationDataModule with balanced per-perturbation cell subsampling.

    After the standard setup(), trims each training Subset to keep at most
    `cells_per_pert` perturbed cells per (cell_type, perturbation) group.
    Control cells are never trimmed.

    Args:
        cells_per_pert: Max perturbed cells per (cell_type, perturbation) group
            in the training split.  None disables subsampling (identical to base
            class behaviour).
        **kwargs: All PerturbationDataModule keyword arguments.
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

            ctrl_code = cache.control_pert_code
            pert_codes = cache.pert_codes[indices]
            is_ctrl = pert_codes == ctrl_code
            ctrl_idx = indices[is_ctrl]
            pert_idx = indices[~is_ctrl]

            if len(pert_idx) == 0:
                total_after += len(subset.indices)
                continue

            ct_codes = cache.cell_type_codes[pert_idx]
            pt_codes = cache.pert_codes[pert_idx]
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
