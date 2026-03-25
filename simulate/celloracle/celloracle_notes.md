# CellOracle — Technical Notes

---

## Stochasticity in Simulations

### Perturbed expression (`simulate_shift`) — Deterministic

The core perturbation is fully deterministic. It works via iterative matrix multiplication through Ridge regression weights:

```python
# celloracle/trajectory/oracle_GRN.py, lines 38–44
for i in range(n_propagation):
    delta_simulated = delta_simulated.dot(coef_matrix)         # L39
    delta_simulated[delta_input != 0] = delta_input            # L40 — pin perturbed genes
    gem_tmp = gem + delta_simulated
    gem_tmp[gem_tmp < 0] = 0                                   # L43 — non-negativity clamp
    delta_simulated = gem_tmp - gem
```

- `coef_matrix`: Ridge regression weights (TF → target gene), fit with `Ridge(random_state=123)`
  — `celloracle/trajectory/oracle_GRN.py:79`
- `n_propagation=3` default: not time steps — propagates the signal up to 3 hops through the GRN
  — `celloracle/trajectory/oracle_core.py:492` (`simulate_shift`) and `:512` (`__simulate_shift`)
- Perturbed genes are pinned at their forced value at every step (`oracle_GRN.py:40`)
- **No noise, no sampling.** Running the same perturbation twice gives identical results.

The downstream visualization step (`estimate_transition_prob`) uses `np.random.seed(15071990)` for neighbor subsampling — reproducible but technically stochastic. Not relevant for generating training data.
- Default declared: `celloracle/trajectory/modified_VelocytoLoom_class.py:252`
- Applied: `modified_VelocytoLoom_class.py:271` (`np.random.seed(random_seed)` inside `if knn_random:`)

---

### Unperturbed expression — Not simulated; derived from real data

CellOracle does **not** have a generative model for unperturbed expression. The control state comes from real scRNA-seq cells that are KNN-imputed when the Oracle object is initialized. The imputed counts are stored as `adata.layers["imputed_count"]`:

```python
# celloracle/trajectory/modified_VelocytoLoom_class.py:244  (inside knn_imputation())
self.adata.layers["imputed_count"] = Xx.transpose().copy()
```

`Xx` is the result of `convolve_by_sparse_weights(X, self.knn_smoothing_w)` — a sparse-weighted average of neighboring cells' expression. All downstream simulation uses this layer as the base expression `gem`.

**Implication for `generate_dataset.py`:** You cannot sample new unperturbed cells from scratch the way SERGIO does. The unperturbed expression vectors must come from the real cells in the Oracle object. Each real cell is a fixed control vector — there is no stochasticity and no way to generate novel control samples without real data.

To produce multiple control cells per GRN (needed for training), options are:
1. Use the real KNN-imputed cells directly (fixed, finite set)
2. Add technical noise post-hoc (e.g. Gaussian or negative binomial) to simulate cell-to-cell variability
3. Use multiple real cells from the same cluster as separate starting points and propagate each through the GRN for perturbation

---

## Summary

| Step | Stochastic? | Source | Notes |
|------|-------------|--------|-------|
| GRN fitting (Ridge regression) | No | `oracle_GRN.py:79` | `Ridge(random_state=123)` |
| `simulate_shift` propagation loop | No | `oracle_GRN.py:38–44` | Pure matrix multiply, no noise |
| `n_propagation=3` default | — | `oracle_core.py:492,512` | 3 hops, not time steps |
| `estimate_transition_prob` | Reproducible | `modified_VelocytoLoom_class.py:252,271` | Fixed seed `15071990` |
| Unperturbed expression (KNN imputation) | No | `modified_VelocytoLoom_class.py:244` | Fixed from real cells |
