# Reptile Notes

**Paper:** "On First-Order Meta-Learning Algorithms"
**Authors:** Alex Nichol, Joshua Achiam, John Schulman (OpenAI)
**arXiv:** 1803.02999

---

## What it is

Reptile is a first-order meta-learning algorithm for learning a parameter initialization that
fine-tunes quickly on new tasks. It belongs to the same family as MAML and FOMAML but is
simpler to implement — no train/test split within each task is required.

**Serial algorithm:**

```
Initialize φ
for iteration = 1, 2, ... do
    Sample task τ
    Compute φ̃ = U^k_τ(φ)   # k steps of SGD/Adam on τ
    φ ← φ + ε(φ̃ − φ)
done
```

The update direction `(φ̃ − φ)` is treated as a gradient and can be plugged into Adam.
The natural definition of the Reptile gradient is `(φ̃ − φ) / α` where `α` is the inner-loop
stepsize.

**Batched version:** average over n tasks per outer step:

$$\phi \leftarrow \phi + \epsilon \frac{1}{n} \sum_{i=1}^{n} (\tilde{\phi}_i - \phi)$$

This is identical to **SimuParallelSGD** (Zinkevich et al.) — a communication-efficient
distributed SGD method.

---

## Relationship to MAML and FOMAML

All three algorithms optimize an initialization φ such that a few gradient steps on a
sampled task τ yields low loss.

**MAML** (full):

$$g_\text{MAML} = U'_{\tau,A}(\phi) \cdot L'_{\tau,B}(\tilde{\phi})$$

Requires differentiating through the inner-loop update (second-order derivatives).

**FOMAML** (first-order MAML): drop the Jacobian, use identity instead:

$$g_\text{FOMAML} = L'_{\tau,B}(\tilde{\phi})$$

The final gradient after k inner steps, evaluated on a **held-out** mini-batch.

**Reptile:** sum of all inner-loop gradients (no separate test split needed):

$$g_\text{Reptile} = -(\tilde{\phi} - \phi)/\alpha = \sum_{i=1}^{k} g_i$$

---

## Why it works — Taylor series analysis

**Setup:** Each task uses k mini-batches with losses $L_1, \ldots, L_k$.
Define $\bar{g}_i = L'_i(\phi_1)$ (gradient at start) and $H_i = L''_i(\phi_1)$ (Hessian).

Expanding to first order in α, the expected meta-gradient of each algorithm decomposes into
two terms:

$$\text{AvgGrad} = \mathbb{E}_\tau[\bar{g}_1]$$

This is the gradient of the expected loss — moves φ toward the joint-training minimum.

$$\text{AvgGradInner} = \mathbb{E}_{\tau,1,2}[H_2 \bar{g}_1] = \frac{1}{2}\mathbb{E}\!\left[\frac{\partial}{\partial\phi}(\bar{g}_1 \cdot \bar{g}_2)\right]$$

This is the gradient of the expected dot product between gradients from different mini-batches
on the same task. Maximizing it aligns the gradient directions across mini-batches, enabling
fast generalization to new data from the same task.

**Expected gradient coefficients (k = 2):**

| Algorithm  | AvgGrad coeff | AvgGradInner coeff |
|------------|---------------|--------------------|
| MAML       | 1             | −2α                |
| FOMAML     | 1             | −α                 |
| Reptile    | 2             | −α                 |

**General k:**

| Algorithm  | AvgGrad coeff | AvgGradInner coeff      |
|------------|---------------|-------------------------|
| MAML       | 1             | −2(k−1)α               |
| FOMAML     | 1             | −(k−1)α                |
| Reptile    | k             | −½k(k−1)α              |

All three algorithms optimize both terms. MAML puts the most weight on AvgGradInner relative
to AvgGrad (best for meta-learning); Reptile puts less (more like joint training), but the
ratio still increases with k and α.

**Key implication:** With k=1, Reptile reduces to plain SGD on the expected loss. The
meta-learning effect only emerges for k ≥ 2. More inner steps → stronger AvgGradInner signal.

---

## Geometric interpretation

Let $W_\tau$ be the manifold of parameters that perfectly solve task τ.
Reptile approximately minimizes:

$$\minimize_{\phi} \; \mathbb{E}_\tau\!\left[\tfrac{1}{2}D(\phi, W_\tau)^2\right]$$

Each iteration does a step toward the projection of φ onto $W_\tau$ (approximated by k gradient
steps). This finds a single point close to all task-specific solution manifolds — unlike
joint training which finds a manifold of solutions (e.g., $f(x)=0$ for zero-mean tasks).

---

## Mini-batch overlap: FOMAML vs Reptile

FOMAML requires the final gradient to be computed on data **not seen** in earlier inner-loop
steps. If the final batch overlaps with earlier ones, FOMAML degrades badly (the model is
near a local minimum for those samples, so their gradient has low information content).

Reptile has no this restriction — all inner-loop gradients contribute, and no separate test
split is needed.

**Practical rule:** If using FOMAML-style updates, hold out a separate mini-batch for the
final gradient. Reptile avoids this bookkeeping entirely.

---

## Transduction and batch normalization

Using batch normalization in the **transductive** setting (BN statistics computed over the
entire test batch) consistently improves accuracy for all methods. Results across papers can
differ substantially depending on whether this is used. Control for it when comparing methods.

---

## Experimental results

### Mini-ImageNet (5-way)

| Algorithm              | 1-shot       | 5-shot       |
|------------------------|--------------|--------------|
| MAML + Transduction    | 48.70 ± 1.84 | 63.11 ± 0.92 |
| FOMAML + Transduction  | 48.07 ± 1.75 | 63.15 ± 0.91 |
| Reptile                | 47.07 ± 0.26 | 62.74 ± 0.37 |
| Reptile + Transduction | 49.97 ± 0.32 | 65.99 ± 0.58 |

### Omniglot

| Algorithm              | 1-shot 5-way | 5-shot 5-way | 1-shot 20-way | 5-shot 20-way |
|------------------------|--------------|--------------|---------------|---------------|
| MAML + Transduction    | 98.7 ± 0.4   | 99.9 ± 0.1   | 95.8 ± 0.3    | 98.9 ± 0.2    |
| FOMAML + Transduction  | 98.3 ± 0.5   | 99.2 ± 0.2   | 89.4 ± 0.5    | 97.9 ± 0.1    |
| Reptile                | 95.39 ± 0.09 | 98.90 ± 0.10 | 88.14 ± 0.15  | 96.65 ± 0.33  |
| Reptile + Transduction | 97.68 ± 0.04 | 99.48 ± 0.06 | 89.43 ± 0.14  | 97.12 ± 0.32  |

All three methods reach similar accuracy; differences are within noise. The transduction boost
is larger than the difference between algorithms.

---

## Implementation details

- Inner optimizer: **Adam with β₁ = 0** (removing momentum). Using standard momentum reduces
  independence between g₁ and g₂, weakening the AvgGradInner signal. Adam's rolling moment
  data is backed up and reset for test evaluation to prevent information leakage.
- Outer optimizer: vanilla SGD (or the Reptile gradient plugged into Adam).
- Outer step size: linearly annealed to 0.
- Training shots > eval shots (e.g., 15-shot training for 1-shot eval) improves performance.
- More inner iterations help Reptile more than FOMAML (consistent with theory — higher k
  increases Reptile's AvgGradInner coefficient relative to AvgGrad).

**Omniglot hyperparameters (Reptile):**

| Parameter       | 5-way  | 20-way |
|-----------------|--------|--------|
| Adam lr         | 0.001  | 0.0005 |
| Inner batch     | 10     | 20     |
| Inner iters     | 5      | 10     |
| Training shots  | 10     | 10     |
| Outer step size | 1.0    | 1.0    |
| Outer iters     | 100K   | 200K   |
| Meta-batch size | 5      | 5      |
| Eval iters      | 50     | 50     |

**Mini-ImageNet hyperparameters (Reptile, both 1-shot and 5-shot):**
Adam lr=0.001, inner batch=10, inner iters=8, training shots=15, outer step=1.0, outer
iters=100K, meta-batch=5, eval iters=50.

---

## Relevance to pre-pre-training

The Reptile analysis suggests that training on a distribution of tasks (rather than joint
training) produces initializations that maximize gradient alignment across mini-batches within
each task — exactly the property that enables fast fine-tuning. Pre-pre-training on SERGIO
synthetic GRNs is doing something analogous: the distribution of GRN seeds plays the role of
the task distribution, and fine-tuning on a target dataset plays the role of inner-loop
adaptation. The AvgGradInner logic predicts the pre-pre-trained initialization should
generalize better in the low-data regime compared to training from scratch (which would not
have this alignment).
