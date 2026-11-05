## Overview  

`local_ball_ivr` (and its helper `_local_ball_ivr_one`) implements **Integrated‑Variance‑Reduction (IVR)** for a Gaussian‑process (GP) surrogate.  
IVR is a *Monte‑Carlo* approximation of the expected decrease in the GP’s posterior variance that would result from **adding a whole batch of candidate points** to the training set.  

The algorithm is deliberately built to handle **mixed‑type inputs**:

* **Continuous dimensions** – e.g. real‑valued physical parameters.  
* **Categorical / fingerprint dimensions** – binary (or integer) bits that encode a molecule, a graph, a configuration, etc.

Below we walk through the code block‑by‑block, explain the *mathematical* idea, and then focus on the **continuous vs categorical handling**.

---

## 1.  Stable Cholesky & (optional) inverse  

```python
def stable_cholesky_and_inverse(K, jitter_factor=1e-9, max_tries=5):
    jitter = jitter_factor * np.mean(np.diag(K))
    for _ in range(max_tries):
        try:
            K_j = K + np.eye(K.shape[0]) * jitter
            L   = cholesky(K_j, lower=True, check_finite=False)
            iL  = solve_triangular(L, np.eye(K.shape[0]), lower=True,
                                   check_finite=False)
            K_inv = iL.T @ iL
            return L, K_inv, True
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return None, None, False
```

* **Why it exists** – a GP covariance matrix must be **positive‑definite** (PD) to admit a Cholesky factor. Numerical round‑off, duplicate points, or very small length‑scales can make the matrix *nearly* singular; adding a tiny **jitter** (`ε·I`) forces PD-ness.  

* **What it returns** – the lower‑triangular factor `L`, the explicit inverse `K_inv` (computed via triangular solves, which is numerically more stable than `np.linalg.inv`), and a `success` flag.  

* **Use in the algorithm** –  
  * `K_train_noisy` (training Gram matrix + observation noise) → `K_inv_train`.  
  * `K_aug` (training + candidate batch) → `L_aug` (the factor needed for the posterior update after adding the batch).

---

## 2.  The Numba kernel – fast IVR for one Monte‑Carlo sample set  

```python
@njit(fastmath=True)                 # or the parallel version
def _numba_calc_group_ivr_generic(samples, X_train, X_aug,
                                  L_aug, K_inv_train,
                                  hps, kernel_func, signal_var):
    # 1) variance using the *old* training set
    k_s_train = kernel_func(samples, X_train, hps)      # (S, N)
    term1     = k_s_train @ K_inv_train                 # (S, N)

    # 2) variance using the *augmented* set (train + candidates)
    k_s_aug   = kernel_func(samples, X_aug, hps)        # (S, N+M)
    Y         = np.linalg.solve(L_aug, k_s_aug.T)       # (N+M, S)

    # 3) compute per‑sample reduction, keep only positive parts
    total_diff = 0.0
    for i in range(S):
        # variance before adding the batch
        var_current = signal_var - np.dot(term1[i, :], k_s_train[i, :])

        # variance after adding the batch
        var_after   = signal_var - np.sum(Y[:, i] ** 2)

        diff = var_current - var_after
        if diff > 0.0:
            total_diff += diff
    return total_diff
```

**Mathematical expression**  

For a single Monte‑Carlo point `z`:

* **Old variance**  

  $
  \sigma_{\text{old}}^2(z)=\sigma_f^2-
  k(z,X_{\text{train}})\,K_{\text{train}}^{-1}\,k(z,X_{\text{train}})^{\top}
  $

* **New variance after appending the candidate batch**  

  $
  \sigma_{\text{new}}^2(z)=\sigma_f^2-
  k(z,X_{\text{aug}})\,L_{\text{aug}}^{-T}L_{\text{aug}}^{-1}\,
  k(z,X_{\text{aug}})^{\top} .
  $

* **Variance reduction**  

  $
  \Delta(z)=\max\bigl(\sigma_{\text{old}}^2(z)-\sigma_{\text{new}}^2(z),\,0\bigr).
  $

The kernel function `kernel_func` is a **joint kernel** that evaluates

$
k\bigl(x,y\bigr)=\sigma_f^2\;
k_{\text{Matern}}^{\text{cont}}(x_{\text{cont}},y_{\text{cont}})
\;
\bigl[w\,k_{\text{Tanimoto}}(x_{\text{disc}},y_{\text{disc}})
      + (1-w)\bigr],
$

so the *same* routine automatically accounts for both continuous and discrete dimensions.

The Numba functions are deliberately loop‑heavy (matrix‑vector products are done once, the rest is a series of dot‑products) because they need to be evaluated **thousands of times** (one MC draw per group, per batch). The parallel variant simply runs the outermost `for i in prange(S)` across threads.

---

## 3.  `_local_ball_ivr_one` – the *single‑batch* driver  

Below is the logical flow, with emphasis on where continuous and categorical dimensions appear.

### 3.1  Cache handling  

* The **training‑only** inverse `K_inv_train` (and the noisy Gram matrix) are cached in a mutable dict (`cache`).  
  *If the training set size or hyper‑parameters haven’t changed*, the expensive Cholesky+inverse is **re‑used**.  

### 3.2  Build the augmented system (train + candidates)  

```python
K_cand   = kernel_func(x, x, hps) + np.eye(n_cand) * hps[noise_index]
K_cross  = kernel_func(X_train, x, hps)          # (N, M)
K_aug    = np.block([[K_train_noisy, K_cross],
                     [K_cross.T,      K_cand]])
L_aug, _, success_aug = stable_cholesky_and_inverse(K_aug)
X_aug = np.vstack([X_train, x])
```

* `x` contains **both** continuous and fingerprint columns.  
* The joint kernel produces a full covariance matrix that already incorporates the *interaction* between the two types of features. No special handling is required here; it is delegated to `kernel_func`.

### 3.3  Grouping candidates – **continuous distance + discrete penalty**  

```python
# continuous part
X_cont = x[:, continuous_indices]                # (M, n_cont)
diff   = X_cont[:, None, :] - X_cont[None, :, :] # (M, M, n_cont)
safe_radii = np.maximum(ls_vals * radius_multiplier, 1e-6)
distances_scaled = np.linalg.norm(diff / safe_radii, axis=2)

# discrete part – infinite distance if any fingerprint bit differs
all_dims = np.arange(x.shape[1])
discrete_indices = np.setdiff1d(all_dims, continuous_indices)
if discrete_indices.size:
    X_disc = x[:, discrete_indices]
    disc_mismatch = (X_disc[:, None, :] != X_disc[None, :, :]).any(axis=2)
    distances_scaled[disc_mismatch] = np.inf

# connected components -> groups
n_groups, labels = connected_components(distances_scaled < 2.0,
                                        directed=False)
```

* **Continuous distance** – scaled by the *anisotropic* radii `cont_radii = lengthscale * radius_multiplier`.  
  If two points are within two “scaled radii” they are considered neighbours (the threshold `< 2.0` corresponds to “within one radius” after scaling).  

* **Categorical / fingerprint dimensions** – the code builds a binary mask `disc_mismatch` that is **True** whenever *any* fingerprint bit differs.  
  The line  

  ```python
  distances_scaled[disc_mismatch] = np.inf
  ```  

  forces those pairs to have *infinite* scaled distance, guaranteeing they **never belong to the same connected component**.  

> **Result:** each group contains points that share exactly the same fingerprint (or have no fingerprint at all) **and** are close in the continuous sub‑space.

### 3.4  Monte‑Carlo sampling *inside* each group  

For a **single‑point group** (only one candidate with a given fingerprint) we draw uniform points inside an *n‑dimensional hypersphere* (the “local ball”).

```python
direction = rng.randn(S, n_cont)
direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
radius    = rng.uniform(0., 1., S) ** (1.0 / n_cont)
perturb   = direction * (radius[:, None] * cont_radii)
samples   = np.tile(centre_ref, (S, 1))
samples[:, continuous_indices] += perturb
```

* `centre_ref` is the **full** candidate vector (continuous + fingerprint).  
* **Only the continuous columns are perturbed**; the fingerprint columns stay identical to `centre_ref`.  

For a **multi‑point group** we use a **bounding hyper‑rectangle** that encloses *all* points of the group, again sampling **only the continuous columns**:

```python
lo_c = np.min(group_pts[:, continuous_indices] - cont_radii, axis=0)
hi_c = np.max(group_pts[:, continuous_indices] + cont_radii, axis=0)

samples = np.tile(centre_ref, (S, 1))
samples[:, continuous_indices] = rng.uniform(lo_c, hi_c,
                                            size=(S, n_cont))
```

If there are **no continuous dimensions** (edge case) the code simply copies the centre row `S` times – the MC points are exactly the same as the candidate.

**Why keep the fingerprint unchanged?**  
Because the *group* was already formed on the basis of **identical fingerprints**. Perturbing them would immediately push the points into a different component (the infinite‑distance penalty would fire). By leaving the fingerprint untouched we guarantee that *every* Monte‑Carlo draw has the same discrete part that the kernel receives, i.e. the Tanimoto similarity term evaluates to **1** for every draw, and the IVR reduction is driven only by the continuous perturbations.

### 3.5  Enforce user‑provided bounds on the continuous dimensions  

```python
cont_samples = samples[:, continuous_indices]
inside = np.all((cont_samples >= bounds[:,0]) &
                (cont_samples <= bounds[:,1]), axis=1)
valid = samples[inside]
```

If too few points survive the bound check the group is skipped (`continue`). This step never touches the fingerprint columns because they are binary (0/1) and already satisfy any conceivable “bounds” (they are not part of `bounds`).

### 3.6  Call the Numba IVR evaluator  

```python
group_sum = _numba_calc_group_ivr(valid, X_train, X_aug, L_aug,
                                 K_inv_train, hps, kernel_func,
                                 sigma_f2)
total_ivr += group_sum / n_samples_per_group
```

`valid` contains a **full‑dimensional** matrix (continuous + fingerprint).  
`kernel_func` receives those rows and computes the joint kernel **exactly** as defined elsewhere (Matern × Tanimoto × signal). Because the fingerprint part never changes inside a group, the Tanimoto contribution is a constant factor for that group, and the variance reduction comes solely from the continuous perturbations.

Finally the function returns a *single* scalar `total_ivr` – the estimated integrated variance reduction for the candidate set `x`.

---

## 4.  `local_ball_ivr` – batch wrapper  

The outer function normalises the input so that it can accept:

* a single candidate set `(n_cand, d)` → treated as a batch of size 1, **or**
* a true batch `(n_batch, n_cand, d)`.

For each batch it creates an independent RNG (deterministic if requested) and calls `_local_ball_ivr_one`.  
A shared mutable dict `cache` is passed down so that **all batches reuse the same training‑only Cholesky factor**, drastically reducing repeated work.

The output is an array of IVR values of shape `(n_batches,)`.  

---

## 5.  How continuous **and** categorical dimensions are handled together  

| Stage | Continuous dims (`continuous_indices`) | Categorical / fingerprint dims (`discrete_indices`) |
|-------|----------------------------------------|---------------------------------------------------|
| **Distance / grouping** | Scaled Euclidean distance using per‑dim radii (`cont_radii`). | If **any** bit differs → distance set to `∞`. This forces different fingerprints into **different connected components**. |
| **Monte‑Carlo sampling** | Random draws inside a hypersphere (single‑point groups) or a hyper‑rectangle (multi‑point groups). Radius is anisotropic (`cont_radii`). | The fingerprint columns are **copied unchanged** from the group centre (`np.tile(centre_ref, ...)`). No randomisation of bits. |
| **Bound enforcement** | The samples are filtered against user‑supplied continuous bounds (`bounds`). | No bounds needed – bits are already 0/1 (or whatever the user encoded). |
| **Kernel evaluation** | The **Matern‑3/2** block of the joint kernel sees the perturbed continuous coordinates. | The **Tanimoto** block sees the **identical fingerprint** for all MC points in a group, so its value is *constant* (usually 1) within the group. |
| **Variance reduction** | The reduction comes from the *continuous* perturbations – the GP becomes more certain in regions of the continuous space that are covered by the MC points. | Because the fingerprint never changes within a group, the contribution of the discrete block does **not** affect the *difference* between `σ_old` and `σ_new`. It still influences the **absolute scale** of the covariance through the `signal_var * k_cont * k_disc` product, but the *change* is driven by the continuous part only. |

Thus the algorithm is **exactly** what you would expect for a mixed‑type GP:

1. **Group** points that share the same categorical description (fingerprint).  
2. **Explore locally** in the continuous sub‑space around each group centre.  
3. **Measure** how much the continuous exploration shrinks the posterior variance, while the categorical part stays fixed.

---

## 6.  Why this design is efficient  

* **Caching** (`K_inv_train`) avoids an O(N³) Cholesky on the *training* data for every batch.  
* **Single Cholesky on the augmented matrix** (`K_aug`) is O((N+M)³) but performed once per candidate set, not per MC draw.  
* **Numba‑accelerated IVR kernel** works on *matrix–vector* products only, not on full Gram matrices for each MC point. The expensive kernel evaluations are vectorised (`kernel_func(samples, X_train, …)`).  
* **Parallel reduction** (`prange`) spreads the MC samples across cores.  
* **Connected‑components** reduces the number of MC draws dramatically: only points that are close *and* share the same fingerprint are grouped; the rest are treated independently.

---

## 7.  Putting it all together (high‑level picture)

```
Given:
    X_train (N × d)      -- training data (continuous + fingerprint)
    x      (M × d)      -- candidate batch
    hps    (vector)      -- GP hyper‑parameters
    bounds (C × 2)       -- lower/upper bounds for continuous dims

1️⃣ Build training‑only inverse (cached).
2️⃣ Build augmented covariance K_aug (train + candidates) → L_aug.
3️⃣ Group candidates:
      • use scaled Euclidean distance on continuous dims,
      • add infinite penalty for any fingerprint mismatch.
4️⃣ For each group:
      • draw S Monte‑Carlo points **only** in the continuous sub‑space,
        keep fingerprint identical to the group centre.
      • discard points outside user’s continuous bounds.
      • evaluate variance reduction for the surviving points with
        the Numba kernel (uses kernel_func that already mixes
        Matern and Tanimoto).
5️⃣ Sum the reductions over all groups → IVR(x).

Return IVR for each batch.
```

The **joint kernel** (`kernel_func`) is the only place where the continuous and categorical dimensions *interact* mathematically. All the surrounding machinery (grouping, sampling, bounds) simply respects the **binary nature** of the fingerprints: they are never “perturbed”, they are used only to decide *which points belong together*.

---

## 8.  Quick checklist for developers  

| ✅ | Item |
|----|------|
| **Distance metric** | Scaled Euclidean on continuous + `∞` penalty for discrete mismatches. |
| **Group formation** | `connected_components(distances_scaled < 2.0)`. |
| **Monte‑Carlo sampling** | `samples = np.tile(centre_ref,…)` then modify only `continuous_indices`. |
| **Kernel** | Must accept full‑dimensional vectors and internally compute `k_Matern * k_Tanimoto`. |
| **Cache** | `K_inv_train` stored in mutable dict; be sure to copy hyper‑parameters (`hps.copy()`) to detect changes correctly. |
| **Bounds** | Applied **after** sampling, only to continuous columns. |
| **Parallel option** | Use `_numba_calc_group_ivr_generic_parallel` for large `S`. |
| **Determinism** | Pass a seeded `rng` to each batch (or use `deterministic=True`). |
| **Edge cases** | If `len(continuous_indices) == 0`, the algorithm simply copies the centre; if `len(discrete_indices) == 0` the `∞` penalty is skipped. |

---

### TL;DR  

`local_ball_ivr` evaluates the *expected* reduction in GP posterior variance after adding a batch of candidates.  
It does this by:

1. **Grouping** candidates that share the same fingerprint (categorical) **and** are close in the continuous space (scaled Euclidean).  
2. **Sampling** Monte‑Carlo points **only** in the continuous dimensions inside a local ball around each group centre; the fingerprint part stays fixed.  
3. **Computing** the variance reduction for those points with a fast Numba kernel that uses the **joint kernel** (`signal × Matern × Tanimoto`).  

Thus continuous and categorical features are treated **jointly but separately**: the categorical part is only used to split the space into disjoint groups, while the continuous part drives the local exploration and the variance‑reduction estimate. This design yields a scalable IVR computation that works for high‑dimensional mixed‑type data.