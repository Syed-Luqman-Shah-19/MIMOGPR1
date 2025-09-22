# Low-Overhead CSI Prediction via Gaussian Process Regression

**Notebook:** `Syed_IEEE_Letter.ipynb`
**Paper target:** IEEE WCL letter (GPR with SpatialCorrelation kernel)

This repository contains a **single, self-contained notebook** that reproduces the main results of the paper: channel generation (Kronecker & Weichselberger), end-to-end **GPR** reconstruction with both **generic kernels** (RBF, Matérn, RationalQuadratic + WhiteKernel) and the **SpatialCorrelation** kernel (exact complex LMMSE from $\mathbf R_H$), and the **LS/MMSE** benchmarks.

We save all Monte-Carlo runs to a compact **HDF5** file (one file per experiment) and then load that file to draw all figures in the paper, plus additional propriety/independence diagnostics for complex MIMO channels.

---

## Environment

* Python ≥ 3.9
* NumPy ≥ 1.23
* SciPy ≥ 1.9
* scikit-learn ≥ 1.2 (for RBF/Matérn/RQ kernels)
* h5py ≥ 3.7
* Matplotlib ≥ 3.6

Optional (for independence diagnostics, if you want faster kernels):

* `numba` (speeds up dCor / HSIC approximations)

> Tip: create a conda env
>
> ```bash
> conda create -n gpr-mimo python=3.10 numpy scipy scikit-learn h5py matplotlib numba -y
> conda activate gpr-mimo
> ```

---

## Data layout (HDF5)

We adopt a **compact HDF5 layout with a Monte-Carlo (MC) axis**, so a single file stores everything needed for all figures.

Top-level groups per **model**:

* `/Kronecker`
* `/Weichselberger`

Within each model:

* `/Original/H` : shape `(MC, Nr, Nt)` — complex64/128
* `/Original/R_H` : shape `(MC, Nrt, Nrt)` — complex64/128, where `Nrt = Nr*Nt`
* For each **Case** ∈ {`CaseI`, `CaseII`, `CaseIII`} and **Kernel** ∈ {`RBF`, `Matern`, `RationalQuadratic`, `WhiteKernel`, `SpatialCorrelation`}:

  * `/CaseI/SpatialCorrelation/H` : `(MC, Nr, Nt)` — complex
  * `/CaseI/SpatialCorrelation/std` : `(MC, Nr, Nt)` — float
  * … (same paths for all kernels, all cases)

**Important conventions**

* **Complex fields** are stored as native HDF5 complex (h5py supports complex dtypes).
* **Posterior std** is the **per-entry** standard deviation from the GP posterior on the **complex field’s real/imag parts modeled independently** (i.e., we compute posterior var for Re and for Im from the same real kernel and then combine: `std_complex = sqrt(std_Re^2 + std_Im^2)`), or for the SpatialCorrelation kernel we use the **diagonal** of the posterior covariance of `vec(H)` reshaped to `(Nr, Nt)` and again combine Re/Im appropriately.
* We **print NMSE** summaries to stdout during generation as:

  ```
  [MC i/MC] Model=Weichselberger, Case=CaseII, Kernel=SpatialCorrelation, NMSE(dB)=...
  ```

---

## Notebook outline

The notebook is organized in **five executable cells** (plus a last diagnostics cell). Each cell is fully documented and can be run independently if the HDF5 already exists.

### Cell 1 — Generate `.hdf5` (1000 MC)

**What it does**

1. Sets the simulation parameters (`Nr=Nt=36`, carrier, array geometries, angle spread, seeds).
2. Generates `MC=1000` channels for both **Kronecker** and **Weichselberger** models:

   * Saves `/Model/Original/H` and the exact **second-order statistics** $\mathbf R_H = \mathbb E[ \mathrm{vec}(\mathbf H)\, \mathrm{vec}(\mathbf H)^{\mathrm H}]$ as `/Model/Original/R_H` (one per MC).

     * Kronecker: $\mathbf R_H = \mathbf R_t \otimes \mathbf R_r$.
     * Weichselberger: $\mathbf R_H = (\mathbf U_t^* \otimes \mathbf U_r)\,\mathrm{diag}(\mathrm{vec}(\Omega))\,(\mathbf U_t^{\top} \otimes \mathbf U_r^{\mathrm H})$.
3. Defines **probing Cases**:

   * **Case I:** first TX column only: $\{(r,0)\}$
   * **Case II:** equispaced half array: $\{(r,t): t\in\{0,2,4,\dots\}\}$
   * **Case III:** main diagonal: $\{(i,i)\}$
4. Runs **GPR** for each **Kernel**:

   * **RBF (2D isotropic)** on coordinates $(r,t)$
   * **Matérn (2D isotropic)** (e.g., $\nu=3/2$)
   * **RationalQuadratic (2D isotropic)**
   * **WhiteKernel** (baseline; interpolates observed entries, prior mean elsewhere)
   * **SpatialCorrelation** (our proposed kernel): the GP kernel is **exactly** the covariance induced by $\mathbf R_H$.

     * Training uses **noiseless entries** $\{H_{r,t}\}_{(r,t)\in\chi}$ (consistent with the paper).
     * Posterior **mean** equals **LMMSE** with that covariance; posterior **per-entry std** is taken from the diagonal of the GP posterior covariance of `vec(H)` reshaped to $(Nr,Nt)$ (for complex: add Re/Im variances, assuming propriety — also empirically verified in the last cell).
5. Writes the predicted **complex** $\widehat{\mathbf H}$ and **per-entry std** to:

   ```
   /Model/<Case>/<Kernel>/H
   /Model/<Case>/<Kernel>/std
   ```
6. Prints **NMSE** in dB per **Monte → Model → Case → Kernel** as it goes:

   $$
   \text{NMSE(dB)} = 10\log_{10}\frac{\|\widehat{\mathbf H} - \mathbf H\|_F^2}{\|\mathbf H\|_F^2}
   $$

**Implementation hints in the cell**

* **Generic kernels** use `sklearn.gaussian_process.GaussianProcessRegressor` with input $X = \{(r,t)\}$ normalized to $[0,1]^2$, output as **two separate GPs** (Re and Im) sharing the same kernel hyperparameters (fit twice). Set `alpha=1e-9` as jitter; no training noise (noiseless regression).
* **SpatialCorrelation** uses the selection matrix $\mathbf S$ over `vec(H)`; the observed Gram is $\mathbf K_{oo} = \mathbf S \mathbf R_H \mathbf S^{\mathrm H} + \epsilon \mathbf I$.

  * Posterior mean (vectorized):

    $$
    \widehat{\mathbf h} = \mathbf R_H \mathbf S^{\mathrm H} (\mathbf K_{oo})^{-1} \mathbf y
    $$
  * Posterior covariance:

    $$
    \mathbf C = \mathbf R_H - \mathbf R_H \mathbf S^{\mathrm H} (\mathbf K_{oo})^{-1} \mathbf S \mathbf R_H
    $$
  * Store per-entry std as `sqrt(diag(C))` reshaped to `(Nr,Nt)` (for complex, combine Re/Im as noted above).

> **Output file name:** `results_36x36_MC1000.h5` (configurable at the top of the cell).

---

### Cell 2 — Figure 2: **Prediction error scatter** (Kronecker & Weichselberger, all cases & kernels)

**What it does**

* Loads the HDF5 file and, for each **model ∈ {Kronecker, Weichselberger}**, **case**, and **kernel**, computes entry-wise complex errors:

  $$
  \epsilon_{ij} = H_{ij}^{\text{true}} - \widehat{H}_{ij}
  $$
* Plots **scatter of Re vs Im** (subsampled points for legibility), overlays the **95% covariance ellipse**, one panel per (Case × Kernel), in a grid for Kronecker and another grid for Weichselberger.

**Implementation notes**

* Ellipse via the empirical covariance of $(\Re\epsilon,\Im\epsilon)$.
* Titles follow the paper’s naming; keep axes equal and centered.

---

### Cell 3 — Figure 4: **NMSE** (Weichselberger; all cases & kernels)

**What it does**

* Loads HDF5 and computes the **NMSE(dB)** per MC and then **averages across MC** for **Weichselberger only**, for **all cases** and **all kernels**.
* Draws a compact multi-curve plot (or grouped bars) comparing kernels across cases.

**Implementation notes**

* Make sure you average in **linear MSE** before converting to dB if you want an expectation of the ratio; or report mean of per-MC dB values (state which you choose in axis label).

---

### Cell 4 — Figure 5: **Empirical 95% coverage** (Weichselberger; all cases & kernels)

**What it does**

* For **Weichselberger**, **all cases** and **all kernels**, computes the **empirical coverage**:

  $$
  \Pr\big\{\, H_{ij}^{\text{true}} \in [\,\widehat{H}_{ij} \pm 1.96\,\mathrm{std}_{ij}\,] \,\big\}
  $$

  counting across **all entries** and **all MC**.
* Plots coverage (%) per kernel and case.

**Implementation notes**

* For complex: check inclusion on both Re and Im **simultaneously** (i.e., both inside their respective intervals). Alternatively, use the **2D Gaussian** ellipse test; the paper uses marginal 1.96σ bands per component.

---

### Cell 5 — Figure 3: **MI vs SNR** (Weichselberger; SpatialCorrelation only; plus LS & MMSE)

**What it does**

* Uses **Weichselberger** model.
* For **SpatialCorrelation** (all 3 cases), and **LS/MMSE baselines** (from **noisy pilots**), computes the **mutual information** curves vs SNR:

  * **LS/MMSE:** simulated **noisy** pilots with orthogonal $\mathbf X$ (DFT), obtain $\widehat{\mathbf H}_{LS}$ and $\widehat{\mathbf H}_{MMSE}$ per SNR; evaluate MI using the robust bound with estimation error covariance.
  * **SpatialCorrelation:** use the **noiselessly trained** GP’s posterior mean and the RX-side error covariance constructed from the GP posterior covariance (consistent with the paper’s approach).
* Plots MI of **true** channel, **SC-GPR (3 cases)**, **LS**, **MMSE**.

**Implementation notes**

* Robust MI computation can reuse your function from the code (`mi_with_error_cov_xcov_robust`).

---

### Cell 6 — **Propriety and $\Re/\Im$ independence** diagnostics

**What it does**

* For three complex fields:

  1. **Kronecker (true)** channel
  2. **Weichselberger (true)** channel
  3. **Weichselberger, SC-GPR prediction**, **Case II** (50% pilots)

* Draws a **3-panel figure** each:

  * **Left:** hexbin of $(\Re z,\Im z)$ with **95% covariance ellipse**.
  * **Middle:** **polar phase histogram** (expects uniform phase in $(-\pi,\pi]$).
  * **Right:** **Q–Q plots** of standardized $\Re z$ and $\Im z$ vs $\mathcal N(0,1)$.

* Prints six **scalar diagnostics** (computed on large random subsets):

  1. Pearson $r=\mathrm{Corr}(\Re z,\Im z)$
  2. Distance correlation dCor$(\Re z,\Im z)$
  3. HSIC($\Re z,\Im z$) (RFF approximation)
  4. Noncircularity $\kappa=\frac{|\,\mathbb E[z^2]\,|}{\mathbb E[|z|^2]}$
  5. Variance ratio $\mathrm{Var}(\Re z)/\mathrm{Var}(\Im z)$
  6. **Rayleigh** test $p$-value for phase uniformity (implement analytic test for mean resultant length)

**Target qualitative outcomes** (numbers vary with seed; the paper example):

* **Kronecker (true):** $r=-0.002$, dCor$=0.003$, HSIC$=0.017$, $\kappa=0.017$, var-ratio$=0.966$, Rayleigh $p=1.00$.
* **Weichselberger (true):** $r=-0.059$, dCor$=0.063$, HSIC$=0.013$, $\kappa=0.061$, var-ratio$=1.028$, Rayleigh $p=1.00$.
* **Weichselberger, GPR (SC), Case II:** $r=0.036$, dCor$=0.083$, HSIC$=0.031$, $\kappa=0.041$, var-ratio$=1.042$, Rayleigh $p=1.00$.

**Why we care:** These support the modeling assumption that **real and imaginary parts can be treated as independent** GPs with a **shared real kernel** (propriety).

---

## Key implementation details

### 1) SpatialCorrelation kernel (exact from $\mathbf R_H$)

* Observations: indices $\chi \subset [N_r]\times[N_t]$, vectorized by Fortran ordering $i = r + t N_r$.
* Selection matrix $\mathbf S \in \mathbb C^{|\chi| \times (N_r N_t)}$ stacks the canonical row selectors.
* Kernel among observations: $\mathbf K_{oo} = \mathbf S \mathbf R_H \mathbf S^{\mathrm H} + \epsilon \mathbf I$.
* Posterior mean (vectorized): $ \widehat{\mathbf h} = \mathbf R_H \mathbf S^{\mathrm H} \mathbf K_{oo}^{-1} \mathbf y$.
* Posterior covariance: $ \mathbf C = \mathbf R_H - \mathbf R_H \mathbf S^{\mathrm H} \mathbf K_{oo}^{-1} \mathbf S \mathbf R_H$.
* Reshape $\widehat{\mathbf h}$ back to $ \widehat{\mathbf H} \in \mathbb C^{N_r\times N_t}$.
* Store **per-entry std** as `sqrt(diag(C))` reshaped (for complex: combine Re/Im from the two real GPs or take diag of complex covariance and split appropriately — our notebook does the former and validates propriety).

### 2) Generic kernels (RBF, Matérn, RQ, WhiteKernel)

* Inputs: scaled coordinates $X=\{(r/N_r,t/N_t)\}$ in $[0,1]^2$.
* Fit **two** GPs per kernel: one for **Re$\{H\}$**, one for **Im$\{H\}$**; both use the **same kernel class** (hyperparameters are learned independently; you may optionally lock them by reusing the kernel learned on Re for Im).
* Predict mean and std on the **entire grid**; assemble complex mean as $\hat H = \hat h_\mathrm{Re} + j\,\hat h_\mathrm{Im}$.
* Store `std = np.sqrt(std_Re**2 + std_Im**2)`.

### 3) LS & MMSE benchmarks

* **Pilots:** unitary DFT $\mathbf X \in \mathbb C^{N_t \times N_t}$ (orthogonal), one SNR loop per benchmark figure.
* **LS:** $\widehat{\mathbf H}_{LS} = \mathbf Y \mathbf X^{\mathrm H}$ with $\mathbf Y = \mathbf H \mathbf X + \mathbf N$ and $ \mathbf X \mathbf X^{\mathrm H} = \mathbf I$.

  * RX-side estimation error covariance: $\Sigma_e^{LS} \approx N_t \sigma^2 \mathbf I$.
* **MMSE (Weichselberger prior):** in eigen domain, shrinkage factor $\Omega / (\Omega + \sigma^2)$ applied to $\mathbf Z = \mathbf Y \mathbf X^{\mathrm H}$.

  * RX-side error covariance aggregated from $\Omega$ and $\sigma^2$.

### 4) NMSE and coverage

* **NMSE(dB)**: default is the mean of per-MC NMSE(dB); if you prefer expectation of the ratio use linear averaging then convert to dB (both are standard; we’ll label what we plot).
* **Coverage:** fraction of entries $H_{ij}$ with **both** Re and Im inside $ \hat H \pm 1.96\,\mathrm{std}$.

---

## Reproducibility

* Global RNG seed is set once and recorded in the HDF5 file attributes along with:

  * `MC`, `Nr`, `Nt`, carrier frequency, array geometry, angle spread, and SNR grid used for MI.
* The notebook prints intermediate NMSE summaries and the final paths written inside the HDF5 to ease inspection with `h5ls`.

---

## File checklist

* `Syed_IEEE_Letter.ipynb` — the only notebook to run.
* `results_36x36_MC1000.h5` — produced by Cell 1 (not tracked by git by default; add to `.gitignore` if large).

---

## Expected outputs (sanity)

* **Figure 2**: tighter, more circular error clouds for **SpatialCorrelation**, especially in Case II; Weichselberger tends to look more isotropic than Kronecker.
* **Figure 3**: MI — **SC-GPR (Case II)** close to MMSE with 50% pilots; **Case I** deviates earlier; **LS** lags both.
* **Figure 4**: NMSE — SC best; RBF/Matérn similar; RQ slightly worse in Case III (heavier-tailed behavior).
* **Figure 5**: coverage — well-calibrated \~95% for SC and similar for generic kernels when lengthscales grow relative to grid diameter (isotropic setting).

---

## Independence / propriety diagnostics (what you’ll see)

* **Joint clouds**: centered near origin, nearly circular 95% ellipse.
* **Phases**: uniform on $(-\pi,\pi]$.
* **Q–Q**: Re and Im standardized marginals lie on 45° line.
* **Scalars**: $r\approx 0$, dCor$\approx 0$, HSIC$\approx 0$, $\kappa \approx 0$, variance ratio $\approx 1$, Rayleigh $p$ large — including **SC-GPR (Case II)**, which supports modeling Re/Im with **independent real GPs** and a **shared real kernel**.

---

## License

MIT (code) — please cite the paper if you use this repository in academic work.

---

## Citation

> Syed Luqman Shah, Nurul Huda Mahmood, Italo Atzeni,
> **“Low-Overhead CSI Prediction via Gaussian Process Regression,”** IEEE Wireless Communications Letters, 2025.

---

### Appendix: snippets you can reuse inside the notebook

**Build $\mathbf K_{oo}$, posterior mean & per-entry std for SpatialCorrelation**

```python
# inputs: R_H  (Nrt x Nrt, complex), pairs idx list over (r,t),
#         H_true (Nr x Nt, complex), epsilon = 1e-9
Nr, Nt = H_true.shape
Nrt = Nr*Nt
# Build selection rows (Fortran order: i = r + t*Nr)
obs_idx = np.array([r + t*Nr for (r,t) in pairs], dtype=int)
S = np.eye(Nrt, dtype=np.complex128)[obs_idx, :]                   # (m x Nrt)
Koo = S @ R_H @ S.conj().T + (1e-9)*np.eye(len(pairs), dtype=np.complex128)
y  = H_true.reshape(-1, order='F')[obs_idx]                        # (m,)

# Posterior mean (vectorized)
L = np.linalg.cholesky(0.5*(Koo + Koo.conj().T))
alpha = np.linalg.solve(L.conj().T, np.linalg.solve(L, y))
h_hat = R_H @ S.conj().T @ alpha                                   # (Nrt,)
H_hat = h_hat.reshape(Nr, Nt, order='F')

# Posterior covariance diagonal (per-entry variance)
C = R_H - R_H @ S.conj().T @ np.linalg.solve(Koo, S @ R_H)
std = np.sqrt(np.real(np.diag(C))).reshape(Nr, Nt, order='F')       # combine Re/Im if you model them separately
```

**Coverage check (marginal 95%)**

```python
in_re = (np.real(H_true) >= np.real(H_hat) - 1.96*std) & (np.real(H_true) <= np.real(H_hat) + 1.96*std)
in_im = (np.imag(H_true) >= np.imag(H_hat) - 1.96*std) & (np.imag(H_true) <= np.imag(H_hat) + 1.96*std)
coverage = np.mean(in_re & in_im)
```

**Ellipse overlay (2×2 empirical covariance)**

```python
def covariance_ellipse(ax, x, y, nsig=2, **kw):
    X = np.vstack([x, y]).T
    mu = X.mean(axis=0)
    S  = np.cov(X, rowvar=False)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:,order]
    t = np.linspace(0, 2*np.pi, 200)
    u = np.vstack([np.cos(t), np.sin(t)])
    # nsig=2 ~ 95% for Gaussian
    el = (V @ np.diag(np.sqrt(w)) @ u) * nsig + mu[:,None]
    ax.plot(el[0,:], el[1,:], **kw)
```

---

If anything in your machine setup differs (e.g., you want fewer than 1000 MC for a quick smoke test), just change `MC` in Cell 1 — the rest of the notebook will still work off the HDF5 you created.
