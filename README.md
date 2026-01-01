# Low-Overhead CSI Prediction via Gaussian Process Regression

This repository provides the **fully reproducible source code and data** for the paper:

> **Low-Overhead CSI Prediction via Gaussian Process Regression**
> *Syed Luqman Shah, Nurul Huda Mahmood, and Italo Atzeni*
> **IEEE Wireless Communications Letters**, vol. 15, pp. 1075–1079, 2026.

---

## 📌 Overview

Accurate channel state information (CSI) is essential for reliable and spectrally efficient multi-antenna wireless systems. However, conventional pilot-based channel estimation methods become increasingly inefficient as the number of antennas grows, due to prohibitive pilot overhead.

This repository implements a **Gaussian Process Regression (GPR)-based CSI prediction framework** that reconstructs the full MIMO channel matrix from **only a small subset of observed entries**, achieving substantial pilot savings while preserving performance. Beyond point estimation, the proposed framework provides **well-calibrated uncertainty quantification** via the posterior variance of GPR.

The codebase reproduces **all numerical results, figures, and tables** reported in the paper.

---

## 🧠 Core Idea

The key insight of this work is that **wireless channels exhibit structured spatial correlations** induced by the antenna array geometry. GPR naturally exploits this structure through its kernel (covariance) function.

With only a few observed channel entries, GPR:

* infers the **entire channel matrix**,
* adapts to the observed samples at run time, and
* provides **statistically meaningful confidence intervals** via its posterior variance.

This enables **low-overhead CSI acquisition with calibrated uncertainty**, making GPR particularly attractive for next-generation large-scale MIMO systems.

---

## 📄 Paper and Preprint

* **Published version**: Included in this repository as
  `Research paper.pdf` [https://doi.org/10.1109/LWC.2025.3648532]
* **Preprint (LaTeX source)**:
  [https://doi.org/10.48550/arXiv.2510.25390](https://doi.org/10.48550/arXiv.2510.25390)

If you use this repository, please cite the published IEEE WCL paper.

---

## 📂 Repository Structure

```
.
├── Research paper.pdf
├── First_Letter.ipynb
├── channel_models_compact.h5
├── figures/
│   ├── per_entry_error/
│   ├── credible_intervals/
│   └── spectral_efficiency/
├── requirements.txt
└── README.md
```

---

## 🧪 Reproducible Experiments

All experiments are implemented in the Jupyter notebook:

```
First_Letter.ipynb
```

The notebook is **self-contained** and organized sequentially to mirror the paper.

### 1️⃣ Channel Generation and GPR Estimation

* Generates the **true MIMO channel matrices** under two channel models.
* Applies **GPR-based CSI prediction** using three kernels:

  * Radial Basis Function (RBF)
  * Matérn
  * Rational Quadratic (RQ)
* Saves the true channels, GPR predictions, and posterior variances to:

  ```
  channel_models_compact.h5
  ```

This file acts as a compact data container used by all subsequent experiments.

---

### 2️⃣ Per-Entry Prediction Error (Fig. 2)

* Loads the `.h5` file.
* Computes per-entry estimation errors.
* Visualizes prediction errors together with **95% credible intervals** derived from the GPR posterior variance.
* This corresponds to **Fig. 2** in the paper.

---

### 3️⃣ Empirical 95% Credible-Interval Coverage (Fig. 4)

This section evaluates the **uncertainty calibration** of GPR.

* Computes the empirical coverage probability of the marginal 95% credible intervals:
  $\widehat{H}*{ij} \pm 1.96,\sigma*{ij}$
* Coverage is reported for:

  * RBF kernel
  * Matérn kernel
  * Rational Quadratic kernel

A well-calibrated model achieves coverage close to the nominal **95%**:

* higher → conservative uncertainty
* lower → overconfident uncertainty

These results are reported in **Fig. 4** of the paper.

---

### 4️⃣ Spectral Efficiency Evaluation (Fig. 3 and Table II)

To assess how well CSI prediction preserves communication performance, the notebook evaluates the **spectral efficiency (SE)** of a multi-stream MIMO link using a linear receiver designed from the *estimated* channel.

#### Linear MMSE Detector

Given an estimated channel $\widehat{\mathbf{H}} \in \mathbb{C}^{N_{\mathrm{r}} \times N_{\mathrm{t}}}$, the linear minimum mean-square error (LMMSE) detector is constructed as

$$
\mathbf{W}(\widehat{\mathbf{H}}) = \left(\widehat{\mathbf{H}}\widehat{\mathbf{H}}^{\mathsf{H}}+\frac{N_{\mathrm{t}}}{\rho}\mathbf{I}_{N_{\mathrm{r}}}\right)^{-1}\widehat{\mathbf{H}}=[\mathbf{w}_1,\ldots,\mathbf{w}_{N_{\mathrm{t}}}]
$$


where $\rho$ denotes the signal-to-noise ratio (SNR), and $\mathbf{w}_k$ is the detector vector for stream $k$.

#### Post-Equalization SINR

Let $\mathbf{h}_k$ denote the $k$-th column of the **true** channel matrix $\mathbf{H}$. The post-equalization signal-to-interference-plus-noise ratio (SINR) of stream $k$ is computed as

$$
\mathrm{SINR}_{k}(\widehat{\mathbf{H}}) \;=\; \frac{ \left| \mathbf{w}_k^{\mathsf{H}} \mathbf{h}_k \right|^{2} }{ \sum_{j \neq k} \left| \mathbf{w}_k^{\mathsf{H}} \mathbf{h}_j \right|^{2} \;+\; \frac{N_t}{\rho} \, \lVert \mathbf{w}_k \rVert^{2} }
$$.





Note that the **true channel $\mathbf{H}$** is always used inside the SINR expression, while the estimate $\widehat{\mathbf{H}}$ affects performance only through the detector $\mathbf{W}(\widehat{\mathbf{H}})$.

#### Spectral Efficiency

The corresponding spectral efficiency is then given by

$$
\mathrm{SE}(\widehat{\mathbf{H}})=\sum_{k=1}^{N_{\mathrm{t}}}\log_2\!\left(1+\mathrm{SINR}_{k}(\widehat{\mathbf{H}})\right)
$$.

#### Compared Estimators

Spectral efficiency is evaluated for the following channel estimates:

- True channel $\mathbf{H}$
- GPR-predicted channel $\widehat{\mathbf{H}}_{\mathrm{GPR}}$
- Least-squares (LS) estimate $\widehat{\mathbf{H}}_{\mathrm{LS}}$
- MMSE estimate $\widehat{\mathbf{H}}_{\mathrm{MMSE}}$

All three GPR kernels (RBF, Matérn, and Rational Quadratic) are implemented in the repository. However, **only the Matérn kernel** is reported in **Fig. 3 and Table II** of the paper.


---

## ⚙️ Requirements

A typical Python scientific stack is sufficient.
A `requirements.txt` file is provided.

Main dependencies include:

* NumPy
* SciPy
* h5py
* scikit-learn
* matplotlib
* Jupyter

---

## 🔁 Reproducibility

Running `First_Letter.ipynb` from top to bottom:

* regenerates all channels,
* reproduces all figures,
* and recomputes all numerical results reported in the paper.

Random seeds are fixed where appropriate to ensure reproducibility.

---


## 📜 Citation

If you use this code or data, please cite the following paper.

### Plain-text citation

```

S. L. Shah, N. H. Mahmood, and I. Atzeni,
“Low-Overhead CSI Prediction via Gaussian Process Regression,”
IEEE Wireless Communications Letters, vol. 15, pp. 1075–1079, 2026.

````

### BibTeX

```
@article{Shah2026LowOverheadCSI,
  author  = {Shah, Syed Luqman and Mahmood, Nurul Huda and Atzeni, Italo},
  title   = {Low-Overhead CSI Prediction via Gaussian Process Regression},
  journal = {IEEE Wireless Communications Letters},
  volume  = {15},
  pages   = {1075--1079},
  year    = {2026},
  publisher={IEEE}
}
```



---

## 📬 Contact

For questions, comments, or reproducibility issues, please contact:

**Syed Luqman Shah**
email: sayedluqmans@gmail.com
