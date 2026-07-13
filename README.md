<div align="center">

# ⚛️ Quantum Encoding Strategies for IC₅₀ Drug Response Prediction
### Exhaustive Benchmark on IQM Garnet 20-Qubit QPU

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.07.08.737310-bd2635?logo=biorxiv)](https://www.biorxiv.org/content/10.64898/2026.07.08.737310v1)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://python.org)
[![Qrisp](https://img.shields.io/badge/Qrisp-0.8.2-orange)](https://qrisp.eu)
[![IQM](https://img.shields.io/badge/IQM-Garnet%2020Q-blueviolet)](https://resonance.iqm.tech)
[![Dataset](https://img.shields.io/badge/Dataset-GDSC2-green)](https://www.cancerrxgene.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--6157--2647-a6ce39?logo=orcid)](https://orcid.org/0009-0009-6157-2647)

> **First hardware benchmark of 12 quantum encoding strategies on a 20-qubit  
> superconducting QPU for a real pharmaceutical regression task.**

</div>

---

## 📋 Overview

This repository contains the full reproducible pipeline for the paper:

> **"Quantum Encoding Strategies for Drug Response Prediction:  
> An Exhaustive Benchmark on a 20-Qubit Superconducting QPU"**  
> Rania Derouich, Nour El Houda Mathlouthi — GenoFlow Agency, Tunis, Tunisia — 2026  
> *Posted on bioRxiv, July 13, 2026 — [doi.org/10.64898/2026.07.08.737310](https://doi.org/10.64898/2026.07.08.737310)*

We benchmark **12 quantum data-encoding strategies** for regression of
IC₅₀ drug response from the **GDSC2 dataset** (242,036 drug–cell-line pairs),
executed on the real **IQM Garnet 20-qubit QPU** via IQM Resonance cloud.

### 🏆 Key Results

| Rank | Encoding | RMSE ↓ | MAE ↓ | R² | Qubits |
|:----:|----------|:------:|:-----:|:--:|:------:|
| 🥇 1 | **QAOA-inspired** | **3.314** | 2.869 | −0.357 | 8 |
| 🥈 2 | Hamiltonian | 3.404 | **2.754** | −0.431 | 8 |
| 🥉 3 | Squeezing | 3.535 | 2.826 | −0.544 | 8 |
| 4 | Amplitude | 3.543 | 3.113 | −0.551 | **4** |
| 5 | ZZ Feature Map | 3.546 | 3.009 | −0.553 | 8 |
| 6 | Displacement | 3.653 | 3.111 | −0.649 | 8 |
| 7 | Data Re-uploading | 3.791 | 3.332 | −0.775 | 8 |
| 8 | Hybrid | 3.918 | 3.441 | −0.897 | 8 |
| 9 | Angle RX | 4.524 | 4.037 | −1.528 | 8 |
| 10 | Basis | 4.793 | 3.809 | −1.839 | 3 |
| 11 | Angle RY+RZ | 5.894 | 5.103 | −3.292 | 8 |
| 12 | IQP | 6.230 | 5.878 | −3.795 | 8 |

> **Statistical test:** QAOA-inspired significantly outperforms 6/11 competitors  
> (Wilcoxon signed-rank, p < 0.05). All R² < 0, confirming NISQ-regime underfitting — results are meaningful as a *relative* hardware benchmark.

---

## 🗂️ Repository Structure

```
quantum-encoding-ic50-qpu/
├── notebooks/
│   └── quantum_encoding_benchmark_qrisp_iqm.ipynb   # Full executable pipeline
├── src/
│   └── encodings_qrisp.py      # 12 Qrisp encoding circuits
├── results/
│   ├── qpu_benchmark_results.csv     # Final leaderboard (12 encodings)
│   └── qpu_benchmark_full.json       # Raw QPU predictions + metadata
├── assets/                           # All figures (10 publication-quality PNGs)
│   ├── scatter_grid.png              # True vs. predicted — 12 panels
│   ├── benchmark_overview.png        # Multi-metric dashboard
│   ├── GDSC2_dataset_overview.png    # Dataset overview
│   ├── fig1_rmse_bar.png             # RMSE leaderboard
│   ├── fig2_rmse_qubits.png          # RMSE vs qubit count
│   ├── fig3_mae_rmse.png             # MAE vs RMSE scatter
│   ├── fig4_r2.png                   # R² scores
│   ├── fig5_wilcoxon.png             # Wilcoxon significance overview
│   ├── fig6_overview_2x2.png         # 2×2 overview panel
│   ├── fig7_exec_time.png            # QPU execution time per encoding
│   ├── fig8_error_dist.png           # Error distribution per encoding
│   ├── fig9_rank_heatmap.png         # Multi-metric rank heatmap
│   └── fig10_wilcoxon_detail.png     # Detailed Wilcoxon results
├── paper/
│   ├── main.tex                      # LaTeX source (v4, arXiv-ready)
│   └── supplementary.tex             # Supplementary material
├── requirements.txt
├── LICENSE                           # MIT
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone and install

```bash
git clone https://github.com/rania-derouich/quantum-encoding-ic50-qpu.git
cd quantum-encoding-ic50-qpu
pip install -r requirements.txt
```

### 2. Get your IQM Resonance token

Sign up at [resonance.iqm.tech](https://resonance.iqm.tech) → Profile → API Tokens.

### 3. Run on mock backend (zero credits)

```python
# In the notebook, set:
CFG["use_real_qpu"] = False   # uses garnet:mock — identical API, no credits
```

### 4. Run on real QPU (requires IQM Resonance credits)

```python
CFG["use_real_qpu"] = True
# Token entered via secure getpass prompt — never hardcoded
```

> ⚠️ Full QPU benchmark requires IQM Resonance credits  
> (600 circuits × 1,024 shots, ~34.6 min wall time).

---

## 🔬 The 12 Encodings

Each encoding $\mathcal{U}_k(x)$ maps 8 molecular features into the quantum
state, followed by 2 variational layers (RY+RZ rotations + CZ ladder),
optimised with COBYLA (200 iterations, classical proxy on all 50 samples).

| # | Encoding | Key mechanism | Reference |
|---|----------|---------------|-----------|
| 1 | **Angle RX** | `RX(xᵢ)` per qubit | Schuld & Killoran 2019 |
| 2 | **Angle RY+RZ** | `RY(xᵢ) RZ(x_{i+N/2})` | Pérez-Salinas+ 2020 |
| 3 | **Amplitude** | Feature vector → statevector amplitudes | Möttönen+ 2005 |
| 4 | **Basis** | Discrete index → computational basis | Nielsen & Chuang 2010 |
| 5 | **ZZ Feature Map** | Two-body `ZZ(xᵢ·xⱼ)` interactions | Havlíček+ 2019 |
| 6 | **IQP** | Diagonal gates in Hadamard basis | Shepherd & Bremner 2009 |
| 7 | **Hamiltonian** | Trotterised evolution `e^{-iH(x)t}` | Lloyd+ 2020 |
| 8 | **QAOA-inspired** | Alternating cost (`RZ(xᵢ)`) + mixer (`RX(β)`) | Farhi+ 2014 |
| 9 | **Squeezing** | `RX(xᵢ) RZ(xᵢ²)` + beam-splitter | Killoran+ 2019 |
| 10 | **Hybrid** | Angle (qubits 0–3) + Amplitude (qubits 4–7) | This work |
| 11 | **Displacement** | `RX(2α cos φ) RY(2α sin φ)` pairs | This work |
| 12 | **Data Re-uploading** | Features re-injected at every layer depth | Pérez-Salinas+ 2020 |

---

## 🖥️ Hardware Details

| Parameter | Value |
|-----------|-------|
| QPU | IQM Garnet 20Q |
| Native 2Q gate | CZ |
| Topology | Crystal (nearest-neighbour) |
| Shots per circuit | 1,024 |
| SDK | Qrisp 0.8.2 |
| Avg. latency/sample | 3.47 ± 0.09 s |
| Total QPU wall time | ~34.6 min (600 circuits) |
| QPU credits | Provided free of charge by IQM Resonance (incl. 280 credits from the *Simulating Complex Materials with Quantum Chemistry and SQD* workshop) |
| Optimiser | COBYLA (200 iters, classical proxy, all 50 samples) |

---

## 📊 Dataset

**GDSC2** (Genomics of Drug Sensitivity in Cancer, release 8.5 — October 2023)

- Source: [cancerrxgene.org](https://www.cancerrxgene.org/downloads/bulk_download)
- 242,036 drug–cell-line pairs · 286 drugs · 969 cell lines
- Target: `LN_IC50` ∈ [−8.75, 13.82]
- 8 features: AUC, RMSE, Z-score, DRUG_ID, COSMIC_ID, cancer type flags (UNCLASSIFIED, LUAD, SCLC)
- QPU subset: 50 stratified samples (10 bins × 5), LN_IC50 ∈ [−5.80, 6.35]

---

## 📈 Statistical Analysis

Wilcoxon signed-rank test comparing QAOA-inspired vs. all other encodings:

| Encoding | W-stat | p-value | Significant (α = 0.05) |
|----------|:------:|:-------:|:----------------------:|
| Angle RX | 65.5 | < 0.001 | ✅ |
| Angle RY+RZ | 110.0 | < 0.001 | ✅ |
| IQP | 34.0 | < 0.001 | ✅ |
| Hybrid | 256.5 | 0.0002 | ✅ |
| Data Re-uploading | 376.5 | 0.019 | ✅ |
| Basis | 397.5 | 0.021 | ✅ |
| Displacement | 514.0 | 0.233 | ❌ |
| Amplitude | 486.5 | 0.145 | ❌ |
| Squeezing | 590.5 | 0.650 | ❌ |
| ZZ Feature Map | 601.0 | 0.730 | ❌ |
| Hamiltonian | 634.0 | 0.977 | ❌ |

---

## 📄 Paper (LaTeX Source)

The full manuscript LaTeX source is in [`paper/main.tex`](paper/main.tex)  
(revtex4-2 format, arXiv-ready v4). Compile with:

```bash
cd paper
pdflatex main.tex && pdflatex main.tex
```

The supplementary material (circuit definitions, pseudocode, extended results)  
is in [`paper/supplementary.tex`](paper/supplementary.tex).

---

## 🛡️ Security Note

**IQM API tokens are never hardcoded in this repository.**  
All token handling uses Python's `getpass` (masked prompt) or environment variables:

```python
import os
from getpass import getpass

IQM_TOKEN = os.environ.get("IQM_TOKEN") or getpass("IQM Token: ")
```

---

## 📄 Citation

If you use this code or results, please cite:

```bibtex
@article{derouich2026quantum,
  title   = {Quantum Encoding Strategies for Drug Response Prediction:
             An Exhaustive Benchmark on a 20-Qubit Superconducting QPU},
  author  = {Derouich, Rania and Mathlouthi, Nour El Houda},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.07.08.737310},
  url     = {https://www.biorxiv.org/content/10.64898/2026.07.08.737310v1}
}
```

> 📌 The BibTeX `note` field will be updated to the arXiv ID once the preprint is live.  
> You can watch this repository for the update.

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Made with ⚛️ by [Rania Derouich](https://orcid.org/0009-0009-6157-2647)**  
*IQM Resonance QPU · Qrisp 0.8.2 · GDSC2 · Tunisia · 2026*

</div>
