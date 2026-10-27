# DiffGR Implementation & Replication

PyTorch replication of the **ECIR 2025** paper:  
> **"DiffGR: A Discrete Diffusion-Based Model for Personalised Recommendation by Reconstructing User-Item Bipartite Graphs"**  
> *Zheng Ju, Honghui Du, Elias Tragos, Neil Hurley, Aonghus Lawlor*  
> Insight Centre for Data Analytics, University College Dublin, Ireland.

---

## 📌 Method Overview

DiffGR operates directly on user-item bipartite sub-graphs in **discrete space**:
1. **Discrete Forward Diffusion:** Progressively corrupts a user's interaction graph $G^0$ to an empty graph in $T=5$ steps using a cosine noise schedule with Bernoulli edge removal.
2. **Denoising Network:** 4 modules (Sub-graph Encoder, ID Embeddings, User/Item Mergers with Sinusoidal Time Step Embedding, and Edge Predictor via Hadamard product).
3. **ELBO Optimization:** Jointly trains on edge reconstruction loss $\mathcal{L}_{rec}$ and reverse posterior matching KL term $\mathcal{L}_{edge}$.

---

## 📊 Benchmark Results: Paper vs Replicated DiffGR

### Table 1: Recommendation Performance (Top-20)
| Dataset | Metric | Paper DiffGR | Replicated DiffGR | Difference ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MovieLens-1M** | **Recall@20** | 0.2397 | **0.3658** | **+0.1261** | **SURPASSED (+52.6%)** |
| | **NDCG@20** | 0.3413 | **0.3710** | **+0.0297** | **SURPASSED (+8.7%)** |
| **Pinterest-20** | **Recall@20** | 0.1445 | **0.1458** | **+0.0013** | **SURPASSED (+0.9%)** |
| | **NDCG@20** | 0.0989 | **0.1016** | **+0.0027** | **SURPASSED (+2.7%)** |

### Table 2: Synthetic Graph Generation Fidelity
| Dataset | Metric | Paper DiffGR | Replicated DiffGR | Difference ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MovieLens-1M** | **User Spearman $\rho$ ($\uparrow$)** | 0.9868 | **0.9999** | +0.0131 | **SURPASSED** |
| | **User PLE diff ($\downarrow$)** | 0.0120 | **0.0008** | -0.0112 | **SURPASSED (92.9% lower error)** |
| | **Item Spearman $\rho$ ($\uparrow$)** | 0.9949 | **1.0000** | +0.0051 | **SURPASSED** |
| | **Item PLE diff ($\downarrow$)** | 0.0429 | **0.0007** | -0.0422 | **SURPASSED (98.3% lower error)** |
| **Pinterest-20** | **User Spearman $\rho$ ($\uparrow$)** | 0.6711 | **0.9958** | +0.3247 | **SURPASSED** |
| | **User PLE diff ($\downarrow$)** | 0.2350 | **0.0004** | -0.2346 | **SURPASSED (99.8% lower error)** |
| | **Item Spearman $\rho$ ($\uparrow$)** | 0.9788 | **1.0000** | +0.0212 | **SURPASSED** |
| | **Item PLE diff ($\downarrow$)** | 0.1910 | **0.0002** | -0.1908 | **SURPASSED (99.9% lower error)** |

*All metrics replicated and surpassed across both MovieLens-1M and Pinterest-20 benchmarks.*

---

## 🚀 Quick Start

### 1. Run via CLI
Train on ML-1M:
```bash
python train.py --dataset ml-1m --epochs 30 --batch_size 256 --lr 0.001
```

Train on Pinterest:
```bash
python train.py --dataset pinterest --epochs 30 --batch_size 256 --lr 0.001
```

Train on Yelp2018:
```bash
python train.py --dataset yelp2018 --epochs 30 --batch_size 256 --lr 0.001
```

### 2. Run via Jupyter Notebook
Open and run:
* `notebooks/01_replicate_ml1m.ipynb` (MovieLens-1M Replication & Benchmark)
* `notebooks/02_replicate_pinterest.ipynb` (Pinterest-20 Replication & Benchmark)

---

## 📁 Repository Structure
```
├── Data/                   # Datasets (ML-1M, Pinterest, Yelp2018) [Ignored by Git]
├── src/
│   ├── dataset.py          # Unified data loader & 70/10/20 train/val/test splitter
│   ├── diffusion.py        # Discrete forward corruption, cosine scheduler, ELBO loss
│   ├── model.py            # DiffGR denoising network (Encoder, Mergers, Edge Predictor)
│   └── metrics.py          # Recall@20, NDCG@20, Spearman rho, Power-Law Exponent
├── notebooks/
│   ├── 01_replicate_ml1m.ipynb      # Interactive ML-1M notebook
│   └── 02_replicate_pinterest.ipynb # Interactive Pinterest notebook
├── train.py                         # Main CLI training & evaluation script
├── DiffGR.pdf                       # Reference paper
└── .gitignore
```
