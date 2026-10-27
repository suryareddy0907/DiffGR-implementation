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

## 📊 Target Benchmark Results (Paper Table 1)

| Dataset | Sparsity | Recall@20 | NDCG@20 |
| :--- | :--- | :--- | :--- |
| **ML-1M** | 95.53% | **0.2397** | **0.3413** |
| **Pinterest** | 99.73% | **0.1445** | **0.0989** |
| **Yelp2018** | 99.87% | **0.0844** | **0.0730** |

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
* `notebooks/01_replicate_ml1m.ipynb`

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
│   └── 01_replicate_ml1m.ipynb # Interactive ML-1M notebook
├── train.py                # Main CLI training & evaluation script
├── DiffGR.pdf              # Reference paper
└── .gitignore
```
