# Topology-Aware HYFA: Targeted Multi-Tissue Gene Expression Imputation

> A topology-aware extension of [HYFA](https://www.nature.com/articles/s42256-023-00684-8) that improves targeted gene expression imputation by enforcing biological network consistency.

Welcome to the repository for **Topology-Aware HYFA**, a hybrid deep learning architecture that integrates the parameter-efficient hypergraph message-passing of **[HYFA](https://www.nature.com/articles/s42256-023-00684-8)** with the biological network priors of **[TEEBoT](https://www.science.org/doi/10.1126/sciadv.abd6991)**.

This pipeline is specifically engineered for **targeted gene expression imputation** (e.g., tracing a 15-gene cardiovascular signature in heart tissue using non-invasive whole blood transcriptomes). By moving away from noisy, global transcriptome prediction and injecting biological co-expression graphs into the loss function, this model aims to improve the recovery of specific, clinically relevant biomarkers.

## 🧬 Integrated Architecture

This repository fuses the methodologies of two foundational papers:

1. **HYFA (Hypergraph Factorization):** Acts as the core generative engine, using a hypergraph neural network to learn factorized representations of individuals, tissues, and metagenes.
2. **TEEBoT (Tissue Expression Estimation using Blood Transcriptome):** Provides the inductive biological prior. TEEBoT demonstrated that predictable genes share regulatory networks. We enforce this via a **Graph Smoothness Regularizer**, penalizing the model if biologically correlated genes learn divergent latent representations.

### Key Features

- **Targeted Bottlenecking:** Dynamically slices the global GTEx feature space to focus exclusively on user-defined target genes, padding missing genes to maintain VAE matrix stability.
- **Topological Regularization:** Computes the Graph Smoothness penalty ($Tr(W^T L W)$) using a provided co-expression/PPI adjacency matrix, integrated natively into the PyTorch Negative Binomial reconstruction loss.
- **Dynamic CLI:** Fully modular Command Line Interface (CLI) allowing rapid ablation studies across varying tissues, target genes, and regularization weights ($\lambda_{reg}$).

## 🧪 Baseline: TEEBoT

We benchmark against **[TEEBoT](https://www.science.org/doi/10.1126/sciadv.abd6991)**, a classical approach for predicting tissue-specific gene expression from whole blood transcriptomes.

TEEBoT operates by:
- Performing **gene-wise prediction** using LASSO regression
- Using **principal components (PCs)** of blood gene expression and splicing as features
- Incorporating demographic confounders (age, sex, race)
- Filtering predictable genes via **likelihood ratio tests (LLR)**

### Limitations of TEEBoT

- ❌ Treats each gene **independently** (no shared structure)
- ❌ Ignores **gene-gene interaction networks**
- ❌ Limited capacity to model **nonlinear relationships**
- ❌ Requires separate models per gene (not scalable)

### How TopoHYFA Improves Upon This

TopoHYFA addresses these limitations by:
- ✅ Learning **shared latent representations** via hypergraph neural networks
- ✅ Injecting **biological topology (co-expression/PPI graphs)** via graph regularization
- ✅ Modeling **nonlinear cross-tissue relationships**
- ✅ Enabling **joint multi-gene prediction** in a unified framework

| Feature | TEEBoT | TopoHYFA |
|---|---|---|
| Model type | LASSO | Hypergraph NN |
| Gene interactions | ❌ | ✅ |
| Nonlinearity | ❌ | ✅ |
| Joint prediction | ❌ | ✅ |
| Biological priors | ❌ | ✅ |

---

## ⚙️ Installation

1. Clone this repository:

```bash
git clone https://github.com/rrsathe/TopoHYFA.git
cd TopoHYFA
```

2. Install the dependencies (using `uv` or `pip`):

```bash
uv pip install -r requirements.txt
```

---

## 📊 Data Preparation

This model requires normalized **GTEx v8** data.

1. **Download the raw data:**
   - Download the processed HYFA expression data: `GTEX_data.csv.zip` ([Link](https://figshare.com/ndownloader/files/40208074))
   - Download the GTEx v8 Subject Phenotypes: `GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt` ([Link](https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt))
   - Extract and place both files in the `data/` directory.

2. **Generate the handoff datasets:**
   Use the included preparation script to subset your target genes, extract demographic confounders, and generate the data-driven $15 \times 15$ co-expression adjacency matrix.

```bash
python prep_handoff.py
```

This generates `target_genes_15.csv`, `confounders.csv`, and `adjacency_matrix.csv` in `Imputation/output/HYFA_export/`.

---

## 🚀 Usage

The architecture is controlled via the `train_gtex.py` CLI.

### 1. Train the Baseline Model (Control)

Train the standard hypergraph model without topological priors by setting `--lambda-reg 0.0`.

```bash
uv run python train_gtex.py \
  --source-tissue "Whole Blood" \
  --target-tissue "Heart_L_Vent" \
  --target-genes Imputation/output/HYFA_export/target_genes_15.csv \
  --confounders Imputation/output/HYFA_export/confounders.csv \
  --topology-matrix Imputation/output/HYFA_export/adjacency_matrix.csv \
  --lambda-reg 0.0

# Preserve the weights
mv data/model.pth data/model_baseline.pth
```

### 2. Train the Topology-Aware Model

Activate the biological priors by setting a positive regularization weight (e.g., `--lambda-reg 0.1`).

```bash
uv run python train_gtex.py \
  --source-tissue "Whole Blood" \
  --target-tissue "Heart_L_Vent" \
  --target-genes Imputation/output/HYFA_export/target_genes_15.csv \
  --confounders Imputation/output/HYFA_export/confounders.csv \
  --topology-matrix Imputation/output/HYFA_export/adjacency_matrix.csv \
  --lambda-reg 0.1

# Preserve the weights
mv data/model.pth data/model_topology.pth
```

### 3. Evaluate and Compare (Ablation Study)

Run the automated evaluation scripts to test both models and compare targeted-gene performance.

```bash
uv run python eval_15.py
uv run python benchmark_teebot.py
```

## 📈 Results (Preliminary)

- Improved Pearson correlation on targeted genes compared to baseline HYFA
- Better recovery of biologically co-expressed gene clusters
- More stable predictions for low-expression genes

*(Full quantitative benchmarks and plots coming soon)*

---

## 📁 Core Project Structure

- `train_gtex.py` - Main CLI training loop with dynamic subsetting.
- `eval_15.py` - Pearson correlation and RMSE benchmarking on the 15-gene task.
- `benchmark_teebot.py` - Side-by-side HYFA vs. TEEBoT baseline benchmarking.
- `prep_handoff.py` - Data processing pipeline bridging GTEx structures to model inputs.
- `src/losses.py` - Contains the custom `sparse_graph_smoothness` regularizer and updated reconstruction losses.
- `src/metagene_decoders.py` - Modified probabilistic decoders featuring dynamic weight extraction for topological pairing.
- `Imputation/` - Legacy R-based exploration scripts and TEEBoT data environments.

---

## 📚 Citations & Acknowledgements

This architecture integrates and builds upon the foundational research from the following publications. If you use this code in your research, please cite both:

**HYFA (Hypergraph Factorization):**

```bibtex
@article{vinas2023hypergraph,
  title={Hypergraph factorization for multi-tissue gene expression imputation},
  author={Vi{\~n}as, Ramon and Joshi, Chaitanya K and Georgiev, Dobrik and Lin, Phillip and Dumitrascu, Bianca and Gamazon, Eric R and Li{\`o}, Pietro},
  journal={Nature Machine Intelligence},
  pages={1--15},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

**TEEBoT (Tissue Expression Estimation):**

```bibtex
@article{basu2021predicting,
  title={Predicting tissue-specific gene expression from whole blood transcriptome},
  author={Basu, Mahashweta and Wang, Kun and Ruppin, Eytan and Hannenhalli, Sridhar},
  journal={Science Advances},
  volume={7},
  number={14},
  pages={eabd6991},
  year={2021},
  publisher={American Association for the Advancement of Science}
}
```