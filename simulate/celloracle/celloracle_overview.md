# CellOracle Plan

Source: http://morris-lab.github.io/CellOracle.documentation/index.html
Paper: "Dissecting cell identity via network inference and in silico gene perturbation" — *Nature* (Kamimoto et al. 2023)

---

## What is CellOracle?

CellOracle is a Python library for **in silico gene perturbation analysis** using single-cell omics data and gene regulatory network (GRN) models. Given a transcription factor knockout (or overexpression), it propagates the perturbation signal through a GRN to predict how cell identity shifts — visualized as a vector field on a UMAP/embedding.

Key distinction from SERGIO: CellOracle works on **real scRNA-seq data** and infers GRNs from it (optionally guided by chromatin accessibility), rather than generating synthetic data from a known GRN.

---

## Core Capabilities

- Infer sample-specific GRN models per cell cluster from scRNA-seq + optional scATAC-seq
- Score genes by network centrality (hub gene identification)
- Simulate TF knockout / overexpression → predict downstream gene expression changes
- Visualize perturbation effects as **vector fields** on cell embedding
- Compare simulated shift direction with observed developmental trajectory (pseudotime)
- Lineage-specific perturbation analysis

---

## Pipeline Overview

```
[scRNA-seq data (AnnData/h5ad)]
        |
        v
[Preprocessing: Scanpy/Seurat]  +  [Pseudotime calculation]
        |
        v                               [ATAC-seq data (optional)]
[Oracle object creation]                        |
        |                               [Cicero: cis-reg elements]
        |                               [TSS annotation]
        |                               [TF motif scan (TFinfo)]
        |                                       |
        +------- Base GRN ←────────────────────┘
        |         (or built-in promoter databases for 10+ species,
        |          or custom TF-target lists)
        v
[KNN imputation of scRNA-seq]
        |
        v
[GRN calculation per cell cluster]  ← fits regression model per gene
        |
        v
[Network analysis: graph scores, hub genes]
        |
        v
[Build predictive models]
        |
        v
[In silico TF perturbation (KO or OE)]
        |
        v
[Vector field: predicted cell identity shift per cell]
        |
        v
[Compare with developmental vectors (pseudotime)]
[Lineage-specific analysis]
```

---

## Inputs

| Input | Description | Required? |
|-------|-------------|-----------|
| scRNA-seq AnnData | Preprocessed, normalized, UMAP/embedding computed | Required |
| Pseudotime | Trajectory ordering for comparing simulated vs. observed vectors | Required |
| Base GRN | Prior TF–gene regulatory scaffold | Required (one of 4 options below) |

**Base GRN options (choose one):**
1. **scATAC-seq peaks** via Cicero → TSS annotation → TF motif scan (`TFinfo` class)
2. **Bulk ATAC-seq peaks** → TSS annotation → motif scan
3. **Built-in promoter databases** — pre-built for 13 species (human hg38/hg19, mouse mm10/mm39, rat, zebrafish, Drosophila, C. elegans, yeast, Arabidopsis, chicken, etc.)
4. **Custom TF-target lists** — user-supplied associations

---

## Outputs

| Output | Description |
|--------|-------------|
| GRN models | Cell-cluster-specific regulatory models (TF → gene weights) |
| Network scores | Graph-theory centrality scores per gene (hub identification) |
| Perturbation vectors | Per-cell predicted shift in gene expression space |
| Vector field visualization | Arrows on UMAP showing direction of cell identity change after TF KO/OE |
| Base GRN `.parquet` | Binary TF–gene association matrix (from motif scan) |

---

## Key Classes

- **`Oracle`** — Central object wrapping scRNA-seq AnnData + GRN models; entry point for all analyses
- **`TFinfo`** — Motif analysis module; converts ATAC-seq peaks → DNA sequences → TF binding motif scan; outputs base GRN

---

## Supported Species

13 species: human (hg38, hg19), mouse (mm39, mm10, mm9), rat, guinea pig, pig, zebrafish, Xenopus, Drosophila, C. elegans, S. cerevisiae, Arabidopsis, chicken.

---

## Installation

```bash
pip install celloracle
# or via Docker: morris-lab/celloracle on Docker Hub
```

Reference genome installation (for motif scan path):
```python
import genomepy
genomepy.install_genome("mm10", provider="UCSC")
```

---

## Relevance to This Project

CellOracle is complementary to SERGIO-based pre-pre-training:

| | SERGIO | CellOracle |
|---|---|---|
| Data type | Synthetic scRNA-seq | Real scRNA-seq + optional ATAC |
| GRN role | Known GRN → simulate data | Real data → infer GRN |
| Perturbation | Ground-truth KO used to generate training data | In silico KO to predict cell identity shift |
| Output | Synthetic count matrices for model training | Perturbation vector fields for biological interpretation |

**Potential use cases here:**
- Use CellOracle to infer GRNs from real Replogle-Nadig data → use inferred GRN topology to generate more realistic SERGIO synthetic data
- Use CellOracle's in silico perturbation predictions as an additional training signal or evaluation baseline
- Compare State model predictions (perturbation effect vectors) against CellOracle's vector field predictions on the same dataset

---

## References

- Kamimoto et al. (2023). "Dissecting cell identity via network inference and in silico gene perturbation." *Nature*. https://doi.org/10.1038/s41586-022-05688-9
- GitHub: https://github.com/morris-lab/CellOracle
- Docs: http://morris-lab.github.io/CellOracle.documentation/
