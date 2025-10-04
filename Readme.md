# Weighted Attention-based Gated GNN for Effective Vulnerable Function Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Joern](https://img.shields.io/badge/Joern-1.2.38-orange.svg)](https://joern.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/warvd-repo.svg?style=social&label=Star&color=orange)](https://github.com/yourusername/warvd-repo)

## Overview

This open-source repository implements the innovative framework introduced in the manuscript *"A Weighted Attention-based Graph Neural Approach for Effective Vulnerable Function Detection"*. The core contribution is **WARVD** (**W**eighted **A**ttention-based **R**epresentation approach for **V**ulnerability **D**etection), a state-of-the-art Graph Neural Network (GNN)-based system designed to automate the detection of vulnerable C functions in software codebases.

In an era where software vulnerabilities surge—exceeding 40,000 reported Common Vulnerabilities and Exposures (CVEs) in 2024 alone—WARVD addresses key challenges in cybersecurity by leveraging Code Property Graphs (CPGs) for holistic code analysis. Unlike traditional sequence-based models that overlook structural dependencies, WARVD integrates contextualized semantic embeddings with multi-relational graph structures to achieve superior precision, recall, and F1-scores, even on severely imbalanced real-world datasets (e.g., 1:65 vulnerable-to-non-vulnerable ratios).

This project processes Joern-generated CPGs as input and outputs vulnerability probabilities.

## Key Features

WARVD stands out through its synergistic design, combining deep learning with gradient boosting for robust performance:

- **Fine-tuned CodeBERT Embeddings**: Utilizes a CodeBERT model pre-trained and fine-tuned on C code corpora to produce contextualized node representations. This captures long-range semantic dependencies, far surpassing non-contextual methods like Word2Vec in encoding nuanced code intent (e.g., +29.4% precision gain on real-world benchmarks).
  
- **Edge Type-Encoding and Attention Mechanism**: Encodes diverse edge types (AST for syntax, CFG for control flows, PDG for data dependencies, and call edges for inter-procedural links) while applying weighted attention to prioritize vulnerability-indicative relationships, such as unchecked branches or unsafe API calls.

- **Gated and Multi-Layer Graph Convolutions**: Employs a Gated GNN (GGNN) architecture with layered convolutions to propagate local node features (e.g., variable assignments) into global graph patterns, enabling detection of subtle exploits like integer overflows or null dereferences.

- **Weight Optimization for Imbalance Handling**: Incorporates learnable weighted parameters and cost-sensitive loss functions to amplify the minority vulnerable class, ensuring reliable generalization on sparse real-world data.

- **Hybrid Classification Pipeline**: Extracts high-level representations from WAGGNN (Weighted Attention-based Gated GNN) hidden layers and feeds them into a LightGBM ensemble classifier, mitigating overfitting and boosting efficiency for production-scale deployment.

These features culminate in competitive results: F1-scores of 0.67 on imbalanced real-world datasets and 0.97 on synthetic benchmarks like SARD, outperforming nine state-of-the-art baselines across four open-source projects.

## Repository Structure

```
warvd/
├── data/                  # Sample datasets and preprocessed CPGs
├── src/
│   ├── models/            # WAGGNN architecture and LightGBM integration
│   ├── embeddings/        # CodeBERT fine-tuning and node embedding logic
│   ├── graphs/            # CPG parsing with Joern and edge encoding
│   └── utils/             # Training, evaluation, and imbalance handling utilities
├── experiments/           # Scripts for ablation studies and benchmarks
├── main.py                # Entry point for running the full pipeline
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Requirements

### Software
- **Joern**: Version 1.2.38 (for CPG generation from C source code). Download from [joern.io](https://joern.io/) and ensure it's in your PATH.
- **Python**: 3.11.x (tested on 3.11.5).

### Python Libraries
Install dependencies via:
```
pip install -r requirements.txt
```
Key libraries include:
- `torch` and `torch-geometric` for GNN operations
- `transformers` for CodeBERT
- `lightgbm` for classification
- `networkx` for graph utilities
- `scikit-learn` for metrics and preprocessing

**Note**: GPU acceleration is recommended for training (CUDA 11.8+); CPU fallback is supported but slower.

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/warvd-repo.git
   cd warvd-repo
   ```

2. **Set Up Virtual Environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Install Joern**:
   - Follow instructions at [Joern Documentation](https://docs.joern.io/).
   - Verify: `joern --version` should output 1.2.38.

5. **(Optional) Download CodeBERT Model and Fine-tune it**:
   - Please refer to: https://github.com/microsoft/CodeBERT

## Getting Started

### Quick Run (End-to-End Inference)
To process a sample C function and predict vulnerability:
```
python main.py run full model warvd
```

## Datasets
- **Real-World Dataset**: Custom collection from four open-source C projects (1:65 imbalance). Subset available in `data/real-world/` (anonymized for privacy).
- **SARD**: Synthetic Juliet Test Suite for vulnerability benchmarks. Download full version from [NIST SARD](https://samate.nist.gov/) and preprocess with `src/graphs/preprocess_sard.py`.
- Preprocessing generates CPG JSONs compatible with Joern.

## License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments
- Inspired by Devign \cite{zhou2019devign} and Joern \cite{yamaguchi2014modeling}.
- We would like to thank: https://github.com/saikat107/Devign for providing insights and ideas.

## Contact

For questions, reach out to the authors, please contact via: guanjun.lin@fjsmu.edu.cn. 
We will continue to improve and extend this repo. Thank you.




