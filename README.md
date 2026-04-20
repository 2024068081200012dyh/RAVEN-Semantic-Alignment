# Region-Aware Semantic Alignment (RAVEN)

Official implementation of the paper:  
**"Region-Aware Semantic Alignment for Efficient Open-Vocabulary Visual Detection"** Submitted to *The Visual Computer*.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 📌 Notice

This repository is directly related to the manuscript currently under submission to **The Visual Computer**. We encourage readers to replicate our experiments and evaluate the results. If you find this work helpful, please cite our manuscript.

## 🌟 Highlights

- **LPA-PAN**: A multi-scale fusion hierarchy that injects linguistic priors to enhance visual-semantic correspondence.
- **Learnable Prompt Anchors**: A scale-conditioned prompting mechanism to bridge regional visual features and textual embeddings.
- **TDSR**: A token-wise distillation strategy designed to recover semantic density for small-scale objects.

## 📂 Repository Structure

```text
RAVEN_Core/
├── data/                  # Data loading and mask generation for TDSR
│   └── dataset.py         
├── models/                # Core architecture implementations
│   ├── raven_arch.py      # LPA-PAN, Prompt Anchors, and Unified Model
│   └── distill.py         # Token-wise Distillation (TDSR) logic
├── results/               # Sample heatmaps and detection results
├── main.py                # Main entry for training, evaluation, and visualization
├── environment.yaml       # Conda environment configuration
├── requirements.txt       # Pip dependencies
└── README.md              # Documentation
```

## 🚀 Getting Started

### 1. Environment Setup

We recommend using Conda for environment management:

Bash

```
conda env create -f environment.yaml
conda activate raven_env
```

Alternatively, use pip:

Bash

```
pip install -r requirements.txt
```

### 2. Data Preparation

Please organize the MS COCO and LVIS v1.0 datasets as follows:

```
data/
├── coco/
│   ├── annotations/
│   └── train2017/
└── lvis/
    ├── lvis_v1_train.json
    └── lvis_v1_val.json
```

### 3. Training & Evaluation

To train RAVEN-L on LVIS with default settings:

```
python main.py --mode train --batch_size 16 --lr 1e-4
```

To evaluate the zero-shot generalization on LVIS validation set:

```
python main.py --mode eval --checkpoint ./weights/raven_l.pth
```

### 4. Visualization (Similarity Response Maps)

To verify the **Region-Aware** mechanism and reproduce **Figure 5** in the manuscript, run the visualization script:

```
python main.py --mode visualize --input ./data/sample.jpg
```

The output activation maps (demonstrating precise region-level alignment) will be saved in the `results/` folder.

## 🛠 Deployment & Practical Value

RAVEN is designed with real-time inference in mind. For industrial applications (e.g., robotics or edge devices), you can export the model to ONNX format:

```
python main.py --mode export --output raven.onnx
```

## 📄 Citation

If you use RAVEN in your research, please cite:

```
@article{deng2026raven,
  title={Region-Aware Semantic Alignment for Efficient Open-Vocabulary Visual Detection},
  author={Deng, Yuhan and Tang, Hong},
  journal={The Visual Computer (Under Revision)},
  year={2026}
}
```

## ✉️ Contact

For any questions regarding reproducibility or implementation details, please contact **Yuhan Deng** (2024068081200012@ecjtu.edu.cn) or **Professor Hong Tang** (tanghong@ecjtu.edu.cn).
