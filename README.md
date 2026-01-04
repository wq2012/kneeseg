# kneeseg: Knee Bone & Cartilage Segmentation in 3D MRI

[![Python Package](https://github.com/wq2012/kneeseg/actions/workflows/python-package.yml/badge.svg)](https://github.com/wq2012/kneeseg/actions/workflows/python-package.yml)
[![PyPI Version](https://img.shields.io/pypi/v/kneeseg.svg)](https://pypi.python.org/pypi/kneeseg)
[![Python Versions](https://img.shields.io/pypi/pyversions/kneeseg.svg)](https://pypi.org/project/kneeseg)
[![Downloads](https://static.pepy.tech/badge/kneeseg)](https://pepy.tech/project/kneeseg)

**kneeseg** is a Python reimplementation of the paper "Semantic Context Forests for Learning-Based Knee Cartilage Segmentation in 3D MR Images". [[paper](resources/knee_MCV_2013_paper.pdf)] [[slides](resources/knee_MCV_2013_slides.pdf)]

|                 | Original MICCAI Workshop Paper | This Implementation |
|-----------------|----------------|---------------------|
| **Bone Segmentation** | Active Shape Model (Siemens proprietary) | Dense Random Forest (Auto-Context) |
| **Cartilage Segmentation** | [Semantic Context Forest (C++)](https://github.com/wq2012/DecisionForest) | Semantic Context Forest (Python) |
| **Dataset** |  Osteoarthritis Initiative (OAI) | [SKI10](https://ski10.grand-challenge.org/) |

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Configuration](#configuration)
4. [Experiments Folder](#experiments-folder)
5. [Experiment Results (SKI10)](#experiment-results-ski10)
6. [Algorithm Details](#algorithm-details)
7. [Citation](#citation)

## Installation

You can install the package via pip:

```bash
pip install kneeseg
```

## Usage

### As a Library

You can use `kneeseg` modules directly in your Python scripts to load data, train models, or run inference.

```python
import os
from kneeseg.io import load_volume, save_volume
from kneeseg.bone_rf import BoneClassifier

# 1. Load Data
# Data should be .mhd/.raw or .hdr/.img
image_path = 'data/image-001.mhd'
image, spacing = load_volume(image_path, return_spacing=True)

# 2. Initialize Model
# Example: Initialize the first-pass Bone Classifier
bone_rf = BoneClassifier(n_estimators=100, max_depth=25)

# 3. Predict (assuming pre-trained model loaded or trained)
# bone_rf.load('models/bone_rf_p1.joblib')
pred_mask, prob_map = bone_rf.predict(image)

# 4. Save Result
save_volume(pred_mask, 'output/prediction.mhd', metadata={'spacing': spacing})
```

### Loading Pretrained Models

To use models you have trained or downloaded (e.g., from the Hugging Face release), simply use the `load()` method:

```python
from kneeseg.bone_rf import BoneClassifier
from kneeseg.rf_seg import CartilageClassifier

# 1. Initialize empty classifiers
# (Parameters must match training, or just use defaults if standard)
bone_p1 = BoneClassifier()
bone_p2 = BoneClassifier()
cartilage_rf = CartilageClassifier()

# 2. Load the weights
bone_p1.load("path/to/bone_rf_p1.joblib")
bone_p2.load("path/to/bone_rf_p2.joblib")
cartilage_rf.load("path/to/cartilage_rf.joblib")

# 3. Predict (Example: Bone Pass 1)
pred_p1, prob_p1 = bone_p1.predict(image)
```

### Running the Pipeline (CLI)

The package provides a command-line interface `kneeseg-pipeline` to orchestrate the full training and inference workflow using the SKI10 dataset split.

**Prerequisites:**
- **Data Structure**: The pipeline expects two directories:
    1.  `.../images`: Contains `.mhd` image files.
    2.  `.../images_labels`: Contains `.mhd` label files (folder name must be `image_dir` + `_labels`).
- **File Naming**: Files must match the SKI10 naming convention (e.g., `image-001.mhd`, `labels-001.mhd`) as defined in `kneeseg/data/ski10_full_split.json`.

**Command:**
```bash
# Point to your image directory. Expects sibling directory with "_labels" suffix.
kneeseg-pipeline --data-dir /path/to/SKI10/data/images
```

**Workflow:**
1.  **Training**: Checks if models exist in `experiments/models`. If not, trains using the 60 training cases.
2.  **Inference**: Checks if `evaluation_report.json` exists in `experiments/predictions`. If not, runs inference on the 20 evaluation cases.

> For advanced usage and reproduction scripts (e.g., training models from scratch), please refer to the [Experiments Documentation](experiments/README.md).

## Configuration

The pipeline relies on a JSON configuration file to define data paths and model parameters. You can create your own config file for custom experiments.

### Structure
A valid configuration file has three main sections:

1.  **`data_config`**: Paths to your data and split files.
2.  **`training_config`**: Parameters for Random Forest training (e.g., number of trees).
3.  **`output_config`**: Directories for saving models and predictions.

### Example Configuration

```json
{
    "data_config": {
        "image_directory": "/path/to/images",
        "label_directory": "/path/to/labels",
        "split_file": "/path/to/split.json"
    },
    "training_config": {
        "augmentation": true,
        "bone_parameters": {
            "n_estimators": 100,
            "max_depth": 25,
            "pca_components": 20
        },
        "cartilage_parameters": {
            "n_estimators": 100,
            "max_depth": 20,
            "training_proximity_mm": 15.0
        }
    },
    "output_config": {
        "model_directory": "/path/to/save/models",
        "prediction_directory": "/path/to/save/predictions"
    }
}
```

> **Note**: The `split_file` should be a JSON containing `{"train": ["file1.mhd", ...], "eval": ["file2.mhd", ...]}`.

## Experiments Folder
The `experiments/` directory contains reproduceable scripts and will store the output models (`models/`) and predictions (`predictions/`) if you run the scripts provided there. See [experiments/README.md](experiments/README.md) for details.

## Experiment Results (SKI10)
Since the [SKI10 dataset](https://ski10.grand-challenge.org/) doesn not provide the ground truth labels for its default testing set, we evaluated the pipeline on a **20% hold-out set** (20 cases) from the SKI10 training data (Total 80 cases: 60 Train, 20 Eval).

### Metrics
| Structure | Dice Similarity Coefficient (DSC) |
|-----------|-----------------------------------|
| **Femur** | 0.9046 ± 0.0361 |
| **Tibia** | 0.9292 ± 0.0260 |
| **Femoral Cartilage** | 0.5944 ± 0.0654 |
| **Tibial Cartilage** | 0.5805 ± 0.0533 |

### Evaluation Set
The following 20 cases were held out for evaluation:
`image-004`, `image-005`, `image-012`, `image-014`, `image-015`, `image-018`, `image-028`, `image-029`, `image-030`, `image-032`, `image-036`, `image-055`, `image-065`, `image-070`, `image-076`, `image-082`, `image-087`, `image-089`, `image-095`, `image-098`.

## Algorithm Details

### Bone Segmentation (Dense Auto-Context RF)
1.  **Pass 1**: Dense Random Forest voxel classification.
    - **Features**: Normalized Intensity, Gaussian Smoothed Intensity ($\sigma=2.0, 4.0$), Spatial Coordinates, RSID (20 offsets).
    - **Target**: 3-class classification (Background, Femur, Tibia).

2.  **Pass 2 (Refinement)**: Auto-Context Random Forest.
    - **Features**: All Pass 1 features + **Probabilities from Pass 1**.
    - **Performance**: Achieves >0.90 DSC on Bones.

### Cartilage Segmentation (Semantic Context Forest)
1.  **Feature Extraction**:
    - **Semantic Context**: Signed Distance Transforms (SDT) computed from Pass 2 bone masks.
    - **Texture**: RSID (30 offsets).
    - **Local**: Intensity, Gaussian ($\sigma=1.0$), Gradient.
    - **Arithmetic**: DT Sum/Diff.
2.  **Classification**: Dense Random Forest (100 trees).
    - **Performance**: Achieves ~0.60 DSC.

## Citation

**Plain Text:**

> Quan Wang, Dijia Wu, Le Lu, Meizhu Liu, Kim L. Boyer, and Shaohua Kevin Zhou.
> "Semantic Context Forests for Learning-Based Knee Cartilage Segmentation in 3D MR Images."
> MICCAI 2013: Workshop on Medical Computer Vision.

> Quan Wang.
> Exploiting Geometric and Spatial Constraints for Vision and Lighting Applications.
> Ph.D. dissertation, Rensselaer Polytechnic Institute, 2014.

**BibTeX:**

```bibtex
@inproceedings{wang2013semantic,
  title={Semantic context forests for learning-based knee cartilage segmentation in 3D MR images},
  author={Wang, Quan and Wu, Dijia and Lu, Le and Liu, Meizhu and Boyer, Kim L and Zhou, Shaohua Kevin},
  booktitle={International MICCAI Workshop on Medical Computer Vision},
  pages={105--115},
  year={2013},
  organization={Springer}
}

@phdthesis{wang2014exploiting,
  title={Exploiting Geometric and Spatial Constraints for Vision and Lighting Applications},
  author={Quan Wang},
  year={2014},
  school={Rensselaer Polytechnic Institute},
}
```