# kneeseg: Knee Bone & Cartilage Segmentation in 3D MRI

[![Python Package](https://github.com/wq2012/kneeseg/actions/workflows/python-package.yml/badge.svg)](https://github.com/wq2012/kneeseg/actions/workflows/python-package.yml)

**kneeseg** is a Python reimplementation of the paper "Semantic Context Forests for Learning-Based Knee Cartilage Segmentation in 3D MR Images".

|                 | Original MICCAI Workshop Paper | This Implementation |
|-----------------|----------------|---------------------|
| **Bone Segmentation** | Active Shape Model (Siemens proprietary) | Dense Random Forest (Auto-Context) |
| **Cartilage Segmentation** | [Semantic Context Forest (C++)](https://github.com/wq2012/DecisionForest) | Semantic Context Forest (Python) |
| **Dataset** |  Osteoarthritis Initiative (OAI) | [SKI10](https://ski10.grand-challenge.org/) |

## Installation

You can install the package via pip:

```bash
pip install kneeseg
```

## Usage

### As a Library
```python
from kneeseg.io import load_volume
from kneeseg.bone_rf import BoneClassifier
# ...
```

### Running the Pipeline
The package includes a unified pipeline script `kneeseg-pipeline` (or via python module):

```bash
# Run using the installed command
kneeseg-pipeline --data-dir /path/to/SKI10/data
```

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