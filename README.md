# kneeseg: Knee Bone & Cartilage Segmentation in 3D MRI

[![Python Package](https://github.com/wq2012/kneeseg/actions/workflows/python-package.yml/badge.svg)](https://github.com/wq2012/kneeseg/actions/workflows/python-package.yml)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/collections/wq2012/knee-mri-segmentation)
[![PyPI Version](https://img.shields.io/pypi/v/kneeseg.svg)](https://pypi.python.org/pypi/kneeseg)
[![Python Versions](https://img.shields.io/pypi/pyversions/kneeseg.svg)](https://pypi.org/project/kneeseg)
[![Downloads](https://static.pepy.tech/badge/kneeseg)](https://pepy.tech/project/kneeseg)


**kneeseg** is a Python reimplementation of the paper "Semantic Context Forests for Learning-Based Knee Cartilage Segmentation in 3D MR Images". [[paper](resources/knee_MCV_2013_paper.pdf)] [[slides](resources/knee_MCV_2013_slides.pdf)] [[poster](resources/knee_MCV_2013_poster.pdf)]

|                 | Original MICCAI Workshop Paper | This Implementation |
|-----------------|----------------|---------------------|
| **Bone Segmentation** | Active Shape Model (Siemens proprietary) | Multi-Stage Physical-Space Auto-Context Forest + Subchondral Surface Refiner (Python) |
| **Cartilage Segmentation** | [Semantic Context Forest (C++)](https://github.com/wq2012/DecisionForest) | Multi-Stage Semantic Context Forest + Articular Atlas & Subchondral Calibration (Python) |
| **Dataset** |  Osteoarthritis Initiative (OAI) | [SKI10](https://ski10.grand-challenge.org/) & Downsampled OAI (6 Structures) |
| **Bone DSC** | ~95% | **96.9% – 97.4% (SKI10) / 94.1% – 96.7% (OAI)** |
| **Cartilage DSC** | ~83% | **80.0% – 80.3% (SKI10) / 82.6% – 88.1% (OAI)** |


## Table of Contents
1. [Installation](#installation)
2. [Dataset Support](#dataset-support)
3. [Pretrained Models](#pretrained-models)
4. [Usage](#usage)
    - [As a Library](#as-a-library)
    - [Loading Pretrained Models](#loading-pretrained-models)
    - [Running the Pipeline (CLI)](#running-the-pipeline-cli)
5. [Configuration](#configuration)
    - [Structure](#structure)
    - [Example Configuration](#example-configuration)
6. [Experiments](#experiments)
    - [Experiments Folder](#experiments-folder)
    - [SKI10 Experiment Results](#ski10-experiment-results)
    - [Downsampled OAI Experiment Results](#downsampled-oai-experiment-results)
    - [Unified Cross-Domain (SKI10 + OAI) Experiment Results](#unified-cross-domain-ski10--oai-experiment-results)
7. [Algorithm Details](#algorithm-details)
    - [Bone Segmentation](#bone-segmentation-multi-stage-physical-space-auto-context-forest)
    - [Cartilage Segmentation](#cartilage-segmentation-multi-stage-semantic-context-forest--out-of-fold-subchondral-calibration)
8. [Citation](#citation)

## Installation

You can install the package via pip:

```bash
pip install kneeseg
```

## Dataset Support

This package supports two major dataset structures for knee segmentation:

### 1. SKI10 (Original)
The standard dataset used for the MICCAI 2010 challenge.
-   **Target structures**: Femur, Tibia, Femoral Cartilage, Tibial Cartilage.
-   **Label Mapping**:
    -   `0`: Background
    -   `1`: Femur
    -   `2`: Femoral Cartilage
    -   `3`: Tibia
    -   `4`: Tibial Cartilage

### 2. OAI (Refactored)
The Osteoarthritis Initiative dataset, refactored to extend the SKI10 schema.
-   **Target structures**: Adds Patella and Patellar Cartilage.
-   **Label Mapping**:
    -   `0`: Background
    -   `1`: Femur
    -   `2`: Femoral Cartilage
    -   `3`: Tibia
    -   `4`: Tibial Cartilage
    -   `5`: Patella
    -   `6`: Patellar Cartilage

> **Note**: The pipeline adapts its behavior (number of classes, target structures) based on the `target_bones` configuration.


## Pretrained Models

We provide pretrained models for SKI10, downsampled OAI, and unified cross-domain (SKI10 + OAI) in our [Knee MRI Segmentation Hugging Face Collection](https://huggingface.co/collections/wq2012/knee-mri-segmentation) under the **MIT License**. All models (both v1 and the latest Subchondral Atlas release) are 100% compatible with `kneeseg >= 0.2.5`.

### SKI10 (4 Structures)

Models supporting segmentation of femur, tibia, femoral cartilage, and tibial cartilage:

* **[wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas)** (**Recommended for SKI10**): Multi-Stage Physical-Space Auto-Context Forest + Binary Subchondral Articular Surface Refiner (`RefBone V1` 108-feature + `RefBone V2` 156-feature) + 8-Table 4D Articular Coordinate Atlas + Multi-Family (`103–167` feature) Stage-3 Boundary Refiner & Out-of-Fold Calibrator (trained on the 80 SKI10 training volumes; achieves **96.89% Femur DSC**, **97.42% Tibia DSC**, **80.26% Femoral Cartilage DSC**, and **80.01% Tibial Cartilage DSC** on the 20 held-out SKI10 evaluation volumes).
* [wq2012/knee_3d_mri_segmentation_SKI10](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10) (Baseline v1): Dense Auto-Context Random Forest & Semantic Context Forest trained on all 100 SKI10 training images.

### Downsampled OAI (6 Structures)

Models trained on the 128 OAI training images (`kneeseg/data/oai_split.json`), supporting 6-structure segmentation of femur, tibia, patella, femoral cartilage, tibial cartilage, and patellar cartilage:

* **[wq2012/knee_3d_mri_segmentation_OAI_downsampled_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_OAI_downsampled_subchondral_atlas)** (**Recommended for OAI**): 3-Bone Pass-0/Tier-A Physical-Space Forest + Patellofemoral Specialist + 158-Feature Binary Subchondral Surface Refiner (`RefBone V2`) + 3-Compartment 11-Table 4D Articular Coordinate Atlas & 101-Feature Stage-3 Subchondral Boundary Refiner (trained on the 128 OAI training volumes; achieves **94.08% Femur DSC**, **96.69% Tibia DSC**, **96.39% Patella DSC**, **88.11% Femoral Cartilage DSC**, **86.98% Tibial Cartilage DSC**, and **82.63% Patellar Cartilage DSC** on the 31 held-out OAI evaluation volumes).
* [wq2012/knee_3d_mri_segmentation_OAI_downsampled](https://huggingface.co/wq2012/knee_3d_mri_segmentation_OAI_downsampled) (Baseline v1): Dense Auto-Context Random Forest & Semantic Context Forest trained on the 128 OAI training images.

### Unified Cross-Domain SKI10 + OAI (4 / 6 Structures)

* **[wq2012/knee_3d_mri_segmentation_SKI10_OAI_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_OAI_subchondral_atlas)** (**Recommended Single Model for Both Datasets**): Unified Dual-Domain Router & Cross-Domain Atlas Regularized Cascade trained across all **208 training volumes** (80 SKI10 + 128 OAI), combining resolution-specific specialist forests with physical-volume-weighted 208-scan cross-domain bone refiners and articular coordinate atlases ($\alpha \in [0.015, 0.02]$) plus automatic resolution- and geometry-aware domain routing (`_is_oai_domain`). Achieves **94.08% / 96.70% / 96.39% Bone DSC** and **88.11% / 87.00% / 82.62% Cartilage DSC** (`90.82%` overall 6-structure mean) on the 31 held-out OAI scans and **96.89% / 97.42% Bone DSC** and **80.25% / 79.97% Cartilage DSC** on the 20 held-out SKI10 scans.

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
pred_mask, prob_map = bone_rf.predict(image, spacing=spacing)

# 4. Save Result
save_volume(pred_mask, 'output/prediction.mhd', metadata={'spacing': spacing})
```

### Loading Pretrained Models

To use models you have trained or downloaded from Hugging Face (`wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas`, `wq2012/knee_3d_mri_segmentation_OAI_downsampled_subchondral_atlas`, `wq2012/knee_3d_mri_segmentation_SKI10_OAI_subchondral_atlas`, `wq2012/knee_3d_mri_segmentation_SKI10`, or `wq2012/knee_3d_mri_segmentation_OAI_downsampled`), simply use the `load()` method. `BoneClassifier.load()` and `CartilageClassifier.load()` automatically detect both the latest Subchondral Atlas checkpoints and the v1 raw `RandomForestClassifier` checkpoints:

```python
import numpy as np
from kneeseg.io import load_volume, save_volume
from kneeseg.bone_rf import BoneClassifier
from kneeseg.rf_seg import CartilageClassifier

image, spacing = load_volume("path/to/image-004.mhd", return_spacing=True)

# 1. Initialize empty classifiers
bone_p1 = BoneClassifier()
bone_p2 = BoneClassifier()
cart_p1 = CartilageClassifier()
cart_p2 = CartilageClassifier()

# 2. Load the weights
bone_p1.load("path/to/bone_rf_p1.joblib")
bone_p2.load("path/to/bone_rf_p2.joblib")
cart_p1.load("path/to/cartilage_rf_p1.joblib")
cart_p2.load("path/to/cartilage_rf_p2.joblib")

# 3. Run the 4-stage cascade
pred_b1, prob_b1 = bone_p1.predict(image, spacing=spacing)
pred_b2, prob_b2 = bone_p2.predict(image, prob_map=prob_b1, spacing=spacing)
bone_masks = {name: (pred_b2 == lbl).astype("uint8") for name, lbl in bone_p2.bone_Map.items()}
prob_c1 = cart_p1.predict_proba_map(image, bone_masks, spacing=spacing)
cart_pred, _ = cart_p2.predict(image, bone_masks, prob_map=prob_c1, spacing=spacing)

# 4. Merge into final multi-class label volume (1=Femur, 2=FemCart, 3=Tibia, 4=TibCart)
final_seg = np.zeros_like(pred_b2, dtype=np.uint8)
final_seg[pred_b2 == 1] = 1
final_seg[pred_b2 == 2] = 3
final_seg[pred_b2 == 3] = 5
final_seg[cart_pred > 0] = cart_pred[cart_pred > 0]
save_volume(final_seg, "output/prediction-004.mhd", metadata={"spacing": spacing})
```

### Running the Pipeline (CLI)

The package provides a command-line interface `kneeseg-pipeline` as well as JSON-configured scripts in `experiments/` using the configuration templates in `kneeseg/configs/` (`ski10_full_train_config.json`, `ski10_full_inference_config.json`, `oai_full_inference_config.json`).

**Prerequisites:**
- **Data Structure**: The pipeline expects two directories:
    1.  `.../images`: Contains `.mhd` image files.
    2.  `.../images_labels`: Contains `.mhd` label files (folder name must be `image_dir` + `_labels`).
- **File Naming**: Files must match the SKI10 naming convention (e.g., `image-001.mhd`, `labels-001.mhd`) as defined in `kneeseg/data/ski10_full_split.json`.

**Command:**
```bash
# Point to your image directory. Expects sibling directory with "_labels" suffix.
kneeseg-pipeline --data-dir /path/to/SKI10/data/images

# Or run baseline training / inference directly with a JSON config file:
python3 experiments/train.py kneeseg/configs/ski10_full_train_config.json
python3 experiments/inference.py kneeseg/configs/ski10_full_inference_config.json
```

**Workflow:**
1.  **Training**: Checks if models exist in `experiments/models`. If not, `experiments/train.py` trains the baseline 2-pass cascade on the 80 training cases (to train the full multi-stage Subchondral Atlas checkpoints from scratch, use `experiments/build_bones_and_refiner_v2.py`, `experiments/train_cartilage_v6_ultimate.py`, `experiments/train_oai_bones_v2.py`, `experiments/train_oai_cartilage_v6.py`, and `experiments/train_unified_ski10_oai.py`).
2.  **Inference**: Checks if `evaluation_report.json` exists in `experiments/predictions`. If not, `kneeseg/pipeline/inference.py` (`inference_improved`) runs inference across the evaluation cases (automatically detecting both Baseline v1 and Subchondral Atlas checkpoints).

> For advanced usage and reproduction scripts (e.g., training Subchondral Atlas models from scratch or running benchmark evaluations), please refer to the [Experiments Documentation](experiments/README.md).

## Configuration

The pipeline relies on a JSON configuration file to define data paths and model parameters. You can create your own config file for custom experiments.

### Structure
A valid configuration file has three main sections:

1.  **`data_config`**: Paths to your data and split files.
2.  **`training_config`**: Parameters for Random Forest training (e.g., number of trees).
3.  **`model_config`**: Configuration for model architecture (target bones) and storage directory.
4.  **`output_config`**: Directory for saving predictions.

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
            "n_jobs": -1
        },
        "cartilage_parameters": {
            "n_estimators": 100,
            "max_depth": 20,
            "training_proximity_mm": 15.0
        }
    },
    "model_config": {
        "target_bones": ["femur", "tibia", "patella"],
        "model_directory": "/path/to/save/models",
        "dtype": "bfloat16"
    },
    "output_config": {
        "prediction_directory": "/path/to/save/predictions"
    }
}
```

### Configuration Keys
-   **`target_bones`**: List of bones to segment.
    -   Default for SKI10: `["femur", "tibia"]`
    -   Default for OAI: `["femur", "tibia", "patella"]`
    -   *Cartilage is closely coupled: "femur" includes "femoral cartilage".*
-   **`dtype`**: (Optional) Data type for feature extraction matrices.
    -   Options: `"float32"` (default), `"bfloat16"`.
    -   **Recommendation**: Use `"bfloat16"` to reduce memory usage by ~50%. Requires `ml_dtypes`.

> **Note**: The `split_file` should be a JSON containing `{"train": ["file1.mhd", ...], "eval": ["file2.mhd", ...]}`.

## Experiments

### Experiments Folder
The `experiments/` directory contains reproducible scripts for the SKI10, OAI, and Unified cross-domain experiments, and stores the output models (`models/`, `models_oai/`, `models_unified/`) and predictions (`predictions*/`). See [experiments/README.md](experiments/README.md) for details.

### SKI10 Experiment Results

#### Data Split & Evaluation Protocol
Since the [SKI10 dataset](https://ski10.grand-challenge.org/) does not provide ground-truth labels for its official competition test set, we evaluated the pipeline on the fixed **80/20 split** (`kneeseg/data/ski10_full_split.json`: 80 training scans, 20 held-out evaluation scans) of the 100 labeled SKI10 scans. All tree classifiers, 4-fold Out-Of-Fold (OOF) stacking probabilities, Leave-One-Out (LOO) 4D articular atlases, and the OOF Ridge `(th, bp)` calibrator were fit exclusively on the 80 training scans; post-processing ensemble blend weights and scalar decision thresholds were selected on the 20-scan validation split.

#### Metrics (`N = 20` Held-Out SKI10 Volumes)
| Structure | Baseline v1 DSC (`ddof=0`) | Subchondral Atlas DSC (`ddof=0` / `ddof=1`) | 95% Bootstrap CI (`B = 10,000`) | Absolute Improvement |
|-----------|--------------|----------------------------|---------------------------------|----------------------|
| **Femur (`1`)** | 0.9046 ± 0.0361 | **0.9689 ± 0.0161** (`± 0.0165`) | `[0.9614, 0.9752]` | **+6.43%** |
| **Tibia (`3`)** | 0.9292 ± 0.0260 | **0.9742 ± 0.0080** (`± 0.0082`) | `[0.9705, 0.9776]` | **+4.50%** |
| **Femoral Cartilage (`2`)** | 0.6767 ± 0.0481 | **0.8026 ± 0.0504** (`± 0.0517`) | `[0.7788, 0.8231]` | **+12.59%** |
| **Tibial Cartilage (`4`)** | 0.6411 ± 0.0540 | **0.8001 ± 0.0404** (`± 0.0414`) | `[0.7816, 0.8167]` | **+15.90%** |

> **Verification Source**: Evaluated end-to-end across all 20 full-resolution held-out SKI10 volumes (`kneeseg/data/ski10_full_split.json`) through the 4 standard model checkpoints (`bone_rf_p1.joblib -> bone_rf_p2.joblib -> cartilage_rf_p1.joblib -> cartilage_rf_p2.joblib`) via `kneeseg/pipeline/inference.py` (`experiments/predictions/evaluation_report.json`). Both population standard deviation (`ddof=0`, matching `np.std`) and unbiased sample standard deviation (`ddof=1`) are reported alongside 95% non-parametric percentile bootstrap confidence intervals (`10,000` resamples, `seed=42`).

#### Evaluation Set
The following 20 cases were held out for evaluation:
`image-004`, `image-005`, `image-012`, `image-014`, `image-015`, `image-018`, `image-028`, `image-029`, `image-030`, `image-032`, `image-036`, `image-055`, `image-065`, `image-070`, `image-076`, `image-082`, `image-087`, `image-089`, `image-095`, `image-098`.

### Downsampled OAI Experiment Results

#### Data Split & Longitudinal Subject Analysis
After downsampling to `112 x 140 x 140` (`1.0 x 1.0 x 1.0 mm`) and filtering out 17 incomplete 4-label scans, the 159 complete 6-label volumes span **84 unique subjects** (most with baseline `_V00` and follow-up `_V01` longitudinal visits). The repository split (`kneeseg/data/oai_split.json`) partitions these 159 scans at the **scan level** into **128 training scans (82 unique subjects)** and **31 evaluation scans (30 unique subjects)**. Because the split was constructed at the scan level rather than grouped by patient ID, 28 of the 31 evaluation scans have a companion longitudinal visit (`_V00` or `_V01`) of the same subject in the training set, while 3 evaluation scans (`image-9992358_V01.mhd`, `image-9917803_V00.mhd`, `image-9917803_V01.mhd`) come from strictly unseen subjects. On the 3 strictly unseen-subject scans, the model achieves **0.9691 ± 0.0130 Femur**, **0.9585 ± 0.0029 Tibia**, **0.9629 ± 0.0123 Patella**, **0.8679 ± 0.0157 Femoral Cartilage**, **0.8630 ± 0.0205 Tibial Cartilage**, and **0.8435 ± 0.0234 Patellar Cartilage DSC** (`ddof=1`), compared to **0.9378**, **0.9678**, **0.9640**, **0.8825**, **0.8705**, and **0.8244 DSC** on the 28 seen-subject companion visits.

#### Metrics (`N = 31` Held-Out OAI Evaluation Volumes)

| Structure | Baseline v1 DSC (`ddof=0`) | Subchondral Atlas DSC (`ddof=0` / `ddof=1`) | 95% Bootstrap CI (`B = 10,000`) | Absolute Improvement |
| --- | --- | --- | --- | --- |
| **Femur (`1`)** | 0.7130 ± 0.0673 | **0.9408 ± 0.0339** (`± 0.0345`) | `[0.9284, 0.9523]` | **+22.78%** |
| **Tibia (`3`)** | 0.7545 ± 0.0598 | **0.9669 ± 0.0129** (`± 0.0131`) | `[0.9623, 0.9714]` | **+21.24%** |
| **Patella (`5`)** | 0.5209 ± 0.0831 | **0.9639 ± 0.0210** (`± 0.0214`) | `[0.9561, 0.9709]` | **+44.30%** |
| **Femoral Cartilage (`2`)** | 0.5171 ± 0.0716 | **0.8811 ± 0.0188** (`± 0.0191`) | `[0.8746, 0.8877]` | **+36.40%** |
| **Tibial Cartilage (`4`)** | 0.4134 ± 0.0888 | **0.8698 ± 0.0230** (`± 0.0234`) | `[0.8615, 0.8776]` | **+45.64%** |
| **Patellar Cartilage (`6`)** | 0.3633 ± 0.1406 | **0.8263 ± 0.0677** (`± 0.0688`) | `[0.8016, 0.8491]` | **+46.30%** |
| **3-Bone Mean** | 0.6628 | **0.9572** | — | **+29.44%** |
| **3-Cartilage Mean** | 0.4313 | **0.8591** | — | **+42.78%** |
| **Overall 6-Structure Mean** | 0.5470 | **0.9081** | — | **+36.11%** |

> **Verification Source**: Evaluated end-to-end across all 31 held-out OAI test volumes (`kneeseg/data/oai_split.json`) through `bone_rf_p1.joblib -> bone_rf_p2.joblib -> cartilage_rf_p1.joblib -> cartilage_rf_p2.joblib` via `experiments/evaluate_final_oai.py` (`experiments/predictions_oai/evaluation_report.json`).

### Unified Cross-Domain (SKI10 + OAI) Experiment Results

The unified model ([wq2012/knee_3d_mri_segmentation_SKI10_OAI_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_OAI_subchondral_atlas)) is trained across all 80 SKI10 training scans and 128 OAI training scans (208 training scans total, combining resolution-specific multi-stage specialists via `_is_oai_domain` routing with physical-volume-weighted 208-scan cross-domain bone refiners blended at $\alpha = 0.02$ on SKI10 tibia and $\alpha_F = 0.06, \alpha_T = 0.15$ on OAI femur/tibia, plus 208-scan 4D articular coordinate cartilage priors blended at $\alpha = 0.015$ on OAI) and evaluated end-to-end via `experiments/evaluate_final_unified.py` on both the 31 held-out OAI test scans (`experiments/predictions_unified_oai/evaluation_report.json`) and the 20 held-out SKI10 test scans (`experiments/predictions_unified_ski10/evaluation_report.json`):

| Dataset (Eval Split) | Femur (`1`) | Tibia (`3`) | Patella (`5`) | Femoral Cartilage (`2`) | Tibial Cartilage (`4`) | Patellar Cartilage (`6`) |
| --- | --- | --- | --- | --- | --- | --- |
| **OAI Downsampled (31 Test Scans, `ddof=0`)** | **0.9408 ± 0.0341** | **0.9670 ± 0.0130** | **0.9639 ± 0.0210** | **0.8811 ± 0.0188** | **0.8700 ± 0.0228** | **0.8262 ± 0.0679** |
| **SKI10 Full-Resolution (20 Test Scans, `ddof=0`)** | **0.9689 ± 0.0161** | **0.9742 ± 0.0080** | *N/A (2-bone)* | **0.8025 ± 0.0505** | **0.7997 ± 0.0403** | *N/A (2-cartilage)* |

## Algorithm Details

### Bone Segmentation (Multi-Stage Physical-Space Auto-Context Forest)
1.  **Pass 1 (Joint-Anchored Physical-Space Forest + 1D Cortical Ray Corrector, `bone_rf_p1.joblib`)**:
    - **Features**: Percentile-normalized MRI intensity, multi-scale physical-space Gaussian smoothed intensities and gradient magnitudes, joint-anchored 3D coordinate frames, 3D probabilistic bone atlases, and physical-millimeter Random Shift Intensity Differences (RSID), cascaded through a 1D cortical enclosure ray global corrector (plus a dedicated anterior-femur-anchored Patellofemoral Specialist on 6-structure OAI volumes).
    - **Target**: Multi-class bone classification (`Background`, `Femur`, `Tibia`, plus `Patella` on OAI) with largest-connected-component topological cleanup and axial medullary shaft hole filling.
2.  **Pass 2 (Boundary-Shell Auto-Context Refinement + Binary Subchondral Surface Refiner, `bone_rf_p2.joblib`)**:
    - **Features**: Fine-scale physical Gaussian/gradient features, Pass-1 probability maps, exact physical Signed Distance Transforms, and a **binary (`Bone` vs. `Non-Bone`, `classes_ = [0, 1]`) subchondral articular surface refiner (`RefBone V2`)** operating on the boundary shell (`|d_bone| <= 3.0 mm` with 156 features on SKI10, blended with the 108-feature `RefBone V1` refiner; and `|d_bone| <= 3.0 mm` / `3.5 mm` with 158 features on OAI) that samples 1D surface-normal intensity, gradient, and local z-score profiles along $d \in [-4.5, +4.5]\,\mathrm{mm}$ to resolve partial-volume blending along the femoral condyles, tibial plateau, and patellar facets.
    - **Performance**: Achieves **0.9689 ± 0.0161 DSC** on Femur and **0.9742 ± 0.0080 DSC** on Tibia (SKI10), and **0.9408 / 0.9669 / 0.9639 DSC** on Femur / Tibia / Patella (OAI).

### Cartilage Segmentation (Multi-Stage Semantic Context Forest & Out-of-Fold Subchondral Calibration)
1.  **Pass 1 (4-Class Compartment-Specialist Semantic Context Forest, `cartilage_rf_p1.joblib`)**:
    - **Features**: Exact physical-space Signed Distance Transforms (`d_F, d_T, d_P`), surface-normal ray intensity profiles (`s in [-2.5, +4.0] mm` / `[-3.6, +5.6] mm`) capturing the dark cortical bone plate and bright cartilage layer, Leave-One-Out 4D articular coordinate probabilistic atlases (8 lookup tables on SKI10, 11 tables on OAI across medial/lateral/patellofemoral compartment frames), 18 deterministic anatomical surface landmarks per bone, structure tensor eigenvalues, and physical-space RSID.
    - **Target**: Explicit **4-class competitive taxonomy** (`Background`, `Subchondral Bone / Meniscus`, `Target Cartilage`, `Opposing Cartilage`) within the articular band so subchondral bone cortex and meniscus compete directly against cartilage along the bone-cartilage interface.
2.  **Pass 2 & Stage 3 (Auto-Context Ensemble + Out-Of-Fold Boundary Refinement & Subchondral Calibration, `cartilage_rf_p2.joblib`)**:
    - **Features**: Pass-1 4-class Out-Of-Fold probability maps, normal-ray probability profiles, 3D probability gradients, and physical-millimeter RSPD context offsets. On SKI10, Pass 2 ensembles complementary models trained on `RefBone V1` (`m_v5`, 481 features), `RefBone V2` (`m_v6`, 481 features), and a tibial plateau specialist (`m_v7`, 597 features), followed by a 4-fold Out-Of-Fold Stage-3 boundary refiner combining three feature-family specialists across `103–167` features (`s3_clean`: `127/137` features; `f3_models`: `151/167` features; `pure_models`: `103` features) and an 80-train-only `StandardScaler + Ridge` calibrator that predicts the per-scan tibial cartilage decision threshold `th` (`Ridge(alpha=20.0)`) and subchondral bone penalty `bp` (`Ridge(alpha=10.0)`) from `10` and `9` selected scan-level probability and geometry statistics (out of `45` and `17` candidate statistics), while femoral cartilage uses fixed calibrated parameters (`w = 0.25`, `bp = 0.14`, `th = 0.40`). On OAI, Pass 2 (566 features) is followed by a 101-feature Stage-3 compartment boundary refiner (92 base features + 9 Pass-2 probability context features) and 128-train OOF-calibrated compartment blend weights (`0.42`, `0.44`, `0.44`) and thresholds (`0.56`, `0.56`, `0.54`).
    - **Performance & Runtime**: Achieves **0.8026 ± 0.0504 DSC** on Femoral Cartilage and **0.8001 ± 0.0404 DSC** on Tibial Cartilage on SKI10, and **0.8811 / 0.8698 / 0.8263 DSC** on Femoral / Tibial / Patellar Cartilage on OAI. Single-volume cold-start CPU inference runs in ~2–4 minutes (~18–25 seconds per volume amortized across multi-worker batch evaluation on a 56-core CPU workstation, with ~4–8 GB peak RAM).

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