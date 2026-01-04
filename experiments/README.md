# Experiments

This directory contains standalone scripts to reproduce the experiments using the `kneeseg` package.

## Contents
- `train.py`: Script to train the full pipeline (Bone Pass 1 -> Bone Pass 2 -> Cartilage).
- `inference.py`: Script to run inference and evaluation.
- `run_pipeline.py`: Unified script to orchestrate the pipeline.
- `models/`: Trained model files (Ignored in git).
- `predictions/`: Prediction outputs (Ignored in git).

## Usage
These scripts are designed to be run from the repository root:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 delivery/experiments/run_pipeline.py
```

## Model Checkpoints (`models/`)
The training pipeline generates three serialized Random Forest models using `joblib`:

1.  **`bone_rf_p1.joblib`** (Bone Pass 1):
    - **Input**: Image intensity & spatial features.
    - **Output**: Initial probability maps for Femur and Tibia.
    - **Role**: Provides the "Auto-Context" context for the second pass.

2.  **`bone_rf_p2.joblib`** (Bone Pass 2):
    - **Input**: Original features + Probability maps from Pass 1.
    - **Output**: Refined, final bone masks.
    - **Role**: Solves the bone segmentation task with high accuracy (>0.90 DSC).

3.  **`cartilage_rf.joblib`** (Cartilage):
    - **Input**: Image features + Signed Distance Transforms (from Pass 2 Bone Masks).
    - **Output**: Cartilage class predictions (Femoral/Tibial).
    - **Role**: Segments cartilage within the bone proximity.
