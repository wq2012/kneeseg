# Experiments

This directory contains standalone scripts to reproduce the experiments using the `kneeseg` package.

## Contents
- `train.py`: Script to train the full pipeline (Bone Pass 1 -> Bone Pass 2 -> Cartilage).
- `inference.py`: Script to run inference and evaluation.
- `run_pipeline.py`: Unified script to orchestrate the pipeline.
- `models/`: Trained model files (Ignored in git).
- `predictions/`: Prediction outputs (Ignored in git).

## Usage
These scripts are wrappers around the `kneeseg` library. You should run them with a configuration file:

```bash
# Training
python3 delivery/experiments/train.py path/to/your_config.json

# Inference
python3 delivery/experiments/inference.py path/to/your_config.json
```

## Model Checkpoints (`models/`)
The training pipeline generates serialized Random Forest models:

1.  **`bone_rf_p1.joblib`** (Bone Pass 1):
    - **Input**: Image intensity & spatial features.
    - **Output**: Initial probability maps for Bones.

2.  **`bone_rf_p2.joblib`** (Bone Pass 2):
    - **Input**: Original features + Probability maps from Pass 1.
    - **Output**: Refined bone masks.

3.  **`cartilage_rf_p1.joblib`** (Cartilage Pass 1):
    - **Input**: Image features + Signed Distance Transforms (from Bone Masks).
    - **Output**: Initial cartilage predictions.

4.  **`cartilage_rf_p2.joblib`** (Cartilage Pass 2):
    - **Input**: Features + Probability maps from Cartilage Pass 1.
    - **Output**: Final cartilage segmentation.
