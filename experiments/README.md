# Experiments

This directory contains standalone scripts to reproduce the experiments using the `kneeseg` package.

## Contents
- `train.py`: Script to train the library pipeline (`BoneClassifier` Pass 1 -> Pass 2 -> `CartilageClassifier` Pass 1 -> Pass 2).
- `inference.py`: Script to run library inference and evaluation.
- `run_pipeline.py`: Unified script to orchestrate the pipeline.
- `build_bones_and_refiner_v2.py`: Script to train the 4-fold Out-Of-Fold Bone Pass-1/Pass-2 classifiers (`bone_rf_p1.joblib`) and the 4-class Subchondral Articular Surface Refiner (`RefBone V2`, packaged into `bone_rf_p2.joblib`) on the 80 training scans.
- `train_cartilage_v6_ultimate.py`: Script to train the 4-class compartment-specialist Semantic Context Forest + HistGradientBoosting / ExtraTrees Pass-1 (`cartilage_rf_p1.joblib`) and Pass-2 ensemble strictly on the 80 training scans.
- `train_tibia_v7_fast.py`: Script to train the Tibia V7 specialist ensemble strictly on the 80 training scans.
- `build_train80_tibia_prep.py`: Script to precompute 80-train Leave-One-Out atlas and Pass-1/Pass-2 ROI features.
- `build_clean_stage3_train80.py`: Helper module for 136-dim Stage-3 boundary candidate feature extraction.
- `train_full_oof_stage3.py`: Script to train the 4-fold Out-Of-Fold 136-dim Stage-3 boundary classifier on the 80 training scans.
- `train_tibia_calibrator_train80.py`: Script to train the 80-train-only `StandardScaler + Ridge` `(th, bp)` threshold and subchondral bone-penalty calibrator (`th_base = 0.508`, packaged into `cartilage_rf_p2.joblib`).
- `evaluate_final_ski10.py`: End-to-end cache-free evaluation script that runs `kneeseg.pipeline.inference.inference_improved()` across all 20 held-out SKI10 volumes (`predictions/labels-*.mhd`) and writes `predictions/evaluation_report.json`.
- `models/`: Trained model files (`bone_rf_p1.joblib`, `bone_rf_p2.joblib`, `cartilage_rf_p1.joblib`, `cartilage_rf_p2.joblib`; ignored in git).
- `predictions/`: Prediction outputs (Ignored in git).

## Usage
These scripts are wrappers around the `kneeseg` library. You should run them with a configuration file:

```bash
# Training
python3 experiments/train.py path/to/your_config.json

# Inference
python3 experiments/inference.py path/to/your_config.json

# Reproduce Final SKI10 20-Case Evaluation (FB=0.9689, TB=0.9742, FC=0.8026, TC=0.8001)
python3 experiments/evaluate_final_ski10.py
```

## Model Checkpoints (`models/`)
All model weights are self-contained inside the 4 standard checkpoints in `experiments/models/` (published on Hugging Face at [wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas), and also backward-compatible with [wq2012/knee_3d_mri_segmentation_SKI10](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10) and [wq2012/knee_3d_mri_segmentation_OAI_downsampled](https://huggingface.co/wq2012/knee_3d_mri_segmentation_OAI_downsampled)):

1.  **`bone_rf_p1.joblib`** (Bone Pass 1, `stage='v2_p1'`):
    - **Input**: 3D MRI volume & physical voxel spacing.
    - **Output**: Pass-1 bone segmentation and 3-class probability map (`Background`, `Femur`, `Tibia`).

2.  **`bone_rf_p2.joblib`** (Bone Pass 2 + Subchondral Surface Refiner, `stage='v2_p2'`):
    - **Input**: 3D MRI volume, Pass-1 bone probability map, and voxel spacing.
    - **Output**: Refined bone masks (`RefBone V2`) and probability map.

3.  **`cartilage_rf_p1.joblib`** (Cartilage Pass 1, `stage='v6_p1'`):
    - **Input**: 3D MRI volume, Pass-2 bone masks, and voxel spacing.
    - **Output**: Pass-1 cartilage probability maps (`predict_proba_map`) and Pass-1 segmentation (`predict`).

4.  **`cartilage_rf_p2.joblib`** (Cartilage Pass 2 + Stage 3 + Subchondral Calibrator, `stage='v6_p2'`):
    - **Input**: 3D MRI volume, Pass-2 bone masks, Pass-1 cartilage probability map, and voxel spacing.
    - **Output**: Final calibrated femoral and tibial cartilage segmentation.
