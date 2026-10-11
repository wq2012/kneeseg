# Experiments

This directory contains standalone scripts to reproduce the experiments using the `kneeseg` package.

## Contents
- `train.py`: Script to train the baseline 2-pass library pipeline (`BoneClassifier` Pass 1 -> Pass 2 -> `CartilageClassifier` Pass 1 -> Pass 2).
- `inference.py`: Script to run library inference and evaluation (`kneeseg.pipeline.inference.inference_improved`, supporting both Baseline v1 and Subchondral Atlas checkpoints).
- `run_pipeline.py`: Unified script to orchestrate the baseline training and evaluation pipeline.
- `build_bones_and_refiner_v2.py`: Script to train the 4-fold Out-Of-Fold Bone Pass-1/Pass-2 classifiers (`bone_rf_p1.joblib`) and the binary Subchondral Articular Surface Refiner (`RefBone V1` 108-feature + `RefBone V2` 156-feature, `classes_ = [0, 1]`, packaged into `bone_rf_p2.joblib`) on the 80 SKI10 training scans.
- `train_cartilage_v6_ultimate.py`: Script to train the 4-class compartment-specialist Semantic Context Forest + `HistGradientBoosting` / `ExtraTrees` Pass-1 (`cartilage_rf_p1.joblib`) and Pass-2 ensemble (`m_v5` on `RefBone V1` and `m_v6` on `RefBone V2`) on the 80 SKI10 training scans.
- `train_tibia_v7_fast.py`: Script to train the Tibia V7 (`m_v7`) specialist ensemble on the 80 SKI10 training scans.
- `build_train80_tibia_prep.py`: Script to precompute 80-train Leave-One-Out 4D articular atlas and Pass-1/Pass-2 ROI features.
- `build_clean_stage3_train80.py`: Helper module for Stage-3 boundary candidate feature extraction (`127/137`-feature `s3_clean`, `151/167`-feature `f3_models`, and `103`-feature `pure_models`).
- `train_full_oof_stage3.py`: Script to train the 4-fold Out-Of-Fold multi-family (`103–167` feature) Stage-3 boundary classifier on the 80 SKI10 training scans.
- `train_tibia_calibrator_train80.py`: Script to train the 80-train-only `StandardScaler + Ridge` `(th, bp)` threshold and subchondral bone-penalty calibrator (`th_base = 0.508`, selecting `10` of `45` scan-level statistics for `th` and `9` of `17` statistics for `bp`, packaged into `cartilage_rf_p2.joblib`).
- `evaluate_final_ski10.py`: End-to-end cache-free evaluation script that runs `kneeseg.pipeline.inference.inference_improved()` across all 20 held-out SKI10 volumes (`predictions/labels-*.mhd`) and writes `predictions/evaluation_report.json`.
- `train_oai_bones_v2.py`: Script to train the 3-Bone (`Femur`, `Tibia`, `Patella`) Pass-1 (`bone_rf_p1.joblib`, including the anterior-femur-anchored Patellofemoral Specialist) and Pass-2 158-feature binary Subchondral Surface Refiner (`RefBone V2`, `bone_rf_p2.joblib`) on the 128 OAI training scans (`kneeseg/data/oai_split.json`).
- `train_oai_cartilage_v6.py`: Script to train the 3-Compartment (`Femoral`, `Tibial`, `Patellar` Cartilage) Pass-1 (`cartilage_rf_p1.joblib`, with 11-table LOO 4D Articular Coordinate Atlas) and Pass-2 + 92-feature Stage-3 Subchondral Boundary Refiner (`cartilage_rf_p2.joblib`) on the 128 OAI training scans.
- `evaluate_final_oai.py`: End-to-end cache-free evaluation script that runs `inference_improved()` across all 31 held-out OAI test volumes (`predictions_oai/labels-*.mhd`) and writes `predictions_oai/evaluation_report.json`.
- `train_unified_ski10_oai.py`: Script to build the unified cross-domain SKI10 + OAI model (`models_unified/`) from the 80 SKI10 training scans and 128 OAI training scans (combining resolution-specialist forests with physical-volume-weighted 208-scan cross-domain bone refiners and 4D articular coordinate atlases).
- `evaluate_final_unified.py`: End-to-end cache-free evaluation script that evaluates the unified SKI10 + OAI model across both the 31 held-out OAI test scans (`predictions_unified_oai/evaluation_report.json`) and the 20 held-out SKI10 test scans (`predictions_unified_ski10/evaluation_report.json`).
- `models/`, `models_oai/`, `models_unified/`: Trained model files (`bone_rf_p1.joblib`, `bone_rf_p2.joblib`, `cartilage_rf_p1.joblib`, `cartilage_rf_p2.joblib`; ignored in git).
- `predictions/`, `predictions_oai/`, `predictions_unified_oai/`, `predictions_unified_ski10/`: Prediction outputs (ignored in git).

## Usage
These scripts are wrappers around the `kneeseg` library. You can run them with the configuration templates in `kneeseg/configs/`:

```bash
# Baseline 2-Pass Training & Inference
python3 experiments/train.py kneeseg/configs/ski10_full_train_config.json
python3 experiments/inference.py kneeseg/configs/ski10_full_inference_config.json
python3 experiments/inference.py kneeseg/configs/oai_full_inference_config.json

# Reproduce Final SKI10 20-Case Evaluation (FB=0.9689, TB=0.9742, FC=0.8026, TC=0.8001)
python3 experiments/evaluate_final_ski10.py

# Reproduce Final OAI 31-Case 6-Structure Evaluation (FB=0.9408, TB=0.9669, PB=0.9639, FC=0.8811, TC=0.8698, PC=0.8263)
python3 experiments/evaluate_final_oai.py

# Reproduce Unified Cross-Domain (SKI10 + OAI) Evaluation on Both Datasets
python3 experiments/evaluate_final_unified.py --dataset both
```

## Model Checkpoints (`models/`, `models_oai/`, `models_unified/`)
All model weights are self-contained inside the 4 standard checkpoints in `experiments/models*/` (published on Hugging Face under the MIT License at [wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas), [wq2012/knee_3d_mri_segmentation_OAI_downsampled_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_OAI_downsampled_subchondral_atlas), and [wq2012/knee_3d_mri_segmentation_SKI10_OAI_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_OAI_subchondral_atlas), and also backward-compatible with [wq2012/knee_3d_mri_segmentation_SKI10](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10) and [wq2012/knee_3d_mri_segmentation_OAI_downsampled](https://huggingface.co/wq2012/knee_3d_mri_segmentation_OAI_downsampled)):

1.  **`bone_rf_p1.joblib`** (Bone Pass 1, `stage='v2_p1'` / `'oai_p1'` / `'unified_p1'`):
    - **Input**: 3D MRI volume & physical voxel spacing.
    - **Output**: Pass-1 bone segmentation and multi-class probability map (`Background`, `Femur`, `Tibia`, plus `Patella` on 6-class OAI).

2.  **`bone_rf_p2.joblib`** (Bone Pass 2 + Binary Subchondral Surface Refiner, `stage='v2_p2'` / `'oai_p2'` / `'unified_p2'`):
    - **Input**: 3D MRI volume, Pass-1 bone probability map, and voxel spacing.
    - **Output**: Refined bone masks (`RefBone V2` binary boundary-shell refiner) and continuous Pass-2 bone probability map in `[0, 1]`.

3.  **`cartilage_rf_p1.joblib`** (Cartilage Pass 1, `stage='v6_p1'` / `'oai_v6_p1'` / `'unified_v6_p1'`):
    - **Input**: 3D MRI volume, Pass-2 bone masks, and voxel spacing.
    - **Output**: Pass-1 4-class cartilage probability maps (`predict_proba_map`: `Background`, `Subchondral Bone/Meniscus`, `Target Cartilage`, `Opposing Cartilage`) and Pass-1 segmentation (`predict`).

4.  **`cartilage_rf_p2.joblib`** (Cartilage Pass 2 + Stage 3 + Subchondral Calibrator, `stage='v6_p2'` / `'oai_v6_p2'` / `'unified_v6_p2'`):
    - **Input**: 3D MRI volume, Pass-2 bone masks, Pass-1 cartilage probability map, and voxel spacing.
    - **Output**: Final calibrated femoral, tibial, and (for OAI) patellar cartilage segmentation.
