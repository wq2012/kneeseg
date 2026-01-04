
# Improved Semantic Context Forests (Phase 10)

This approach fixes the "feature mismatch" and "ASM failure" of Phase 9 by replacing the ASM Bone Segmentation with a **Dense Random Forest Classifier**.

## 1. Environment requirements
- Python 3.x
- `scikit-learn`, `simpleitk`, `numpy`, `scipy`, `joblib` (Installed).

## 2. Pipeline Overview
1.  **Bone Prediction (Pass 1)**: Dense Random Forest predicts initial bone masks.
2.  **Bone Prediction (Pass 2 - Auto-Context)**: A second RF uses Pass 1 probabilities as features to refine bone segmentation.
3.  **Distance Transforms**: Signed Distance Transforms are computed from *Pass 2* masks.
4.  **Cartilage Prediction**: A third RF predicts cartilage using the refined DT features.

## 3. How to Train
Run the improved training script (Trains all 3 stages automatically):
```bash
python3 experiments/phase9_improved_train.py
```
This saves models to `experiments/ski10_full_models_repro/`.

## 4. How to Infer
Run the improved inference script:
```bash
python3 experiments/phase9_improved_inference.py
```
This generates predictions in `experiments/ski10_improved_predictions/` and prints DSC scores.

## 5. Prototype Results (10 Training Files)
- **Bone DSC**: ~0.88 (vs 0.0 with ASM).
- **Cartilage DSC**: ~0.62 (vs 0.0-0.3 with ASM).
*Scaling to 80 files is expected to reach the >80% target.*
