
import json
import os
import numpy as np
import joblib
from tqdm import tqdm
from kneeseg.io import load_volume, save_volume
from kneeseg.bone_rf import BoneClassifier
from kneeseg.rf_seg import CartilageClassifier
from kneeseg.features import extract_features, compute_signed_distance_transforms

def calculate_dice(pred, gt):
    smooth = 1e-6
    if np.sum(gt) == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    intersection = np.sum(pred * gt)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)

def inference_improved(config=None):
    if config is None:
         # Load default
         with open('experiments/ski10_full_inference_config.json', 'r') as f:
             config = json.load(f)

    data_cfg = config['data_config']
    model_cfg = config['model_config']
    out_cfg = config['output_config']
    
    image_dir = data_cfg['image_directory']
    label_dir = data_cfg['label_directory']
    split_file = data_cfg['split_file']
    model_dir = model_cfg['model_directory']
    output_dir = out_cfg['prediction_directory']
    os.makedirs(output_dir, exist_ok=True)
    
    with open(split_file, 'r') as f:
        split = json.load(f)
    test_files = split['eval'] # Predict on Eval set
    
    print(f"Loading Models from {model_dir}...")
    try:
        bone_rf_p1 = BoneClassifier()
        bone_rf_p1.load(os.path.join(model_dir, 'bone_rf_p1.joblib'))
        
        bone_rf_p2 = BoneClassifier()
        bone_rf_p2.load(os.path.join(model_dir, 'bone_rf_p2.joblib'))
    except FileNotFoundError:
        print("Error: Bone RF models not found.")
        return

    cart_rf = None
    try:
        cart_rf = CartilageClassifier()
        cart_rf.load(os.path.join(model_dir, 'cartilage_rf.joblib'))
    except FileNotFoundError:
        print("Warning: Cartilage RF model not found. Skipping cartilage prediction.")
        cart_rf = None

    scores = {}
    
    print("Running Inference...")
    for f in tqdm(test_files):
        img_path = os.path.join(image_dir, f)
        lbl_path = os.path.join(label_dir, f.replace('image', 'labels'))
        
        if not os.path.exists(lbl_path): continue
        
        img = load_volume(img_path)
        lbl = load_volume(lbl_path)
        
        # 1. Predict Bones (Pass 1)
        _, prob1 = bone_rf_p1.predict(img)
        
        # 2. Predict Bones (Pass 2 - Auto-Context)
        bone_pred_flat, _ = bone_rf_p2.predict(img, prob_map=prob1)
        bone_pred = bone_pred_flat.reshape(img.shape)
        
        bone_masks = {
            'femur': (bone_pred == 1).astype(np.uint8),
            'tibia': (bone_pred == 2).astype(np.uint8)
        }
        
        cart_pred = np.zeros_like(img, dtype=np.uint8)
        if cart_rf is not None:
             # 2. Predict Cartilage
             cart_pred, _ = cart_rf.predict(img, bone_masks, proximity_mm=20.0)
        
        # 3. Evaluate
        # Map Cartilage Pred (1, 2) to (1, 2) ?
        # Cartilage RF y was: 0=Bg, 1=FemCart, 2=TibCart (based on labels[mask]?)
        # Let's check training: y = lbl.flatten()[mask]. 
        # lbl has 1=Femur, 2=FemCart, 3=Tibia, 4=TibCart.
        # If mask is near bones, we might see 1,2,3,4.
        # But we want to classify cartilage.
        # Ideally, we should have trained binary or mapped labels.
        # In `rf_seg.py` training: y = lbl.flatten()[mask].
        # It takes whatever labels are in the mask.
        # If the mask covers bones, it sees bones too.
        # Result of predict is 1,2,3,4.
        
        # So we separate:
        pred_fc = (cart_pred == 2).astype(np.uint8)
        pred_tc = (cart_pred == 4).astype(np.uint8)
        
        # Also evaluate Bones
        pred_fem = (bone_pred == 1).astype(np.uint8)
        pred_tib = (bone_pred == 2).astype(np.uint8)
        
        gt_fem = (lbl == 1).astype(np.uint8)
        gt_fc = (lbl == 2).astype(np.uint8)
        gt_tib = (lbl == 3).astype(np.uint8)
        gt_tc = (lbl == 4).astype(np.uint8)
        
        s = {
            'Femur': calculate_dice(pred_fem, gt_fem),
            'Tibia': calculate_dice(pred_tib, gt_tib),
            'Femoral Cartilage': calculate_dice(pred_fc, gt_fc),
            'Tibial Cartilage': calculate_dice(pred_tc, gt_tc)
        }
        scores[f] = s
        print(f"  {f}: {s}")
        
    # Summary
    print("\n--- Improved Results ---")
    summary = {}
    for k in ['Femur', 'Tibia', 'Femoral Cartilage', 'Tibial Cartilage']:
        vals = [v[k] for v in scores.values()]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"{k}: {mean_val:.4f} +/- {std_val:.4f}")
        summary[k] = {'mean': mean_val, 'std': std_val}
        
    # Save Report
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Evaluation report saved to {report_path}")

if __name__ == "__main__":
    inference_improved()
