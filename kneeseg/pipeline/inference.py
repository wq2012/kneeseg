
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
    label_dir = data_cfg.get('label_directory') # Optional
    split_file = data_cfg.get('split_file') # Optional
    select_files = data_cfg.get('select_files') # Optional

    model_dir = model_cfg['model_directory']
    output_dir = out_cfg['prediction_directory']
    os.makedirs(output_dir, exist_ok=True)
    
    test_files = []
    if select_files:
        test_files = select_files
    elif split_file:
        with open(split_file, 'r') as f:
            split = json.load(f)
        test_files = split.get('eval', [])
    else:
        # Fallback: All files in image directory (careful)
        # Assuming user knows what they are doing if providing simple dir
        files = [f for f in os.listdir(image_dir) if f.endswith('.mhd')]
        test_files = sorted(files)
        
    print(f"Running on {len(test_files)} files...")
    
    print(f"Loading Models from {model_dir}...")
    try:
        bone_rf_p1 = BoneClassifier()
        bone_rf_p1.load(os.path.join(model_dir, 'bone_rf_p1.joblib'))
        
        bone_rf_p2 = BoneClassifier()
        bone_rf_p2.load(os.path.join(model_dir, 'bone_rf_p2.joblib'))
    except FileNotFoundError:
        print("Error: Bone RF models not found.")
        return

    cart_rf_p1 = None
    cart_rf_p2 = None
    try:
        cart_rf_p1 = CartilageClassifier()
        # Try finding p1 or fallback to old name
        p1_path = os.path.join(model_dir, 'cartilage_rf_p1.joblib')
        if not os.path.exists(p1_path):
             p1_path = os.path.join(model_dir, 'cartilage_rf.joblib')
        
        cart_rf_p1.load(p1_path)
        
        # Load P2
        p2_path = os.path.join(model_dir, 'cartilage_rf_p2.joblib')
        if os.path.exists(p2_path):
            cart_rf_p2 = CartilageClassifier()
            cart_rf_p2.load(p2_path)
        else:
            print("Warning: Cartilage RF P2 model not found.")
            
    except FileNotFoundError:
        print("Warning: Cartilage RF models not found. Skipping cartilage prediction.")
        cart_rf_p1 = None
        cart_rf_p2 = None

    scores = {}
    
    print("Running Inference...")
    for f in tqdm(test_files):
        img_path = os.path.join(image_dir, f)
        
        lbl = None
        if label_dir:
             lbl_path = os.path.join(label_dir, f.replace('image', 'labels'))
             if os.path.exists(lbl_path):
                 lbl = load_volume(lbl_path)

        img, spacing = load_volume(img_path, return_spacing=True)
        
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
        if cart_rf_p1 is not None and cart_rf_p2 is not None:
             # 2. Predict Cartilage
             # Pass 1: Prob Map
             c_prob1 = cart_rf_p1.predict_proba_map(img, bone_masks, proximity_mm=20.0)
             
             # Pass 2: Final Prediction (Auto-Context)
             cart_pred, _ = cart_rf_p2.predict(img, bone_masks, proximity_mm=20.0, prob_map=c_prob1)
        elif cart_rf_p1 is not None:
             # Fallback to single pass if p2 missing (compatibility)
             print("Warning: Only P1 model found. Running single pass.")
             cart_pred, _ = cart_rf_p1.predict(img, bone_masks, proximity_mm=20.0)
        
        # 3. Evaluate
        if lbl is not None:
             pred_fem = (bone_pred == 1).astype(np.uint8)
             pred_tib = (bone_pred == 2).astype(np.uint8)
             pred_fc = (cart_pred == 2).astype(np.uint8)
             pred_tc = (cart_pred == 4).astype(np.uint8)
             
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
             
        # Save Prediction
        # Reconstruct label map: 0=Bg, 1=Fem, 2=FemCart, 3=Tib, 4=TibCart
        # Bone Pred: 1, 2. Cart Pred: 1, 2, 3, 4 (raw)
        # We need to merge. 
        # Actually current CartPred logic is independent?
        # Let's check cartilage_rf.predict logic.
        # It returns `pred_map` where labels are from the training set.
        # If we assume training labels were 1,2,3,4.
        
        # We want to save the final segmentation.
        # Simple merge:
        final_seg = np.zeros_like(img, dtype=np.uint8)
        
        # Add Bones
        final_seg[bone_pred == 1] = 1
        final_seg[bone_pred == 2] = 3
        
        # Add Cartilage (Overwrite bones at boundary? or priority?)
        # Usually cartilage classifier is more specific near boundaries?
        # Or just trust cartilage classifier if it predicts cartilage?
        if cart_rf_p1 is not None:
            # Cartilage labels: 2 (FemCart), 4 (TibCart)
            final_seg[cart_pred == 2] = 2
            final_seg[cart_pred == 4] = 4
            
        save_volume(final_seg, os.path.join(output_dir, f.replace('image', 'labels')), metadata={'spacing': spacing})
        
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
