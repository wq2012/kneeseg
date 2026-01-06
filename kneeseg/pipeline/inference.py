
import json
import os
import numpy as np
import joblib
from tqdm import tqdm
from kneeseg.io import load_volume, save_volume
from kneeseg.bone_rf import BoneClassifier
from kneeseg.rf_seg import CartilageClassifier
from kneeseg.features import extract_features, compute_signed_distance_transforms
from kneeseg.validate_config import validate_config

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
    
    validate_config(config)

    data_cfg = config['data_config']
    model_cfg = config['model_config']
    target_bones = model_cfg.get('target_bones', ['femur', 'tibia'])
    out_cfg = config['output_config']
    
    image_dir = data_cfg['image_directory']
    label_dir = data_cfg.get('label_directory') # Optional
    split_file = data_cfg.get('split_file') # Optional
    select_files = data_cfg.get('select_files') # Optional

    model_dir = model_cfg.get('model_directory')
    if not model_dir:
         # Fallback default
         model_dir = 'experiments2/models'
         
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
        
        # Re-init with target_bones
        cart_rf_p1 = CartilageClassifier(target_bones=target_bones)
        cart_rf_p1.load(p1_path)
        
        # Load P2
        p2_path = os.path.join(model_dir, 'cartilage_rf_p2.joblib')
        if os.path.exists(p2_path):
            cart_rf_p2 = CartilageClassifier(target_bones=target_bones)
            cart_rf_p2.load(p2_path)
        else:
            print("Warning: Cartilage RF P2 model not found.")
            
    except FileNotFoundError as e:
        print(f"Warning: Cartilage RF models not found. Skipping cartilage prediction. Details: {e}")
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
        
        bone_masks = {}
        bone_label_map = {'femur': 1, 'tibia': 2, 'patella': 3}
        for bone in target_bones:
            lbl_idx = bone_label_map.get(bone)
            if lbl_idx:
                bone_masks[bone] = (bone_pred == lbl_idx).astype(np.uint8)
        
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
             s = {}
             # Define Structure Mapping: Name -> (Target Label, GT Label)
             # GT Label: SKI10=1,2,3,4. OAI=1,2,3,4,5,6.
             # Target Label (Prediction):
             # Bone: 1(Fem), 2(Tib), 3(Pat)
             # Cart: 2(Fem), 4(Tib), 6(Pat) [Note: CartilageClassifier predicts mapped labels directly if trained that way?
             # No, CartilageRF predicts labels as they appeared in training y.
             # In train.py, labels were passed directly.
             # OAI refactoring mapped PatCart to 6.
             # SKI10 has 2, 4.
             # So cartilage RF predicts 2, 4, 6.
             
             # Metric Check List
             structure_map = {
                 'Femur': (1, 1), # (BonePred Val, GT Val)
                 'Tibia': (2, 3), # BonePred=2 -> GT=3 (Tibia)
                 'Patella': (3, 5), # BonePred=3 -> GT=5
                 'Femoral Cartilage': (2, 2), # CartPred=2 -> GT=2
                 'Tibial Cartilage': (4, 4), # CartPred=4 -> GT=4
                 'Patellar Cartilage': (6, 6) # CartPred=6 -> GT=6
             }
             
             # Filter structures based on target_bones
             # Always evaluate bones if in target_bones
             # Always evaluate cartilage if matching bone in target_bones
             
             start_eval_keys = []
             if 'femur' in target_bones: start_eval_keys.extend(['Femur', 'Femoral Cartilage'])
             if 'tibia' in target_bones: start_eval_keys.extend(['Tibia', 'Tibial Cartilage'])
             if 'patella' in target_bones: start_eval_keys.extend(['Patella', 'Patellar Cartilage'])
             
             for name in start_eval_keys:
                 idx_pred, idx_gt = structure_map[name]
                 
                 if 'Cartilage' in name:
                     curr_pred = (cart_pred == idx_pred).astype(np.uint8)
                 else:
                     curr_pred = (bone_pred == idx_pred).astype(np.uint8)
                 
                 curr_gt = (lbl == idx_gt).astype(np.uint8)
                 s[name] = calculate_dice(curr_pred, curr_gt)
                 
             scores[f] = s
             print(f"  {f}: {s}")
             
        # Save Prediction
        final_seg = np.zeros_like(img, dtype=np.uint8)
        
        # Add Bones (Map Pred 1->1, 2->3, 3->5)
        # Bone 1 (Femur) -> 1
        final_seg[bone_pred == 1] = 1
        # Bone 2 (Tibia) -> 3
        final_seg[bone_pred == 2] = 3
        # Bone 3 (Patella) -> 5
        final_seg[bone_pred == 3] = 5
        
        # Add Cartilage (Overwrite)
        if cart_rf_p1 is not None:
            # Cart Labels are likely correct (2, 4, 6)
            # Use unique predicted labels
            u_labels = np.unique(cart_pred)
            for l in u_labels:
                if l == 0: continue
                final_seg[cart_pred == l] = l
            
        save_volume(final_seg, os.path.join(output_dir, f.replace('image', 'labels')), metadata={'spacing': spacing})
        
    # Summary
    print("\n--- Improved Results ---")
    summary = {}
    
    # Collect all keys seen
    all_keys = set()
    for s in scores.values():
        all_keys.update(s.keys())
        
    for k in sorted(all_keys):
        vals = [v[k] for v in scores.values() if k in v]
        if vals:
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
