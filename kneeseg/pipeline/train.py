
import argparse
import json
import os
import numpy as np
import joblib
from tqdm import tqdm
from kneeseg.io import load_volume
from kneeseg.bone_rf import BoneClassifier
from kneeseg.rf_seg import CartilageClassifier
from scipy.ndimage import distance_transform_edt
from kneeseg.validate_config import validate_config

def train_improved(config):
    validate_config(config)
    data_cfg = config['data_config']
    out_cfg = config['output_config']
    model_cfg = config.get('model_config', {}) # Ensure model_cfg exists
    
    target_bones = model_cfg.get('target_bones', ['femur', 'tibia']) # Default to SKI10 legacy
    n_jobs = model_cfg.get('n_jobs', -1) # Default to all cores if not specified
    target_dtype = model_cfg.get('dtype', 'float32')
    print(f"Target Bones: {target_bones}, n_jobs: {n_jobs}, dtype: {target_dtype}")

    image_dir = data_cfg['image_directory']
    label_dir = data_cfg['label_directory']
    
    # Handle model_directory location (strictly from model_config)
    model_cfg = config.get('model_config', {})
    model_dir = model_cfg.get('model_directory', 'experiments2/models') # Default fallback
    
    if not model_dir:
         raise ValueError("model_directory must be specified in model_config")
        
    os.makedirs(model_dir, exist_ok=True)
    
    # Load Files
    with open(data_cfg['split_file'], 'r') as f:
        split = json.load(f)
    train_files = split['train'] # Full training set
    
    print(f"Training on {len(train_files)} files...")
    
    images = []
    labels = []
    
    for f in tqdm(train_files):
        img = load_volume(os.path.join(image_dir, f))
        lbl = load_volume(os.path.join(label_dir, f.replace('image', 'labels')))
        images.append(img)
        labels.append(lbl)

    # --- STAGE 1: BONE SEGMENTATION (Pass 1) ---
    print("\n--- Training Bone RF (Pass 1) ---")
    bone_params = config['training_config']['bone_parameters']
    # Use config params or defaults. Prioritize specific bone_params['n_jobs'] if present, else global n_jobs
    bp_n_estimators = bone_params.get('n_estimators', 100)
    bp_max_depth = bone_params.get('max_depth', 25)
    bp_n_jobs = bone_params.get('n_jobs', n_jobs)
    
    print(f"    Params: n_estimators={bp_n_estimators}, max_depth={bp_max_depth}, n_jobs={bp_n_jobs}")
    bone_rf_p1 = BoneClassifier(n_estimators=bp_n_estimators, max_depth=bp_max_depth, n_jobs=bp_n_jobs)
    bone_rf_p1.train(images, labels, dtype=target_dtype)
    bone_rf_p1.save(os.path.join(model_dir, 'bone_rf_p1.joblib'))
    
    # Generate Prob Maps
    print("    Generating Bone Prob Maps for Pass 2...")
    prob_maps = []
    
    for img in tqdm(images):
        _, prob = bone_rf_p1.predict(img, dtype=target_dtype)
        prob_maps.append(prob)
        
    # --- STAGE 2: BONE SEGMENTATION (Pass 2 - Auto-Context) ---
    print("\n--- Training Bone RF (Pass 2 - Auto-Context) ---")
    bone_rf_p2 = BoneClassifier(n_estimators=bp_n_estimators, max_depth=bp_max_depth, n_jobs=bp_n_jobs)
    bone_rf_p2.train(images, labels, prob_maps=prob_maps, dtype=target_dtype)
    bone_rf_p2.save(os.path.join(model_dir, 'bone_rf_p2.joblib'))
    
    # Generate Final Bone Masks for Cartilage Training
    print("    Generating Final Bone Masks (from Pass 2)...")
    bone_masks = []
    
    for img, pm in tqdm(zip(images, prob_maps)):
        # Pass 2 predict
        _, prob2 = bone_rf_p2.predict(img, prob_map=pm, dtype=target_dtype)
        pred = np.argmax(prob2, axis=-1)
        
        mask_dict = {}
        # Map: femur->1, tibia->2, patella->3 (Based on BoneClassifier mapping)
        bone_label_map = {'femur': 1, 'tibia': 2, 'patella': 3}
        
        for bone in target_bones:
            lbl_idx = bone_label_map.get(bone)
            if lbl_idx:
                mask_dict[bone] = (pred == lbl_idx).astype(np.uint8)
            else:
                # Handle unknown bone if necessary, or warn
                pass
        
        bone_masks.append(mask_dict)
    
    # --- STAGE 3: CARTILAGE SEGMENTATION (Pass 1) ---
    print("\n--- Training Cartilage RF (Pass 1) ---")
    # Now we train cartilage using FEATURES derived from PREDICTED BONES (bone_masks)
    cart_params = config['training_config']['cartilage_parameters']
    cp_n_estimators = cart_params.get('n_estimators', 100)
    cp_max_depth = cart_params.get('max_depth', 20)
    cp_n_jobs = cart_params.get('n_jobs', n_jobs)
    cp_prox = cart_params.get('training_proximity_mm', 10.0)
    
    print(f"    Params: n_estimators={cp_n_estimators}, max_depth={cp_max_depth}, n_jobs={cp_n_jobs}, prox={cp_prox}")
    
    cart_rf_p1 = CartilageClassifier(n_estimators=cp_n_estimators, max_depth=cp_max_depth, target_bones=target_bones, n_jobs=cp_n_jobs)
    cart_rf_p1.train(images, bone_masks, labels, landmarks_list=None, dtype=target_dtype) # No ASM landmarks used
    cart_rf_p1.save(os.path.join(model_dir, 'cartilage_rf_p1.joblib'))
    
    # Generate Prob Maps (P1)
    print("    Generating Cartilage Prob Maps for Pass 2...")
    cart_prob_maps = []
    
    for img, b_mask in tqdm(zip(images, bone_masks)):
        pm = cart_rf_p1.predict_proba_map(img, b_mask, proximity_mm=cp_prox, dtype=target_dtype)
        cart_prob_maps.append(pm)
        
    # --- STAGE 4: CARTILAGE SEGMENTATION (Pass 2 - Auto-Context) ---
    print("\n--- Training Cartilage RF (Pass 2 - Auto-Context) ---")
    cart_rf_p2 = CartilageClassifier(n_estimators=cp_n_estimators, max_depth=cp_max_depth, target_bones=target_bones, n_jobs=cp_n_jobs)
    cart_rf_p2.train(images, bone_masks, labels, prob_maps=cart_prob_maps, dtype=target_dtype)
    cart_rf_p2.save(os.path.join(model_dir, 'cartilage_rf_p2.joblib'))
    
    print("Training Complete.")

if __name__ == "__main__":
    with open('experiments/ski10_full_train_config.json', 'r') as f:
        config = json.load(f)
    train_improved(config)
