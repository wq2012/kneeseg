
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

def train_improved(config):
    data_cfg = config['data_config']
    out_cfg = config['output_config']

    image_dir = data_cfg['image_directory']
    label_dir = data_cfg['label_directory']
    
    # Handle model_directory location (check model_config then output_config)
    model_cfg = config.get('model_config', {})
    model_dir = model_cfg.get('model_directory')
    if not model_dir:
        model_dir = out_cfg.get('model_directory', 'experiments2/models')
        
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
    bone_rf_p1 = BoneClassifier(n_estimators=100, max_depth=25)
    bone_rf_p1.train(images, labels)
    bone_rf_p1.save(os.path.join(model_dir, 'bone_rf_p1.joblib'))
    
    # Generate Prob Maps
    print("    Generating Bone Prob Maps for Pass 2...")
    prob_maps = []
    
    for img in tqdm(images):
        _, prob = bone_rf_p1.predict(img)
        prob_maps.append(prob)
        
    # --- STAGE 2: BONE SEGMENTATION (Pass 2 - Auto-Context) ---
    print("\n--- Training Bone RF (Pass 2 - Auto-Context) ---")
    bone_rf_p2 = BoneClassifier(n_estimators=100, max_depth=25)
    bone_rf_p2.train(images, labels, prob_maps=prob_maps)
    bone_rf_p2.save(os.path.join(model_dir, 'bone_rf_p2.joblib'))
    
    # Generate Final Bone Masks for Cartilage Training
    print("    Generating Final Bone Masks (from Pass 2)...")
    bone_masks = []
    
    for img, pm in tqdm(zip(images, prob_maps)):
        # Pass 2 predict
        _, prob2 = bone_rf_p2.predict(img, prob_map=pm)
        pred = np.argmax(prob2, axis=-1)
        
        mask_dict = {
            'femur': (pred == 1).astype(np.uint8),
            'tibia': (pred == 2).astype(np.uint8)
        }
        bone_masks.append(mask_dict)
    
    # --- STAGE 3: CARTILAGE SEGMENTATION (Pass 1) ---
    print("\n--- Training Cartilage RF (Pass 1) ---")
    # Now we train cartilage using FEATURES derived from PREDICTED BONES (bone_masks)
    
    cart_rf_p1 = CartilageClassifier(n_estimators=100, max_depth=20)
    cart_rf_p1.train(images, bone_masks, labels, landmarks_list=None) # No ASM landmarks used
    cart_rf_p1.save(os.path.join(model_dir, 'cartilage_rf_p1.joblib'))
    
    # Generate Prob Maps (P1)
    print("    Generating Cartilage Prob Maps for Pass 2...")
    cart_prob_maps = []
    
    for img, b_mask in tqdm(zip(images, bone_masks)):
        pm = cart_rf_p1.predict_proba_map(img, b_mask, proximity_mm=10.0)
        cart_prob_maps.append(pm)
        
    # --- STAGE 4: CARTILAGE SEGMENTATION (Pass 2 - Auto-Context) ---
    print("\n--- Training Cartilage RF (Pass 2 - Auto-Context) ---")
    cart_rf_p2 = CartilageClassifier(n_estimators=100, max_depth=20)
    cart_rf_p2.train(images, bone_masks, labels, prob_maps=cart_prob_maps)
    cart_rf_p2.save(os.path.join(model_dir, 'cartilage_rf_p2.joblib'))
    
    print("Training Complete.")

if __name__ == "__main__":
    with open('experiments/ski10_full_train_config.json', 'r') as f:
        config = json.load(f)
    train_improved(config)
