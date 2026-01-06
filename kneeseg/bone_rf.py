
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from .features import extract_features, compute_rsid_features
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import ml_dtypes

class BoneClassifier:
    def __init__(self, n_estimators=100, max_depth=25, n_jobs=-1):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            n_jobs=n_jobs,
            random_state=42,
            class_weight='balanced'
        )

    def extract_bone_features(self, image, prob_map=None, spacing=None, target_dtype='float32'):
        """
        Extract features suitable for Bone Segmentation.
        Focuses on larger context and spatial coordinates.
        """
        # Resolve dtype
        if isinstance(target_dtype, str):
            if target_dtype == 'bfloat16':
                dtype = ml_dtypes.bfloat16
            else:
                dtype = np.float32
        else:
            dtype = target_dtype

        features = []
        
        # Normalize
        img_mean = image.mean()
        img_std = image.std()
        img_norm = (image.astype(np.float32) - img_mean) / (img_std + 1e-6)
        img_norm = img_norm.astype(dtype)
        
        # 1. Intensity & Smooth (Multi-scale)
        features.append(img_norm.flatten())
        features.append(gaussian_filter(img_norm.astype(np.float32), sigma=2.0).astype(dtype).flatten())
        features.append(gaussian_filter(img_norm.astype(np.float32), sigma=4.0).astype(dtype).flatten())
        
        # 2. Spatial Coordinates (Normalized 0-1)
        z, y, x = np.mgrid[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]]
        features.append((z.flatten() / image.shape[0]).astype(dtype))
        features.append((y.flatten() / image.shape[1]).astype(dtype))
        features.append((x.flatten() / image.shape[2]).astype(dtype))
        
        # 3. RSID (Texture) - Sparse but helpful
        # Keep shift small-ish but larger than cartilage
        # Downsample image for RSID to save memory? No, standard RSID.
        rsid = compute_rsid_features(img_norm, num_shifts=20, max_shift=20, seed=42, dtype=dtype)
        for i in range(rsid.shape[-1]):
             features.append(rsid[..., i].flatten())
             
        # 4. Auto-Context Probabilities
        if prob_map is not None:
            # prob_map is (Z, Y, X, C)
            for c in range(prob_map.shape[-1]):
                p_ch = prob_map[..., c].astype(dtype)
                features.append(p_ch.flatten())
                features.append(gaussian_filter(p_ch.astype(np.float32), sigma=2.0).astype(dtype).flatten())
                
        return np.stack(features, axis=1)

    def train(self, images, labels, prob_maps=None, subsample=50000, dtype='float32'):
        """
        images: list of 3D arrays
        labels: list of 3D arrays (0=bg, 1=Femur, 2=FemCart, 3=Tibia, 4=TibCart)
        prob_maps: list of 3D probability maps (optional)
        """
        X_all = []
        y_all = []
        
        print(f"    Extracting features for Bone RF (ProbMap={prob_maps is not None})...")
        for i, (img, lbl) in enumerate(tqdm(zip(images, labels), total=len(images))):
            # Map Labels
            # Femur(1) + FemCart(2) -> Femur (1)
            # Tibia(3) + TibCart(4) -> Tibia (2)
            # Patella(5) + PatCart(6) -> Patella (3)
            # Background -> 0
            
            y_mapped = np.zeros_like(lbl, dtype=np.uint8)
            y_mapped[lbl == 1] = 1 # Femur
            y_mapped[lbl == 2] = 1 # FemCart -> Femur
            y_mapped[lbl == 3] = 2 # Tibia
            y_mapped[lbl == 4] = 2 # TibCart -> Tibia
            y_mapped[lbl == 5] = 3 # Patella
            y_mapped[lbl == 6] = 3 # PatCart -> Patella
            
            # Subsample
            # We need balanced sampling.
            # 1. All Bone Voxels (or large subset)
            # 2. Subset of Background (especially near bones)
            
            mask_bone = (y_mapped > 0)
            target_indices = np.argwhere(mask_bone)
            
            # Dilate bone mask to find "Near Background"
            from scipy.ndimage import binary_dilation
            mask_near = binary_dilation(mask_bone, iterations=10) & (~mask_bone)
            near_indices = np.argwhere(mask_near)
            
            # Far background
            far_indices = np.argwhere((~mask_bone) & (~mask_near))
            
            # Select samples
            n_bone = len(target_indices)
            n_near = min(len(near_indices), n_bone) # 1:1 Bone:Near
            n_far = min(len(far_indices), n_bone // 4) # fewer far samples
            
            idx_bone = np.random.choice(len(target_indices), min(n_bone, subsample//2), replace=False)
            idx_near = np.random.choice(len(near_indices), min(n_near, subsample//4), replace=False)
            idx_far = np.random.choice(len(far_indices), min(n_far, subsample//8), replace=False)
            
            coords = np.vstack([
                target_indices[idx_bone],
                near_indices[idx_near],
                far_indices[idx_far]
            ])
            
            pm = prob_maps[i] if prob_maps else None
            feats_flat = self.extract_bone_features(img, prob_map=pm, target_dtype=dtype) # (N_all, F)
            
            # Convert 3D coords to 1D indices
            flat_indices = np.ravel_multi_index(coords.T, img.shape)
            
            X_case = feats_flat[flat_indices]
            y_case = y_mapped.flatten()[flat_indices]
            
            X_all.append(X_case)
            y_all.append(y_case)
            
        print("    Fitting Bone Random Forest...")
        self.clf.fit(np.vstack(X_all), np.concatenate(y_all))

    def predict(self, image, prob_map=None, dtype='float32'):
        feats_flat = self.extract_bone_features(image, prob_map, target_dtype=dtype)
        
        # Predict in chunks to be safe? Or full.
        # 100GB RAM. Image ~10M voxels. Features ~30 floats -> 300MB * 4 = 1.2GB.
        # Easy.
        
        y_pred = self.clf.predict(feats_flat)
        y_prob = self.clf.predict_proba(feats_flat)
        
        return y_pred.reshape(image.shape), y_prob.reshape(image.shape + (-1,))

    def save(self, path):
        joblib.dump(self.clf, path)

    def load(self, path):
        self.clf = joblib.load(path)
