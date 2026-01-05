from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from .features import extract_features, compute_signed_distance_transforms

class CartilageClassifier:
    def __init__(self, n_estimators=100, max_depth=15, n_jobs=-1):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            n_jobs=n_jobs,
            random_state=42,
            class_weight='balanced'
        )

    def train(self, images, bone_masks, labels, landmarks_list=None, landmark_indices=None, prob_maps=None):
        """
        Trains the classifier on a set of images and ground truth labels.
        landmarks_list: list of dicts {'femur': (N,3), ...} matched to images
        landmark_indices: dict {'femur': [idx1...], ...} fixed indices
        prob_maps: list of probability maps from previous pass (Auto-Context)
        """
        X_all = []
        y_all = []
        
        from tqdm import tqdm
        print(f"    Extracting features for Random Forest (ProbMap={prob_maps is not None})...")
        for i, (img, b_masks, lbl) in enumerate(tqdm(zip(images, bone_masks, labels), total=len(images), desc="Case progress")):
            dts = compute_signed_distance_transforms(b_masks)
            
            # Subsample to avoid memory issues and balance classes
            min_dt = np.min([np.abs(d) for d in dts.values()], axis=0)
            mask = min_dt.flatten() < 10 # reduced to 10mm for tighter focus and memory
            
            # Extract landmarks for this case if available
            lms = landmarks_list[i] if landmarks_list else None
            pm = prob_maps[i] if prob_maps else None
            
            X = extract_features(img, dts, mask=mask, landmarks_dict=lms, landmark_indices=landmark_indices, prob_map=pm)
            y = lbl.flatten()[mask]
            
            # Additional random subsampling to speed up training
            if len(y) > 50000:
                np.random.seed(42) # For reproducibility
                indices = np.random.choice(len(y), 50000, replace=False)
                X = X[indices]
                y = y[indices]

            # Print info about data types and memory
            if i == 0:
                print(f"      Feature matrix shape: {X.shape}, dtype: {X.dtype}")

            X_all.append(X)
            y_all.append(y)
            
        print("    Fitting Random Forest (this may take a while)...")
        self.clf.fit(np.vstack(X_all), np.concatenate(y_all))

    def predict(self, image, bone_masks, proximity_mm=80.0, landmarks_dict=None, landmark_indices=None, prob_map=None):
        """
        Predicts cartilage labels for a single image.
        prob_map: (Z,Y,X,C) feature from previous pass (Auto-Context).
        """
        dts = compute_signed_distance_transforms(bone_masks)
        
        # We can still restrict prediction to voxels near bones for speed/memory
        min_dt = np.min([np.abs(d) for d in dts.values()], axis=0)
        mask = min_dt.flatten() < proximity_mm # Use configurable proximity
        
        X = extract_features(image, dts, mask=mask, landmarks_dict=landmarks_dict, 
                             landmark_indices=landmark_indices, prob_map=prob_map)
        
        # Initial prediction (only for masked voxels)
        y_pred_masked = self.clf.predict(X)
        y_prob_masked = self.clf.predict_proba(X)
        
        print(f"    [DEBUG] Masked voxels: {np.sum(mask)}, Predicted non-zero: {np.sum(y_pred_masked > 0)}")

        # Reconstruct full volumes
        y_pred = np.zeros(image.size, dtype=np.uint8)
        y_pred[mask] = y_pred_masked
        
        y_prob = np.zeros((*image.shape, self.clf.n_classes_), dtype=np.float32)
        y_prob.reshape(-1, self.clf.n_classes_)[mask] = y_prob_masked
        
        return y_pred.reshape(image.shape), y_prob

    def predict_proba_map(self, image, bone_masks, proximity_mm=80.0, landmarks_dict=None, landmark_indices=None):
        """
        Predicts dense probability map for Auto-Context.
        Returns: (Z, Y, X, C) probability array.
        """
        dts = compute_signed_distance_transforms(bone_masks)
        min_dt = np.min([np.abs(d) for d in dts.values()], axis=0)
        mask = min_dt.flatten() < proximity_mm
        
        X = extract_features(image, dts, mask=mask, landmarks_dict=landmarks_dict, landmark_indices=landmark_indices) # Pass 1 has no prob_map
        
        y_prob_masked = self.clf.predict_proba(X)
        
        y_prob = np.zeros((*image.shape, self.clf.n_classes_), dtype=np.float32)
        y_prob.reshape(-1, self.clf.n_classes_)[mask] = y_prob_masked
        
        return y_prob

    def save(self, path):
        joblib.dump(self.clf, path)

    def load(self, path):
        self.clf = joblib.load(path)
