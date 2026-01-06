import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_gradient_magnitude, gaussian_filter

def compute_signed_distance_transforms(bone_masks, spacing=None):
    """
    Computes Signed Distance Transforms (SDT) for each bone mask.
    Negative inside bone, positive outside.
    """
    sdts = {}
    for name, mask in bone_masks.items():
        if np.any(mask > 0):
            d_out = distance_transform_edt(1 - mask, sampling=spacing).astype(np.float32)
            d_in = distance_transform_edt(mask, sampling=spacing).astype(np.float32)
            sdts[name] = d_out - d_in
        else:
            sdts[name] = np.full(mask.shape, 100.0, dtype=np.float32)
    return sdts

def compute_dts_from_landmarks(image_shape, landmarks_dict, spacing=None):
    """
    Computes Unsigned Distance Transforms from sparse landmarks.
    Used for consistency between Train (ASM) and Inference.
    """
    dts = {}
    for name, pts in landmarks_dict.items():
        if 'patella' in name: continue 
        
        mask_pts = np.zeros(image_shape, dtype=np.uint8)
        if len(pts) > 0:
            c = np.round(pts).astype(int)
            c[:, 0] = np.clip(c[:, 0], 0, image_shape[0]-1)
            c[:, 1] = np.clip(c[:, 1], 0, image_shape[1]-1)
            c[:, 2] = np.clip(c[:, 2], 0, image_shape[2]-1)
            mask_pts[c[:,0], c[:,1], c[:,2]] = 1
            
            dist = distance_transform_edt(1 - mask_pts, sampling=spacing).astype(np.float32)
            dts[name] = dist
        else:
            dts[name] = np.full(image_shape, 200.0, dtype=np.float32)
    return dts

def compute_rsid_features(image, num_shifts=30, max_shift=10, seed=42, mask=None, dtype=np.float32):
    """
    Computes Random Shift Intensity Difference (RSID) features only at mask locations.
    Returns (num_masked_voxels, num_shifts) if mask provided, else full volume.
    """
    np.random.seed(seed)
    offsets = np.random.randint(-max_shift, max_shift + 1, size=(num_shifts, 3))
    
    if mask is None:
        # Fallback to full volume if no mask (used in training initialization sometimes)
        padded = np.pad(image, max_shift, mode='edge')
        z, y, x = image.shape
        base_slice = padded[max_shift:max_shift+z, max_shift:max_shift+y, max_shift:max_shift+x]
        features = []
        for i in range(num_shifts):
            dz, dy, dx = offsets[i]
            features.append(base_slice - padded[max_shift+dz : max_shift+z+dz,
                                               max_shift+dy : max_shift+y+dy,
                                               max_shift+dx : max_shift+x+dx])
        return np.stack(features, axis=-1).astype(dtype)

    # Masked computation
    if mask.ndim == 3:
        mask_indices = np.argwhere(mask)
    else:
        # If mask is 1D, we can't easily do spatial shifts without the original indices
        # But in this pipeline, mask is usually 3D ROI or 1D flat from 3D.
        # We assume 3D mask for spatial efficiency.
        raise ValueError("RSID requires a 3D mask for spatial optimization.")

    n_vox = len(mask_indices)
    rsid_features = np.zeros((n_vox, num_shifts), dtype=dtype)
    
    # Pad image once
    padded = np.pad(image, max_shift, mode='edge')
    
    # Base intensities at mask locations (offset by padding)
    base_indices = mask_indices + max_shift
    base_vals = padded[base_indices[:,0], base_indices[:,1], base_indices[:,2]]
    
    for i in range(num_shifts):
        dz, dy, dx = offsets[i]
        shifted_indices = base_indices + [dz, dy, dx]
        shifted_vals = padded[shifted_indices[:,0], shifted_indices[:,1], shifted_indices[:,2]]
        rsid_features[:, i] = base_vals - shifted_vals
        
    return rsid_features

def compute_landmark_features(image_shape, landmarks_dict, indices_dict, mask=None, spacing=None, dtype=np.float32):
    """
    Computes distance to landmarks only at mask locations.
    Operates in Physical Space (MM) if spacing provided.
    """
    if mask is not None:
        if mask.ndim == 3:
            coords = np.argwhere(mask).astype(np.float32)
        else: # 1D flattened mask indices
            # Convert 1D indices to 3D coords
            flat_idx = np.nonzero(mask)[0]
            coords = np.array(np.unravel_index(flat_idx, image_shape)).T.astype(np.float32)
    else:
        # Full volumegrid - expensive!
        z, y, x = np.mgrid[0:image_shape[0], 0:image_shape[1], 0:image_shape[2]]
        coords = np.stack([z.flatten(), y.flatten(), x.flatten()], axis=1).astype(np.float32)

    if spacing is not None:
        coords = coords * spacing

    features = []
    for bone in sorted(landmarks_dict.keys()):
        points = landmarks_dict[bone]
        indices = indices_dict.get(bone, [])
        for idx in indices:
            if idx < len(points):
                p = points[idx]
                dist = np.sqrt(np.sum((coords - p)**2, axis=1))
                features.append(dist.astype(dtype))
            else:
                features.append(np.full(len(coords), 200.0, dtype=dtype))
    return features

def compute_dt_arithmetic_features(dts, mask=None):
    # This just returns float arrays, casting happens in extract_features usually
    # But let's check its usage.
    # It returns list of arrays. extract_features casts them.
    # So we don't strictly need to update this unless we want intermediate memory savings.
    # Let's keep it as is, casting is done by caller.
    features = []
    if 'femur' in dts and 'tibia' in dts:
        f = dts['femur']
        t = dts['tibia']
        if mask is not None:
            # Handle mismatch if f is 3D and mask is Flat
            if f.ndim == 3 and mask.ndim == 1:
                f = f.flatten()
                t = t.flatten()
            elif f.ndim == 3 and mask.ndim == 3:
                # Normal 3D indexing
                pass
            f = f[mask]
            t = t[mask]
        features.append(f + t) 
        features.append(f - t) 
    return features

def extract_features(image, dts, sigma=1.0, mask=None, r_shifts=30, landmarks_dict=None, landmark_indices=None, prob_map=None, spacing=None, sorted_bones_override=None, target_dtype='float32'):
    """
    Computes Optimized feature extraction that avoids 4D intermediate arrays and respects masks.
    """
    import ml_dtypes
    
    # Resolve dtype
    if isinstance(target_dtype, str):
        if target_dtype == 'bfloat16':
            dtype = ml_dtypes.bfloat16
        else:
            dtype = np.float32
    else:
        dtype = target_dtype
        
    img_mean = image.mean()
    img_std = image.std()
    
    # Normalize and cast immediately to target dtype
    image_norm = (image.astype(np.float32) - img_mean) / (img_std + 1e-6)
    image_norm = image_norm.astype(dtype)

    def get_masked(arr):
        if mask is not None:
            if mask.ndim == arr.ndim:
                return arr[mask].astype(dtype)
            else:
                return arr.flatten()[mask].astype(dtype)
        return arr.flatten().astype(dtype)

    features = []
    
    # 1. Intensity
    features.append(get_masked(image_norm))
    
    # 2. Gaussian
    features.append(get_masked(gaussian_filter(image_norm.astype(np.float32), sigma=sigma)))
    
    # 3. Gradient
    features.append(get_masked(gaussian_gradient_magnitude(image_norm.astype(np.float32), sigma=sigma)))
    
    # 5. DTs
    if sorted_bones_override is None:
        # Legacy behavior: use whatever is in dts
        bones_to_use = sorted(dts.keys())
    else:
        # Strict mode: use specific bones in specific order
        bones_to_use = sorted_bones_override

    for bone in bones_to_use:
        if bone in dts:
            features.append(get_masked(dts[bone]))
        else:
            # Missing bone feature (e.g. Patella missing in SKI10 inference but expected by OAI model, 
            # or vice versa if we want robust feature vectors of fixed size)
            # Use large distance (e.g. 100.0) for missing bone
            # We need to append a vector of same size as others
            # get_masked returns flat array of size N_masked
            # We can use shape from an existing feature (like Intensity)
            ref_shape = features[0].shape
            features.append(np.full(ref_shape, 100.0, dtype=dtype))
        
    # 6. DT Arithmetic
    # Only compute if 'femur' and 'tibia' are present/logic applies
    dt_arith = compute_dt_arithmetic_features(dts, mask=mask)
    for f in dt_arith:
        if mask is None:
            features.append(f.flatten().astype(dtype))
        else:
            features.append(f.astype(dtype))
        
    # 7. RSID (Mask-aware)
    if mask is not None and mask.ndim == 3:
        rsid_masked = compute_rsid_features(image_norm, num_shifts=r_shifts, max_shift=10, mask=mask, dtype=dtype)
        for i in range(rsid_masked.shape[1]):
            features.append(rsid_masked[:, i])
    else:
        # Fallback to full then mask (wasteful)
        rsid_full = compute_rsid_features(image_norm, num_shifts=r_shifts, max_shift=10, dtype=dtype)
        for i in range(rsid_full.shape[-1]):
            features.append(get_masked(rsid_full[..., i]))

    # 8. Landmarks
    if landmarks_dict and landmark_indices:
        lm_features = compute_landmark_features(image.shape, landmarks_dict, landmark_indices, mask=mask, spacing=spacing, dtype=dtype)
        for f in lm_features:
            features.append(f)
            
    # 9. Probability (Auto-Context)
    if prob_map is not None:
        if prob_map.ndim == 3: channels = [prob_map]
        else: channels = [prob_map[..., c] for c in range(prob_map.shape[-1])]
        
        for p_ch in channels:
            features.append(get_masked(p_ch))
            features.append(get_masked(gaussian_filter(p_ch.astype(np.float32), sigma=sigma)))
            
            # Context RSID
            if mask is not None and mask.ndim == 3:
                rsid_p = compute_rsid_features(p_ch, num_shifts=15, max_shift=15, mask=mask, dtype=dtype)
                for i in range(rsid_p.shape[1]):
                    features.append(rsid_p[:, i])
            else:
                rsid_p_full = compute_rsid_features(p_ch, num_shifts=15, max_shift=15, dtype=dtype)
                for i in range(rsid_p_full.shape[-1]):
                    features.append(get_masked(rsid_p_full[..., i]))
    
    return np.stack(features, axis=1)
