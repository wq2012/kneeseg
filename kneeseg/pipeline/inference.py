import json
import os
from contextlib import nullcontext
import numpy as np
from tqdm import tqdm
from kneeseg.io import load_volume, save_volume
from kneeseg.bone_rf import BoneClassifier, _BONE_OOF_PROB_CACHE, _BONE_V2_SESSION_CACHE, _vol_fingerprint
from kneeseg.cartilage_v6 import _CART_P1_SESSION_CACHE
from kneeseg.oai_subchondral_atlas import _OAI_BONE_AUX_CACHE, _OAI_CART_P1_CACHE, _is_oai_domain
from kneeseg.rf_seg import CartilageClassifier
from kneeseg.validate_config import validate_config

def calculate_dice(pred, gt):
    smooth = 1e-6
    if np.sum(gt) == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    intersection = np.sum(pred * gt)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)

_POOL_STATE = {}


def _process_one_file(f):
    st = _POOL_STATE
    omp_threads = st.get('openmp_threads')
    if omp_threads:
        from threadpoolctl import threadpool_limits
        tp_ctx = threadpool_limits(limits=omp_threads, user_api='openmp')
    else:
        tp_ctx = nullcontext()

    with tp_ctx:
        image_dir = st['image_dir']
        label_dir = st['label_dir']
        output_dir = st['output_dir']
        target_bones = st['target_bones']
        target_dtype = st['target_dtype']
        bone_rf_p1 = st['bone_rf_p1']
        bone_rf_p2 = st['bone_rf_p2']
        cart_rf_p1 = st['cart_rf_p1']
        cart_rf_p2 = st['cart_rf_p2']

        img_path = os.path.join(image_dir, f)

        lbl = None
        if label_dir:
            lbl_path = os.path.join(label_dir, f.replace('image', 'labels'))
            if os.path.exists(lbl_path):
                lbl = load_volume(lbl_path)

        img, spacing = load_volume(img_path, return_spacing=True)
        fp = _vol_fingerprint(img)

        try:
            # 1. Predict Bones (Pass 1)
            _, prob1 = bone_rf_p1.predict(img, dtype=target_dtype, spacing=spacing)

            # 2. Predict Bones (Pass 2 - Auto-Context)
            bone_pred_flat, _ = bone_rf_p2.predict(img, prob_map=prob1, dtype=target_dtype, spacing=spacing)
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
                c_prob1 = cart_rf_p1.predict_proba_map(
                    img, bone_masks, proximity_mm=20.0, dtype=target_dtype, spacing=spacing
                )

                # Pass 2: Final Prediction (Auto-Context)
                cart_pred, _ = cart_rf_p2.predict(
                    img, bone_masks, proximity_mm=20.0, prob_map=c_prob1, dtype=target_dtype, spacing=spacing
                )
            elif cart_rf_p1 is not None:
                # Fallback to single pass if p2 missing (compatibility)
                print("Warning: Only P1 model found. Running single pass.")
                cart_pred, _ = cart_rf_p1.predict(
                    img, bone_masks, proximity_mm=20.0, dtype=target_dtype, spacing=spacing
                )
        finally:
            _CART_P1_SESSION_CACHE.pop(fp, None)
            _BONE_V2_SESSION_CACHE.pop(fp, None)
            _BONE_OOF_PROB_CACHE.pop(fp, None)
            _OAI_BONE_AUX_CACHE.clear()
            _OAI_CART_P1_CACHE.clear()

        # Save Prediction (Mutually Exclusive 3D Segmentation)
        final_seg = np.zeros_like(img, dtype=np.uint8)

        # Add Bones (Map Pred 1->1, 2->3, 3->5)
        final_seg[bone_pred == 1] = 1
        final_seg[bone_pred == 2] = 3
        final_seg[bone_pred == 3] = 5

        # Add Cartilage (Overwrite)
        if cart_rf_p1 is not None:
            u_labels = np.unique(cart_pred)
            for l in u_labels:
                if l == 0:
                    continue
                final_seg[cart_pred == l] = l

        save_volume(
            final_seg,
            os.path.join(output_dir, f.replace('image', 'labels')),
            metadata={'spacing': spacing},
        )

        # 3. Evaluate
        s = None
        if lbl is not None:
            s = {}
            eff_target_bones = list(target_bones)
            if (
                getattr(bone_rf_p2, 'stage', 0) != 0
                and min(img.shape) >= 32
                and not _is_oai_domain(img, spacing)
            ):
                eff_target_bones = [b for b in eff_target_bones if b != 'patella']

            start_eval_keys = []
            if 'femur' in eff_target_bones:
                start_eval_keys.extend(['Femur', 'Femoral Cartilage'])
            if 'tibia' in eff_target_bones:
                start_eval_keys.extend(['Tibia', 'Tibial Cartilage'])
            if 'patella' in eff_target_bones:
                start_eval_keys.extend(['Patella', 'Patellar Cartilage'])

            if getattr(bone_rf_p2, 'stage', 0) == 0:
                legacy_map = {
                    'Femur': (1, 1),
                    'Tibia': (2, 3),
                    'Patella': (3, 5),
                    'Femoral Cartilage': (2, 2),
                    'Tibial Cartilage': (4, 4),
                    'Patellar Cartilage': (6, 6),
                }
                for name in start_eval_keys:
                    idx_pred, idx_gt = legacy_map[name]
                    if 'Cartilage' in name:
                        curr_pred = (cart_pred == idx_pred).astype(np.uint8)
                    else:
                        curr_pred = (bone_pred == idx_pred).astype(np.uint8)
                    curr_gt = (lbl == idx_gt).astype(np.uint8)
                    s[name] = float(calculate_dice(curr_pred, curr_gt))
            else:
                lbl_eval = lbl.copy()
                if 'patella' not in eff_target_bones:
                    lbl_eval[lbl_eval == 5] = 1
                    lbl_eval[lbl_eval == 6] = 3

                structure_map = {
                    'Femur': (1, 1),
                    'Tibia': (3, 3),
                    'Patella': (5, 5),
                    'Femoral Cartilage': (2, 2),
                    'Tibial Cartilage': (4, 4),
                    'Patellar Cartilage': (6, 6),
                }

                for name in start_eval_keys:
                    idx_pred, idx_gt = structure_map[name]
                    curr_pred = (final_seg == idx_pred).astype(np.uint8)
                    curr_gt = (lbl_eval == idx_gt).astype(np.uint8)
                    s[name] = float(calculate_dice(curr_pred, curr_gt))
        return f, s


def inference_improved(config=None):
    if config is None:
         # Load default
         with open('kneeseg/configs/ski10_full_inference_config.json', 'r') as f:
             config = json.load(f)
    
    validate_config(config)

    data_cfg = config['data_config']
    model_cfg = config['model_config']
    target_bones = model_cfg.get('target_bones', ['femur', 'tibia'])
    out_cfg = config['output_config']
    
    target_dtype = model_cfg.get('dtype', 'float32')
    print(f"Target Bones: {target_bones}, dtype: {target_dtype}")
    
    image_dir = data_cfg['image_directory']
    label_dir = data_cfg.get('label_directory') # Optional
    split_file = data_cfg.get('split_file') # Optional
    select_files = data_cfg.get('select_files') # Optional

    model_dir = model_cfg.get('model_directory')
    if not model_dir:
         # Fallback default
         model_dir = 'experiments/models'
         
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

    use_fork_pool = (
        isinstance(bone_rf_p1, BoneClassifier)
        and getattr(bone_rf_p1, 'stage', None) in ('v2_p1', 'oai_p1', 'unified_p1')
        and len(test_files) >= 4
    )
    n_workers = min(8, len(test_files)) if use_fork_pool else 1
    scores = {}
    _POOL_STATE.update({
        'image_dir': image_dir,
        'label_dir': label_dir,
        'output_dir': output_dir,
        'target_bones': target_bones,
        'target_dtype': target_dtype,
        'bone_rf_p1': bone_rf_p1,
        'bone_rf_p2': bone_rf_p2,
        'cart_rf_p1': cart_rf_p1,
        'cart_rf_p2': cart_rf_p2,
        'openmp_threads': max(1, min(14, 56 // max(n_workers, 1))) if use_fork_pool else None,
    })

    print("Running Inference...")
    if use_fork_pool:
        import multiprocessing as mp

        if getattr(bone_rf_p1, 'stage', None) == 'v2_p1':
            bone_rf_p1._get_tierA()
        if getattr(bone_rf_p2, 'stage', None) == 'v2_p2':
            bone_rf_p2._get_tierA()
            bone_rf_p2._get_ref_v2()
        elif getattr(bone_rf_p2, 'stage', None) == 'unified_p2':
            ref_v2 = (getattr(bone_rf_p2, 'meta', {}) or {}).get('ski10_p2', {}).get('ref_v2', {})
            for b in ('tibia', 'femur', 'patella'):
                if b in ref_v2 and isinstance(ref_v2[b], dict):
                    for ek in ('et_v1', 'et_v2', 'et_whole', 'et_sub'):
                        et = ref_v2[b].get(ek)
                        if et is not None and (getattr(et, 'n_jobs', 8) < 0 or getattr(et, 'n_jobs', 8) > 8):
                            et.n_jobs = 8
        ctx = mp.get_context('fork')
        with ctx.Pool(n_workers) as pool:
            for f, s in tqdm(pool.imap(_process_one_file, test_files), total=len(test_files)):
                if s is not None:
                    scores[f] = s
                    print(f"  {f}: {s}", flush=True)
    else:
        for f in tqdm(test_files):
            _, s = _process_one_file(f)
            if s is not None:
                scores[f] = s
                print(f"  {f}: {s}", flush=True)
        
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
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            print(f"{k}: {mean_val:.4f} +/- {std_val:.4f}")
            summary[k] = {'mean': mean_val, 'std': std_val}
    summary['per_case'] = scores
        
    # Save Report
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Evaluation report saved to {report_path}")

if __name__ == "__main__":
    inference_improved()
