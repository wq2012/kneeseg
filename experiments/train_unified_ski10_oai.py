"""Train and assemble the Unified SKI10 (80 train) + OAI (128 train) 208-Scan Subchondral Atlas Model."""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import joblib
from joblib import Parallel, delayed
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import HistGradientBoostingClassifier

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_EXP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from kneeseg.features import fast_signed_distance_transform
from kneeseg.io import load_volume
from kneeseg.oai_subchondral_atlas import (
    coords_to_bone_atlas_bins_3bone,
    detect_medial_is_low_z_3bone,
    detect_oai_layout,
    extract_ray_feats_3bone_v2,
    get_robust_bone_frame_3bone,
    to_canonical_oai,
)
from experiments.train_oai_cartilage_v6 import (
    compute_case_atlas_hist_oai_v6,
    make_atlas_priors_oai_v6,
)


def main():
    t0 = time.time()
    ski10_model_dir = os.path.join(_EXP_DIR, 'models')
    oai_model_dir = os.path.join(_EXP_DIR, 'models_oai')
    unified_model_dir = os.path.join(_EXP_DIR, 'models_unified')
    os.makedirs(unified_model_dir, exist_ok=True)

    with open(os.path.join(_ROOT_DIR, 'kneeseg', 'data', 'ski10_full_split.json')) as f:
        ski10_split = json.load(f)
    with open(os.path.join(_ROOT_DIR, 'kneeseg', 'data', 'oai_split.json')) as f:
        oai_split = json.load(f)

    ski10_data_dir = os.environ.get('SKI10_DATA_DIR', '/path/to/SKI10/TrainingData')
    ski10_lbl_dir = os.environ.get('SKI10_LABEL_DIR', ski10_data_dir + '_labels')
    oai_data_dir = os.environ.get('OAI_DATA_DIR', '/path/to/OAI_refactored/Data')
    oai_lbl_dir = os.environ.get('OAI_LABEL_DIR', oai_data_dir + '_labels')

    print("Loading SKI10 (80 train) and OAI (128 train) bone masks for 208-scan Joint Atlas & Cross-Domain Training...", flush=True)
    ski10_cache_path = os.path.join(_EXP_DIR, 'cache', 'train80_ref_bones_v2.joblib')
    ski10_tr_bones = joblib.load(ski10_cache_path) if os.path.exists(ski10_cache_path) else None
    oai_tr_bones = joblib.load(os.path.join(_EXP_DIR, 'cache_oai', 'train128_ref_bones_v2.joblib'))

    def load_ski10_tr(fname):
        img, sp = load_volume(os.path.join(ski10_data_dir, fname), return_spacing=True)
        lbl = load_volume(os.path.join(ski10_lbl_dir, fname.replace('image-', 'labels-')), return_spacing=False).astype(np.uint8)
        lbl_c = lbl.copy()
        lbl_c[lbl_c == 5] = 1
        lbl_c[lbl_c == 6] = 3
        if ski10_tr_bones is not None and fname in ski10_tr_bones:
            bm = ski10_tr_bones[fname]
        else:
            sig_j = [1.5 / sp[0], 0.8 / sp[1], 0.8 / sp[2]]
            bm = {
                'femur': (gaussian_filter((lbl_c == 1).astype(np.float32), sigma=sig_j) >= 0.48).astype(np.uint8),
                'tibia': (gaussian_filter((lbl_c == 3).astype(np.float32), sigma=sig_j) >= 0.48).astype(np.uint8),
            }
        return {'domain': 'ski10', 'fname': fname, 'img': img.astype(np.float32), 'lbl': lbl_c, 'sp': sp, 'bm': bm}

    def load_oai_tr(fname):
        img, _ = load_volume(os.path.join(oai_data_dir, fname), return_spacing=True)
        lbl = load_volume(os.path.join(oai_lbl_dir, fname.replace('image-', 'labels-')), return_spacing=False).astype(np.uint8)
        layout = detect_oai_layout(img)
        img_c = to_canonical_oai(img, layout=layout).astype(np.float32)
        lbl_c = to_canonical_oai(lbl, layout=layout).astype(np.uint8)
        bm = oai_tr_bones[fname]
        return {'domain': 'oai', 'fname': fname, 'img': img_c, 'lbl': lbl_c, 'sp': (1.0, 1.0, 1.0), 'bm': bm}

    ski10_tr = Parallel(n_jobs=16, prefer='threads')(delayed(load_ski10_tr)(fn) for fn in ski10_split['train'])
    oai_tr = Parallel(n_jobs=16, prefer='threads')(delayed(load_oai_tr)(fn) for fn in oai_split['train'])
    joint_208 = ski10_tr + oai_tr
    print(f"Loaded {len(ski10_tr)} SKI10 train + {len(oai_tr)} OAI train = {len(joint_208)} total scans in {time.time() - t0:.1f}s", flush=True)

    # 1. Build 208-Scan Joint 3D Subchondral Bone Atlas (Femur & Tibia) + Cross-Domain HGB Refiner
    t_b = time.time()
    joint_bone_atlases = {}
    joint_bone_models = {}
    for bone_name, b_lbl, max_other in [('femur', 1, 45.0), ('tibia', 3, 25.0)]:
        def hist_bone_208(c):
            bm = c['bm']
            sp = c['sp']
            lbl = c['lbl']
            df = fast_signed_distance_transform(bm['femur'] > 0, sp)
            dt = fast_signed_distance_transform(bm['tibia'] > 0, sp)
            d_self = df if bone_name == 'femur' else dt
            d_other = dt if bone_name == 'femur' else df
            med_low_z = detect_medial_is_low_z_3bone(bm, sp)
            q50, scale, _ = get_robust_bone_frame_3bone(bm[bone_name], d_other, max_other, sp, med_low_z)
            shell = np.abs(d_self) <= 4.0
            inds = np.argwhere(shell)
            coords_mm = inds.astype(np.float32) * np.array(sp, dtype=np.float32)
            bins_s, bins_m = coords_to_bone_atlas_bins_3bone(coords_mm, q50, scale, med_low_z)
            yb = (lbl[shell] == b_lbl)
            w_vox = float(np.prod(sp))
            ct_s = np.zeros((24, 32, 24), dtype=np.float32)
            cp_s = np.zeros((24, 32, 24), dtype=np.float32)
            ct_m = np.zeros((28, 32, 24), dtype=np.float32)
            cp_m = np.zeros((28, 32, 24), dtype=np.float32)
            np.add.at(ct_s, bins_s, w_vox)
            np.add.at(cp_s, tuple(b[yb] for b in bins_s), w_vox)
            np.add.at(ct_m, bins_m, w_vox)
            np.add.at(cp_m, tuple(b[yb] for b in bins_m), w_vox)
            return ct_s, cp_s, ct_m, cp_m

        h_list = Parallel(n_jobs=16, prefer='threads')(delayed(hist_bone_208)(c) for c in joint_208)
        p_joint_full = (
            (gaussian_filter(sum(x[1] for x in h_list), 0.8) / (gaussian_filter(sum(x[0] for x in h_list), 0.8) + 2.5)).astype(np.float32),
            (gaussian_filter(sum(x[3] for x in h_list), 0.8) / (gaussian_filter(sum(x[2] for x in h_list), 0.8) + 2.5)).astype(np.float32),
        )
        joint_bone_atlases[bone_name] = p_joint_full

        def sample_joint_bone_208(idx):
            c = joint_208[idx]
            bm = c['bm']
            sp = c['sp']
            lbl = c['lbl']
            shell, d_self, d_other, X1, _ = extract_ray_feats_3bone_v2(c['img'], sp, bm, bone_name, p_joint_full)
            yb = (lbl[shell] == b_lbl).astype(np.uint8)
            d_o_sh = d_other[shell]
            art_idx = np.where(d_o_sh <= max_other)[0]
            non_idx = np.where(d_o_sh > max_other)[0]
            rng = np.random.RandomState(900 + idx)
            sel_a = rng.choice(art_idx, min(len(art_idx), 1600), replace=False) if len(art_idx) > 0 else art_idx
            sel_n = rng.choice(non_idx, min(len(non_idx), 900), replace=False) if len(non_idx) > 0 else non_idx
            sel = np.concatenate([sel_a, sel_n])
            return X1[sel], yb[sel]

        s_jb = Parallel(n_jobs=16, prefer='threads')(delayed(sample_joint_bone_208)(i) for i in range(len(joint_208)))
        X_jb = np.vstack([s[0] for s in s_jb])
        y_jb = np.concatenate([s[1] for s in s_jb])
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=16, user_api='openmp'):
            clf_jb = HistGradientBoostingClassifier(
                max_iter=200, max_leaf_nodes=63, min_samples_leaf=30,
                learning_rate=0.05, max_features=0.70, l2_regularization=1.5,
                early_stopping=False, random_state=777,
            ).fit(X_jb, y_jb)
        joint_bone_models[bone_name] = clf_jb
        print(f"  [Joint-208 Bone {bone_name.upper()}] Trained cross-domain refiner on {X_jb.shape} in {time.time() - t_b:.1f}s", flush=True)

    # 2. Build 208-Scan Joint 4D Articular Coordinate Cartilage Atlas across 80 SKI10 + 128 OAI
    t_c = time.time()
    per_case_h_208 = Parallel(n_jobs=16, prefer='threads')(
        delayed(compute_case_atlas_hist_oai_v6)(c['img'], c['bm'], c['sp'], c['lbl'])
        for c in joint_208
    )
    n_h = len(per_case_h_208[0])
    vox_weights = [float(np.prod(c['sp'])) for c in joint_208]
    is_oai_flags = [c['domain'] == 'oai' for c in joint_208]
    w_all_sum = sum(vox_weights)
    w_oai_sum = max(sum(w for w, is_o in zip(vox_weights, is_oai_flags) if is_o), 1e-6)
    oai_scale = float(w_all_sum / w_oai_sum)
    tot_h_208 = []
    for k in range(n_h):
        if k >= 37:
            tot_h_208.append(sum(h[k] * w for h, w, is_o in zip(per_case_h_208, vox_weights, is_oai_flags) if is_o))
        elif k in (25, 29, 33):
            tot_h_208.append(sum(h[k] * w * oai_scale for h, w, is_o in zip(per_case_h_208, vox_weights, is_oai_flags) if is_o))
        else:
            tot_h_208.append(sum(h[k] * w for h, w in zip(per_case_h_208, vox_weights)))
    ap_joint_208 = make_atlas_priors_oai_v6(tot_h_208)
    print(f"Built 208-Scan Joint Cross-Domain 4D Articular Cartilage Atlas in {time.time() - t_c:.1f}s", flush=True)

    # 3. Assemble Unified Model Checkpoints in experiments/models_unified/
    ski10_b1 = joblib.load(os.path.join(ski10_model_dir, 'bone_rf_p1.joblib'))
    ski10_b2 = joblib.load(os.path.join(ski10_model_dir, 'bone_rf_p2.joblib'))
    ski10_c1 = joblib.load(os.path.join(ski10_model_dir, 'cartilage_rf_p1.joblib'))
    ski10_c2 = joblib.load(os.path.join(ski10_model_dir, 'cartilage_rf_p2.joblib'))

    oai_b1 = joblib.load(os.path.join(oai_model_dir, 'bone_rf_p1.joblib'))
    oai_b2 = joblib.load(os.path.join(oai_model_dir, 'bone_rf_p2.joblib'))
    oai_c1 = joblib.load(os.path.join(oai_model_dir, 'cartilage_rf_p1.joblib'))
    oai_c2 = joblib.load(os.path.join(oai_model_dir, 'cartilage_rf_p2.joblib'))

    joblib.dump({
        'format_version': 'unified_ski10_oai_bone_p1',
        'n_train_scans': 208,
        'n_ski10_train': 80,
        'n_oai_train': 128,
        'ski10_p1': ski10_b1,
        'oai_p1': oai_b1,
        'joint_bone_atlases_208': joint_bone_atlases,
    }, os.path.join(unified_model_dir, 'bone_rf_p1.joblib'), compress=3)

    joblib.dump({
        'format_version': 'unified_ski10_oai_bone_p2',
        'n_train_scans': 208,
        'n_ski10_train': 80,
        'n_oai_train': 128,
        'ski10_p2': ski10_b2,
        'oai_p2': oai_b2,
        'oai_p1': oai_b1,
        'joint_bone_atlases_208': joint_bone_atlases,
        'joint_bone_models_208': joint_bone_models,
    }, os.path.join(unified_model_dir, 'bone_rf_p2.joblib'), compress=3)

    joblib.dump({
        'format_version': 'unified_ski10_oai_cart_p1',
        'n_train_scans': 208,
        'n_ski10_train': 80,
        'n_oai_train': 128,
        'ski10_p1': ski10_c1,
        'oai_p1': oai_c1,
        'joint_atlas_priors_208': ap_joint_208,
    }, os.path.join(unified_model_dir, 'cartilage_rf_p1.joblib'), compress=3)

    joblib.dump({
        'format_version': 'unified_ski10_oai_cart_p2',
        'n_train_scans': 208,
        'n_ski10_train': 80,
        'n_oai_train': 128,
        'ski10_p2': ski10_c2,
        'oai_p2': oai_c2,
        'joint_atlas_priors_208': ap_joint_208,
    }, os.path.join(unified_model_dir, 'cartilage_rf_p2.joblib'), compress=3)

    print(f"Saved Unified 208-Scan SKI10+OAI Models to {unified_model_dir} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
