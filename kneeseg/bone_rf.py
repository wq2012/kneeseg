import os

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

import joblib
import ml_dtypes
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    label,
    maximum_filter1d,
    minimum_filter1d,
    zoom,
)
from sklearn.ensemble import ExtraTreesClassifier

from .features import compute_rsid_features, fast_signed_distance_transform


_BONE_OOF_PROB_CACHE = {}
_BONE_V2_SESSION_CACHE = {}
_LAST_V2_REFINER_CLF = None


def _vol_fingerprint(img):
    flat = img.ravel()
    step = max(1, len(flat) // 64)
    return (img.shape, float(flat[::step].sum()), float(flat[:100].mean()))


def _infer_default_spacing(shape):
    if shape[1] >= 1.8 * shape[0]:
        return (1.2, 0.4, 0.4)
    return (1.0, 1.0, 1.0)


def _largest_cc(mask):
    labeled, n_comp = label(mask)
    if n_comp == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == np.argmax(sizes)


def _postprocess_bone_mask(mask):
    if not np.any(mask):
        return mask
    cc = _largest_cc(mask)
    inds = np.argwhere(cc)
    z0, y0, x0 = np.maximum(inds.min(axis=0) - 1, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + 2, cc.shape)
    crop = cc[z0:z1, y0:y1, x0:x1]
    filled = binary_fill_holes(crop)
    for z in range(filled.shape[0]):
        filled[z] = binary_fill_holes(filled[z])
    out = np.zeros_like(cc, dtype=bool)
    out[z0:z1, y0:y1, x0:x1] = filled
    return out


def _compute_shifted_diffs_3d(vol, z_idx, y_idx, x_idx, offsets_mm, spacing):
    z_max, y_max, x_max = vol.shape
    n_vox = len(z_idx)
    n_shifts = len(offsets_mm)
    out = np.empty((n_vox, n_shifts), dtype=np.float32)
    base_vals = vol[z_idx, y_idx, x_idx]
    for i in range(n_shifts):
        dz = int(round(offsets_mm[i, 0] / spacing[0]))
        dy = int(round(offsets_mm[i, 1] / spacing[1]))
        dx = int(round(offsets_mm[i, 2] / spacing[2]))
        sz = np.clip(z_idx + dz, 0, z_max - 1)
        sy = np.clip(y_idx + dy, 0, y_max - 1)
        sx = np.clip(x_idx + dx, 0, x_max - 1)
        out[:, i] = base_vals - vol[sz, sy, sx]
    return out


def _extract_bone_p1_masked(img_s, sp_s, mask_s):
    p5, p50, p95 = np.percentile(img_s, [5, 50, 95])
    scale = max(float(p95 - p5), 1.0)
    img_norm = ((img_s.astype(np.float32) - p50) / scale).astype(np.float32)

    inds = np.argwhere(mask_s)
    z_idx, y_idx, x_idx = inds[:, 0], inds[:, 1], inds[:, 2]
    base_val = img_norm[z_idx, y_idx, x_idx]

    g_dict = {}
    grad_dict = {}
    for sig_mm in [1.0, 2.0, 4.0, 8.0, 14.0]:
        g = gaussian_filter(img_norm, sigma=[sig_mm / s for s in sp_s])
        g_dict[sig_mm] = g[z_idx, y_idx, x_idx]
        if sig_mm in [1.0, 2.0, 4.0]:
            gz, gy, gx = np.gradient(g, *sp_s)
            grad_dict[sig_mm] = np.sqrt(gz**2 + gy**2 + gx**2)[z_idx, y_idx, x_idx]

    zn = z_idx.astype(np.float32) / img_s.shape[0]
    yn = y_idx.astype(np.float32) / img_s.shape[1]
    xn = x_idx.astype(np.float32) / img_s.shape[2]
    r_yx = np.sqrt((yn - 0.5) ** 2 + (xn - 0.5) ** 2)
    r_3d = np.sqrt((zn - 0.5) ** 2 + (yn - 0.5) ** 2 + (xn - 0.5) ** 2)

    cols = [
        base_val,
        g_dict[1.0],
        g_dict[2.0],
        g_dict[4.0],
        g_dict[8.0],
        g_dict[14.0],
        base_val - g_dict[4.0],
        base_val - g_dict[8.0],
        base_val - g_dict[14.0],
        g_dict[1.0] - g_dict[2.0],
        g_dict[2.0] - g_dict[4.0],
        g_dict[4.0] - g_dict[8.0],
        g_dict[8.0] - g_dict[14.0],
        grad_dict[1.0],
        grad_dict[2.0],
        grad_dict[4.0],
        zn,
        yn,
        xn,
        np.abs(zn - 0.5),
        r_yx,
        r_3d,
    ]
    rng = np.random.RandomState(42)
    offsets = np.vstack([
        rng.uniform(-15.0, 15.0, size=(20, 3)),
        rng.uniform(-40.0, 40.0, size=(20, 3)),
    ])
    rsid = _compute_shifted_diffs_3d(img_norm, z_idx, y_idx, x_idx, offsets, sp_s)
    return np.hstack([np.column_stack(cols), rsid])


def _extract_bone_p2_shell(img, sp, prob_p1, dts_p1_list, shell_mask):
    p5, p50, p95 = np.percentile(img, [5, 50, 95])
    scale = max(float(p95 - p5), 1.0)
    img_norm = ((img.astype(np.float32) - p50) / scale).astype(np.float32)

    inds = np.argwhere(shell_mask)
    z_idx, y_idx, x_idx = inds[:, 0], inds[:, 1], inds[:, 2]
    margin = [int(np.ceil(10.0 / s)) for s in sp]
    z0, y0, x0 = np.maximum(inds.min(axis=0) - margin, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + margin + 1, img.shape)
    zc, yc, xc = z_idx - z0, y_idx - y0, x_idx - x0

    img_crop = img_norm[z0:z1, y0:y1, x0:x1]
    base_val = img_norm[z_idx, y_idx, x_idx]

    g_dict = {}
    grad_dict = {}
    for sig_mm in [0.6, 1.2, 2.5, 5.0, 10.0]:
        g = gaussian_filter(img_crop, sigma=[sig_mm / s for s in sp])
        g_dict[sig_mm] = g[zc, yc, xc]
        if sig_mm in [0.6, 1.2, 2.5]:
            gz, gy, gx = np.gradient(g, *sp)
            grad_dict[sig_mm] = np.sqrt(gz**2 + gy**2 + gx**2)[zc, yc, xc]

    zn = z_idx.astype(np.float32) / img.shape[0]
    yn = y_idx.astype(np.float32) / img.shape[1]
    xn = x_idx.astype(np.float32) / img.shape[2]

    cols = [
        base_val,
        g_dict[0.6],
        g_dict[1.2],
        g_dict[2.5],
        g_dict[5.0],
        g_dict[10.0],
        base_val - g_dict[2.5],
        base_val - g_dict[5.0],
        base_val - g_dict[10.0],
        g_dict[0.6] - g_dict[1.2],
        g_dict[1.2] - g_dict[2.5],
        g_dict[2.5] - g_dict[5.0],
        g_dict[5.0] - g_dict[10.0],
        grad_dict[0.6],
        grad_dict[1.2],
        grad_dict[2.5],
        zn,
        yn,
        xn,
    ]

    # Add Pass-1 probabilities & Pass-1 SDTs for each bone channel
    n_ch = prob_p1.shape[-1]
    for c in range(1, n_ch):
        pc = prob_p1[..., c].astype(np.float32)
        pc_s = gaussian_filter(pc[z0:z1, y0:y1, x0:x1], sigma=[2.5 / s for s in sp])
        cols.append(pc[z_idx, y_idx, x_idx])
        cols.append(pc_s[zc, yc, xc])

    for dt_vol in dts_p1_list:
        cols.append(dt_vol[z_idx, y_idx, x_idx])
    if len(dts_p1_list) >= 2:
        df_m = dts_p1_list[0][z_idx, y_idx, x_idx]
        dt_m = dts_p1_list[1][z_idx, y_idx, x_idx]
        cols.append(df_m - dt_m)
        cols.append(np.minimum(np.abs(df_m), np.abs(dt_m)))

    rng = np.random.RandomState(42)
    offsets_i = np.vstack([
        rng.uniform(-6.0, 6.0, size=(15, 3)),
        rng.uniform(-20.0, 20.0, size=(15, 3)),
    ])
    rsid = _compute_shifted_diffs_3d(img_norm, z_idx, y_idx, x_idx, offsets_i, sp)

    rng_p = np.random.RandomState(99)
    offsets_p = rng_p.uniform(-12.0, 12.0, size=(10, 3))
    rspd_list = [
        _compute_shifted_diffs_3d(prob_p1[..., c], z_idx, y_idx, x_idx, offsets_p, sp)
        for c in range(1, n_ch)
    ]
    return np.hstack([np.column_stack(cols), rsid] + rspd_list)


class BoneClassifier:
    def __init__(self, n_estimators=100, max_depth=25, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.stage = 1
        self.bone_Map = {'femur': 1, 'tibia': 2}
        self.clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=n_jobs,
            random_state=42,
        )

    def extract_bone_features(self, image, prob_map=None, spacing=None, target_dtype='float32'):
        """
        Extract features suitable for Bone Segmentation (legacy 26-feature interface).
        """
        if isinstance(target_dtype, str):
            if target_dtype == 'bfloat16':
                dtype = ml_dtypes.bfloat16
            else:
                dtype = np.float32
        else:
            dtype = target_dtype

        features = []
        img_mean = image.mean()
        img_std = image.std()
        img_norm = (image.astype(np.float32) - img_mean) / (img_std + 1e-6)
        img_norm = img_norm.astype(dtype)

        features.append(img_norm.flatten())
        features.append(gaussian_filter(img_norm.astype(np.float32), sigma=2.0).astype(dtype).flatten())
        features.append(gaussian_filter(img_norm.astype(np.float32), sigma=4.0).astype(dtype).flatten())

        z, y, x = np.mgrid[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]]
        features.append((z.flatten() / image.shape[0]).astype(dtype))
        features.append((y.flatten() / image.shape[1]).astype(dtype))
        features.append((x.flatten() / image.shape[2]).astype(dtype))

        rsid = compute_rsid_features(img_norm, num_shifts=20, max_shift=20, seed=42, dtype=dtype)
        for i in range(rsid.shape[-1]):
            features.append(rsid[..., i].flatten())

        if prob_map is not None:
            for c in range(prob_map.shape[-1]):
                p_ch = prob_map[..., c].astype(dtype)
                features.append(p_ch.flatten())
                features.append(gaussian_filter(p_ch.astype(np.float32), sigma=2.0).astype(dtype).flatten())

        return np.stack(features, axis=1)

    @staticmethod
    def _map_pure_bone_labels(lbl):
        """
        Map pure bone labels only (do NOT merge cartilage labels 2, 4, 6 into bone).
        1 (Femur) -> 1
        3 (Tibia) -> 2
        5 (Patella) -> 3
        """
        y_mapped = np.zeros_like(lbl, dtype=np.uint8)
        y_mapped[lbl == 1] = 1
        y_mapped[lbl == 3] = 2
        y_mapped[lbl == 5] = 3
        return y_mapped

    def train(self, images, labels, prob_maps=None, subsample=50000, dtype='float32', spacings=None):
        """
        Trains BoneClassifier (Pass 1 when prob_maps is None, Pass 2 boundary refinement when prob_maps is provided).
        """
        # Small synthetic volume fallback (for unit tests with 10x10x10 arrays)
        if len(images) > 0 and min(images[0].shape) < 32:
            self.stage = 0
            X_all, y_all = [], []
            for i, (img, lbl) in enumerate(zip(images, labels)):
                y_mapped = self._map_pure_bone_labels(lbl)
                mask_bone = y_mapped > 0
                target_indices = np.argwhere(mask_bone)
                mask_near = binary_dilation(mask_bone, iterations=3) & (~mask_bone)
                near_indices = np.argwhere(mask_near)
                far_indices = np.argwhere((~mask_bone) & (~mask_near))

                coords_list = []
                if len(target_indices) > 0:
                    idx_b = np.random.choice(len(target_indices), min(len(target_indices), max(1, subsample // 2)), replace=False)
                    coords_list.append(target_indices[idx_b])
                if len(near_indices) > 0:
                    idx_n = np.random.choice(len(near_indices), min(len(near_indices), max(1, subsample // 4)), replace=False)
                    coords_list.append(near_indices[idx_n])
                if len(far_indices) > 0:
                    idx_f = np.random.choice(len(far_indices), min(len(far_indices), max(1, subsample // 4)), replace=False)
                    coords_list.append(far_indices[idx_f])
                coords = np.vstack(coords_list)
                pm = prob_maps[i] if prob_maps else None
                feats_flat = self.extract_bone_features(img, prob_map=pm, target_dtype=dtype)
                flat_indices = np.ravel_multi_index(coords.T, img.shape)
                X_all.append(feats_flat[flat_indices])
                y_all.append(y_mapped.flatten()[flat_indices])
            self.clf.fit(np.vstack(X_all), np.concatenate(y_all))
            return

        if spacings is None:
            spacings = [_infer_default_spacing(img.shape) for img in images]

        if prob_maps is None:
            # --- PASS 1: Strided (1, 2, 2) Physical-Space Global + Boundary Forest ---
            self.stage = 1
            print(f"    Extracting Pass-1 strided physical-space bone features ({len(images)} volumes)...")

            def _sample_p1_case(idx):
                img, lbl, sp = images[idx], labels[idx], spacings[idx]
                stride = (1, 2, 2)
                img_s = img[::stride[0], ::stride[1], ::stride[2]]
                lbl_s = lbl[::stride[0], ::stride[1], ::stride[2]]
                sp_s = tuple(s * st for s, st in zip(sp, stride))

                y_clean = self._map_pure_bone_labels(lbl_s)
                dts_gt = [
                    fast_signed_distance_transform(y_clean == c, sp_s)
                    for c in [1, 2, 3]
                    if np.any(y_clean == c)
                ]
                min_abs_d = np.min([np.abs(d) for d in dts_gt], axis=0) if dts_gt else np.zeros_like(img_s, dtype=np.float32)

                rng = np.random.RandomState(42 + idx)
                cart_mask = (lbl_s == 2) | (lbl_s == 4) | (lbl_s == 6)
                subchondral_mask = binary_dilation(cart_mask, iterations=2) & (y_clean > 0)
                shell = np.argwhere(min_abs_d <= 8.0)
                cart_neg = np.argwhere(cart_mask)
                sub_pos = np.argwhere(subchondral_mask)
                interior = np.argwhere((y_clean > 0) & (min_abs_d > 8.0))
                far_bg = np.argwhere((y_clean == 0) & (min_abs_d > 8.0))

                n_interface = min(len(cart_neg), len(sub_pos), 4500)
                arrs = []
                for pool, n_take in [
                    (shell, 18000),
                    (cart_neg, n_interface),
                    (sub_pos, n_interface),
                    (interior, 10000),
                    (far_bg, 8000),
                ]:
                    if len(pool) > 0 and n_take > 0:
                        arrs.append(pool[rng.choice(len(pool), min(len(pool), n_take), replace=False)])

                sample_mask = np.zeros_like(y_clean, dtype=bool)
                for a in arrs:
                    sample_mask[a[:, 0], a[:, 1], a[:, 2]] = True

                X_c = _extract_bone_p1_masked(img_s, sp_s, sample_mask)
                y_c = y_clean[sample_mask]
                return X_c, y_c

            sampled = Parallel(n_jobs=min(16, os.cpu_count() or 4, max(1, len(images))), prefer='threads')(
                delayed(_sample_p1_case)(i) for i in range(len(images))
            )
            X_train = np.vstack([s[0] for s in sampled])
            y_train = np.concatenate([s[1] for s in sampled])
            print(f"    Fitting Pass-1 Bone Forest on {X_train.shape}...")
            self.clf.fit(X_train, y_train)

            if len(images) >= 4:
                mid = len(images) // 2
                n_mid = sum(len(s[1]) for s in sampled[:mid])
                clf_a = ExtraTreesClassifier(
                    n_estimators=65, max_depth=self.max_depth, min_samples_leaf=8, max_features='sqrt', n_jobs=self.n_jobs, random_state=42
                ).fit(X_train[:n_mid], y_train[:n_mid])
                clf_b = ExtraTreesClassifier(
                    n_estimators=65, max_depth=self.max_depth, min_samples_leaf=8, max_features='sqrt', n_jobs=self.n_jobs, random_state=43
                ).fit(X_train[n_mid:], y_train[n_mid:])

                print("    Precomputing 2-fold Out-of-Fold Pass-1 Bone Probability Maps...")
                _BONE_OOF_PROB_CACHE.clear()
                n_classes = getattr(self.clf, 'n_classes_', len(self.clf.classes_))

                def _prep_p1_infer(idx):
                    img, sp = images[idx], spacings[idx]
                    stride = (1, 2, 2)
                    img_s = img[::stride[0], ::stride[1], ::stride[2]]
                    sp_s = tuple(s * st for s, st in zip(sp, stride))
                    X_s = _extract_bone_p1_masked(img_s, sp_s, np.ones(img_s.shape, dtype=bool))
                    return img, img_s, sp_s, X_s

                def _zoom_p1_probs(img_shape, img_s_shape, sp_s, probs_s):
                    for c in range(n_classes):
                        probs_s[..., c] = gaussian_filter(probs_s[..., c], sigma=[1.2 / s for s in sp_s])
                    zf = [img_shape[d] / img_s_shape[d] for d in range(3)]
                    pf = np.zeros((*img_shape, n_classes), dtype=np.float32)
                    for c in range(n_classes):
                        pf[..., c] = zoom(probs_s[..., c], zf, order=1)
                    return pf

                for start in range(0, len(images), 10):
                    idxs = list(range(start, min(start + 10, len(images))))
                    preps = Parallel(n_jobs=len(idxs), prefer='threads')(delayed(_prep_p1_infer)(i) for i in idxs)
                    probs_s_list = []
                    for i, (img, img_s, sp_s, X_s) in zip(idxs, preps):
                        c_oof = clf_b if i < mid else clf_a
                        ps = c_oof.predict_proba(X_s).reshape(*img_s.shape, n_classes)
                        probs_s_list.append((img, img_s.shape, sp_s, ps))
                    pfs = Parallel(n_jobs=len(idxs), prefer='threads')(
                        delayed(_zoom_p1_probs)(it[0].shape, it[1], it[2], it[3]) for it in probs_s_list
                    )
                    for (img, _, _, _), pf in zip(probs_s_list, pfs):
                        while len(_BONE_OOF_PROB_CACHE) >= 128:
                            _BONE_OOF_PROB_CACHE.pop(next(iter(_BONE_OOF_PROB_CACHE)), None)
                        _BONE_OOF_PROB_CACHE[_vol_fingerprint(img)] = pf
        else:
            # --- PASS 2: Full-Resolution Boundary-Shell + Pass-1 SDT Refinement ---
            self.stage = 2
            print(f"    Extracting Pass-2 full-resolution boundary-shell bone features ({len(images)} volumes)...")

            def _sample_p2_case(idx):
                img, lbl, pm, sp = images[idx], labels[idx], prob_maps[idx], spacings[idx]
                y_clean = self._map_pure_bone_labels(lbl)
                n_ch = pm.shape[-1]
                dts_p1_list = []
                for c in range(1, n_ch):
                    m_p1 = _largest_cc((pm[..., c] >= 0.42) & (pm[..., c] == np.max(pm, axis=-1)))
                    dts_p1_list.append(fast_signed_distance_transform(m_p1, sp))

                dts_gt = [
                    fast_signed_distance_transform(y_clean == c, sp)
                    for c in range(1, n_ch)
                    if np.any(y_clean == c)
                ]
                min_abs_gt = np.min([np.abs(d) for d in dts_gt], axis=0) if dts_gt else np.zeros_like(img, dtype=np.float32)

                rng = np.random.RandomState(200 + idx)
                cart_mask = (lbl == 2) | (lbl == 4) | (lbl == 6)
                subchondral_mask = binary_dilation(cart_mask, iterations=3) & (y_clean > 0)
                tight_shell = np.argwhere(min_abs_gt <= 5.0)
                cart_neg = np.argwhere(cart_mask)
                sub_pos = np.argwhere(subchondral_mask)
                mid_shell = np.argwhere((min_abs_gt > 5.0) & (min_abs_gt <= 10.0))

                n_interface = min(len(cart_neg), len(sub_pos), 6000)
                arrs = []
                for pool, n_take in [
                    (tight_shell, 18000),
                    (cart_neg, n_interface),
                    (sub_pos, n_interface),
                    (mid_shell, 8000),
                ]:
                    if len(pool) > 0 and n_take > 0:
                        arrs.append(pool[rng.choice(len(pool), min(len(pool), n_take), replace=False)])

                sample_mask = np.zeros_like(y_clean, dtype=bool)
                for a in arrs:
                    sample_mask[a[:, 0], a[:, 1], a[:, 2]] = True

                X_c = _extract_bone_p2_shell(img, sp, pm, dts_p1_list, sample_mask)
                y_c = y_clean[sample_mask]
                return X_c, y_c

            sampled = Parallel(n_jobs=min(16, os.cpu_count() or 4, max(1, len(images))), prefer='threads')(
                delayed(_sample_p2_case)(i) for i in range(len(images))
            )
            X_train = np.vstack([s[0] for s in sampled])
            y_train = np.concatenate([s[1] for s in sampled])
            print(f"    Fitting Pass-2 Bone Forest on {X_train.shape}...")
            self.clf.fit(X_train, y_train)
            _BONE_OOF_PROB_CACHE.clear()

    def predict(self, image, prob_map=None, dtype='float32', spacing=None, pass1_probs=None):
        """
        Predicts bone labels and probability map for a 3D volume.
        """
        if prob_map is None and pass1_probs is not None:
            prob_map = pass1_probs

        if spacing is None:
            spacing = _infer_default_spacing(image.shape)

        if getattr(self, 'stage', 1) == 'v2_p1':
            return self._predict_v2_p1(image, spacing)

        if getattr(self, 'stage', 1) == 'v2_p2':
            return self._predict_v2_refiner(image, spacing, prob_map=prob_map)

        if getattr(self, 'stage', 1) == 'oai_p1':
            from .oai_subchondral_atlas import predict_oai_bone_p1
            return predict_oai_bone_p1(image, spacing, self.meta)

        if getattr(self, 'stage', 1) == 'oai_p2':
            from .oai_subchondral_atlas import predict_oai_bone_p2
            p1_d = self.meta.get('p1_dict')
            if p1_d is None and getattr(self, 'model_dir', None):
                p1_path = os.path.join(self.model_dir, 'bone_rf_p1.joblib')
                if os.path.exists(p1_path):
                    p1_d = joblib.load(p1_path)
            ref_v2_d = self.meta.get('ref_v2') or self.meta['models']
            return predict_oai_bone_p2(image, prob_map, spacing, ref_v2_d, p1_dict=p1_d)

        if getattr(self, 'stage', 1) == 'unified_p1':
            from .oai_subchondral_atlas import predict_unified_bone_p1
            return predict_unified_bone_p1(image, spacing, self.meta, ski10_clf_helper=self)

        if getattr(self, 'stage', 1) == 'unified_p2':
            from .oai_subchondral_atlas import predict_unified_bone_p2
            return predict_unified_bone_p2(image, prob_map, spacing, self.meta, ski10_clf_helper=self)

        if min(image.shape) < 32 or getattr(self, 'stage', 1) == 0:
            feats_flat = self.extract_bone_features(image, prob_map, target_dtype=dtype)
            y_pred = self.clf.predict(feats_flat)
            y_prob = self.clf.predict_proba(feats_flat)
            return y_pred.reshape(image.shape), y_prob.reshape(image.shape + (-1,))

        fp = _vol_fingerprint(image)
        if (prob_map is None or getattr(self, 'stage', 1) == 1) and fp in _BONE_OOF_PROB_CACHE:
            pf = _BONE_OOF_PROB_CACHE.pop(fp)
            if isinstance(pf, np.ndarray):
                return np.argmax(pf, axis=-1).astype(np.uint8), pf

        n_classes = getattr(self.clf, 'n_classes_', len(self.clf.classes_))

        if prob_map is None or getattr(self, 'stage', 1) == 1:
            # --- PASS 1 INFERENCE ---
            stride = (1, 2, 2)
            img_s = image[::stride[0], ::stride[1], ::stride[2]]
            sp_s = tuple(s * st for s, st in zip(spacing, stride))
            full_s = np.ones(img_s.shape, dtype=bool)
            X_s = _extract_bone_p1_masked(img_s, sp_s, full_s)
            probs_s = self.clf.predict_proba(X_s).reshape(*img_s.shape, n_classes)

            for c in range(n_classes):
                probs_s[..., c] = gaussian_filter(probs_s[..., c], sigma=[1.2 / s for s in sp_s])

            zoom_factors = [image.shape[d] / img_s.shape[d] for d in range(3)]
            probs_full = np.zeros((*image.shape, n_classes), dtype=np.float32)
            for c in range(n_classes):
                probs_full[..., c] = zoom(probs_s[..., c], zoom_factors, order=1)

            pred = np.argmax(probs_full, axis=-1).astype(np.uint8)
            clean_pred = np.zeros_like(pred, dtype=np.uint8)
            for c in range(1, n_classes):
                clean_pred[_postprocess_bone_mask(pred == c)] = c
            return clean_pred, probs_full
        else:
            # --- PASS 2 INFERENCE (Full-Resolution Boundary Shell) ---
            dts_p1_list = []
            p1_masks = []
            for c in range(1, n_classes):
                m_p1 = _largest_cc((prob_map[..., c] >= 0.42) & (prob_map[..., c] == np.max(prob_map, axis=-1)))
                p1_masks.append(m_p1)
                dts_p1_list.append(fast_signed_distance_transform(m_p1, spacing))

            shell_mask = np.zeros(image.shape, dtype=bool)
            for dt_vol in dts_p1_list:
                shell_mask |= np.abs(dt_vol) <= 10.0

            X_shell = _extract_bone_p2_shell(image, spacing, prob_map, dts_p1_list, shell_mask)
            probs_shell = self.clf.predict_proba(X_shell)

            probs_p2 = prob_map.copy()
            probs_p2[shell_mask] = probs_shell
            for c in range(1, n_classes):
                probs_p2[..., c] = gaussian_filter(probs_p2[..., c], sigma=[1.0 / spacing[0], 0.8 / spacing[1], 0.8 / spacing[2]])

            clean_pred = np.zeros(image.shape, dtype=np.uint8)
            for idx_c, c in enumerate(range(1, n_classes)):
                dt_c = dts_p1_list[idx_c]
                m_c = ((probs_p2[..., c] >= 0.48) & (probs_p2[..., c] == np.max(probs_p2, axis=-1))) | (dt_c < -10.0)
                m_c = _postprocess_bone_mask(m_c)
                clean_pred[m_c] = c

            return clean_pred, probs_p2

    def _get_tierA(self):
        if not hasattr(self, '_tierA_cached') or self._tierA_cached is None:
            meta = getattr(self, 'meta', {}) or {}
            tierA = meta.get('tierA')
            if tierA is None and 'ski10_p1' in meta and isinstance(meta['ski10_p1'], dict):
                tierA = meta['ski10_p1'].get('tierA')
            if tierA is None and 'ski10_p2' in meta and isinstance(meta['ski10_p2'], dict):
                tierA = meta['ski10_p2'].get('tierA')
            if tierA is None and getattr(self, 'model_dir', None):
                p1_path = os.path.join(self.model_dir, 'bone_rf_p1.joblib')
                if os.path.exists(p1_path):
                    p1_data = joblib.load(p1_path)
                    if isinstance(p1_data, dict):
                        tierA = p1_data.get('tierA') or (p1_data.get('ski10_p1', {}) or {}).get('tierA')
            self._tierA_cached = tierA
        return self._tierA_cached

    def _get_ref_v2(self):
        if not hasattr(self, '_ref_v2_cached') or self._ref_v2_cached is None:
            meta = getattr(self, 'meta', {}) or {}
            ref_v2 = meta.get('ref_v2')
            if ref_v2 is None and 'ski10_p2' in meta:
                ref_v2 = meta['ski10_p2']['ref_v2']
            max_et_jobs = min(8, os.cpu_count() or 4)
            for b in ('tibia', 'femur', 'patella'):
                if b in ref_v2 and isinstance(ref_v2[b], dict):
                    for ek in ('et_v1', 'et_v2', 'et_whole', 'et_sub'):
                        et = ref_v2[b].get(ek)
                        if et is not None and (getattr(et, 'n_jobs', max_et_jobs) < 0 or getattr(et, 'n_jobs', max_et_jobs) > max_et_jobs):
                            et.n_jobs = max_et_jobs
            self._ref_v2_cached = ref_v2
        return self._ref_v2_cached

    def _predict_v2_p1(self, image, spacing):
        fp = _vol_fingerprint(image)
        tierA = self._get_tierA()
        p_a = predict_tierA_probs(image, spacing, tierA)
        bm_raw = predict_bones_from_probs(p_a, spacing)

        clean_pred = np.zeros(image.shape, dtype=np.uint8)
        clean_pred[bm_raw['femur'] > 0] = 1
        clean_pred[bm_raw['tibia'] > 0] = 2

        probs_p1 = np.zeros((*image.shape, 3), dtype=np.float32)
        probs_p1[..., 1] = p_a['femur']
        probs_p1[..., 2] = p_a['tibia']
        probs_p1[..., 0] = np.clip(1.0 - probs_p1[..., 1] - probs_p1[..., 2], 0.0, 1.0)

        while len(_BONE_V2_SESSION_CACHE) >= 16:
            _BONE_V2_SESSION_CACHE.pop(next(iter(_BONE_V2_SESSION_CACHE)), None)
        _BONE_V2_SESSION_CACHE[fp] = {
            'bm_raw': bm_raw,
            'p_a': p_a,
            'prob_p1_fp': _vol_fingerprint(probs_p1),
        }
        return clean_pred, probs_p1

    def _predict_v2_refiner(self, image, spacing, prob_map=None):
        global _LAST_V2_REFINER_CLF
        _LAST_V2_REFINER_CLF = self
        fp = _vol_fingerprint(image)
        cached = _BONE_V2_SESSION_CACHE.get(fp)
        has_valid_prob_map = (
            prob_map is not None
            and prob_map.ndim == 4
            and prob_map.shape[:3] == image.shape
            and prob_map.shape[3] >= 3
            and np.any(prob_map[..., 1:3] > 0.2)
        )
        prob_matches_cache = (
            isinstance(cached, dict)
            and 'bm_raw' in cached
            and (prob_map is None or _vol_fingerprint(prob_map) == cached.get('prob_p1_fp'))
        )
        if prob_matches_cache:
            bm_raw = cached['bm_raw']
        elif has_valid_prob_map:
            p_a = {
                'femur': prob_map[..., 1].astype(np.float32),
                'tibia': prob_map[..., 2].astype(np.float32),
            }
            bm_raw = predict_bones_from_probs(p_a, spacing)
        elif isinstance(cached, dict) and 'bm_raw' in cached:
            bm_raw = cached['bm_raw']
        else:
            tierA = self._get_tierA()
            p_a = predict_tierA_probs(image, spacing, tierA)
            bm_raw = predict_bones_from_probs(p_a, spacing)

        ref_v2 = self._get_ref_v2()
        bm, bm_v1, probs_v2 = refine_bone_v2_single(
            image, spacing, bm_raw, ref_v2, return_v1=True, return_prob=True
        )
        aux = {'bm_v2': bm, 'bm_v1': bm_v1, 'bm_raw': bm_raw}
        while len(_BONE_V2_SESSION_CACHE) >= 16:
            _BONE_V2_SESSION_CACHE.pop(next(iter(_BONE_V2_SESSION_CACHE)), None)
        while len(_BONE_OOF_PROB_CACHE) >= 16:
            _BONE_OOF_PROB_CACHE.pop(next(iter(_BONE_OOF_PROB_CACHE)), None)
        _BONE_V2_SESSION_CACHE[fp] = aux
        _BONE_OOF_PROB_CACHE[fp] = aux

        clean_pred = np.zeros(image.shape, dtype=np.uint8)
        clean_pred[bm['femur'] > 0] = 1
        clean_pred[bm['tibia'] > 0] = 2
        probs_p2 = np.zeros((*image.shape, 3), dtype=np.float32)
        probs_p2[..., 1] = np.clip(probs_v2['femur'], 0.0, 1.0)
        probs_p2[..., 2] = np.clip(probs_v2['tibia'], 0.0, 1.0)
        s_fg = probs_p2[..., 1] + probs_p2[..., 2]
        over = s_fg > 1.0
        if np.any(over):
            probs_p2[over, 1] /= s_fg[over]
            probs_p2[over, 2] /= s_fg[over]
        probs_p2[..., 0] = np.clip(1.0 - probs_p2[..., 1] - probs_p2[..., 2], 0.0, 1.0)
        return clean_pred, probs_p2

    def save(self, path):
        payload = dict(getattr(self, 'meta', {}) or {})
        payload['clf'] = self.clf
        payload['stage'] = getattr(self, 'stage', 1)
        joblib.dump(payload, path, compress=3)

    def load(self, path):
        global _LAST_V2_REFINER_CLF
        self.model_dir = os.path.dirname(os.path.abspath(path))
        data = joblib.load(path)
        if isinstance(data, dict) and data.get('format_version') in (
            'oai_3bone_v2', 'oai_3bone_refiner_v2', 'unified_ski10_oai_bone_p1', 'unified_ski10_oai_bone_p2'
        ):
            fmt = data['format_version']
            self.clf = None
            self.meta = data
            self.bone_Map = {'femur': 1, 'tibia': 2, 'patella': 3}
            if fmt == 'oai_3bone_v2':
                self.stage = 'oai_p1'
            elif fmt == 'oai_3bone_refiner_v2':
                self.stage = 'oai_p2'
            elif fmt == 'unified_ski10_oai_bone_p1':
                self.stage = 'unified_p1'
            else:
                self.stage = 'unified_p2'
                _LAST_V2_REFINER_CLF = self
            return

        if isinstance(data, dict) and 'clf' in data:
            self.clf = data['clf']
            self.stage = data.get('stage', 1)
            self.meta = data
            if self.stage in ('v2_p2', 'v2_refiner'):
                _LAST_V2_REFINER_CLF = self
        else:
            self.clf = data
            self.stage = 0
            self.meta = {}
        classes = getattr(self.clf, 'classes_', []) if self.clf is not None else []
        if len(classes) >= 4 or (3 in classes):
            self.bone_Map = {'femur': 1, 'tibia': 2, 'patella': 3}
        else:
            self.bone_Map = {'femur': 1, 'tibia': 2}


def get_or_compute_bone_v2_aux(image, spacing, bm_v2=None, model_dir=None):
    """Retrieve or reconstruct (bm_v2, bm_v1, bm_raw) auxiliary bone masks for V6 cartilage inference."""
    fp = _vol_fingerprint(image)
    cached = _BONE_V2_SESSION_CACHE.get(fp)
    if isinstance(cached, dict) and 'bm_v1' in cached and 'bm_raw' in cached:
        return cached

    if min(image.shape) >= 32:
        clf = _LAST_V2_REFINER_CLF
        if clf is None and model_dir:
            cand_path = os.path.join(model_dir, 'bone_rf_p2.joblib')
            if os.path.exists(cand_path):
                clf = BoneClassifier()
                clf.load(cand_path)
        if clf is not None:
            if getattr(clf, 'stage', None) == 'unified_p2':
                clf.predict(image, prob_map=None, spacing=spacing)
            elif getattr(clf, 'stage', None) in ('v2_p2', 'v2_refiner'):
                clf._predict_v2_refiner(image, spacing, prob_map=None)
            cached = _BONE_V2_SESSION_CACHE.get(fp)
            if isinstance(cached, dict) and 'bm_v1' in cached and 'bm_raw' in cached:
                return cached

    return cached if isinstance(cached, dict) else {}


# ==============================================================================
# TIER A: Joint-Anchored 3D Bone Spatial Prior Atlas + Continuous 1D Cortical
#         Enclosure Ray Global Corrector
# ==============================================================================


def compute_joint_frame_strided(fb_s, tb_s, sp_s):
    df_s = fast_signed_distance_transform(fb_s, sp_s)
    dt_s = fast_signed_distance_transform(tb_s, sp_s)
    j_mask = (df_s <= 12.0) & (dt_s <= 12.0) & (np.abs(df_s - dt_s) <= 8.0)
    if np.sum(j_mask) < 50:
        j_mask = (df_s <= 20.0) & (dt_s <= 20.0)
    if np.sum(j_mask) < 10:
        Z, Y, X = fb_s.shape
        c_j = np.array([0.5 * Z * sp_s[0], 0.55 * Y * sp_s[1], 0.5 * X * sp_s[2]], dtype=np.float32)
        return df_s, dt_s, c_j, 65.0
    pts = np.argwhere(j_mask).astype(np.float32) * np.array(sp_s, dtype=np.float32)
    c_j = np.median(pts, axis=0).astype(np.float32)
    span_z = max(float(np.percentile(pts[:, 0], 95) - np.percentile(pts[:, 0], 5)), 40.0)
    return df_s, dt_s, c_j, span_z


def get_joint_bins(shape, sp_s, c_j, span_z):
    Z, Y, X = shape
    bz1 = np.clip((np.arange(Z) / Z * 20).astype(int), 0, 19)[:, None, None]
    by1 = np.clip((np.arange(Y) / Y * 32).astype(int), 0, 31)[None, :, None]
    bx1 = np.clip((np.arange(X) / X * 32).astype(int), 0, 31)[None, None, :]
    flat1 = (bz1 * 1024 + by1 * 32 + bx1).ravel()
    jz = (np.arange(Z) * sp_s[0] - c_j[0]) / span_z
    jy = (np.arange(Y) * sp_s[1] - c_j[1]) / span_z
    jx = (np.arange(X) * sp_s[2] - c_j[2]) / span_z
    bz2 = np.clip(((jz + 0.70) / 1.40 * 24).astype(int), 0, 23)[:, None, None]
    by2 = np.clip(((jy + 1.20) / 2.40 * 40).astype(int), 0, 39)[None, :, None]
    bx2 = np.clip(((jx + 0.80) / 1.60 * 32).astype(int), 0, 31)[None, None, :]
    flat2 = (bz2 * 1280 + by2 * 32 + bx2).ravel()
    bzs = np.clip((np.abs(jz) / 0.70 * 16).astype(int), 0, 15)[:, None, None]
    flats = (bzs * 1280 + by2 * 32 + bx2).ravel()
    return flat1, flat2, flats


def dir_min_max_1d(vol, axis, d_max_mm, sp_axis):
    L = max(3, int(round(d_max_mm / sp_axis)))
    if L % 2 == 0:
        L += 1
    half = L // 2
    vmin_neg = minimum_filter1d(vol, size=L, axis=axis, origin=half, mode='nearest')
    vmin_pos = minimum_filter1d(vol, size=L, axis=axis, origin=-half, mode='nearest')
    vmax_neg = maximum_filter1d(vol, size=L, axis=axis, origin=half, mode='nearest')
    vmax_pos = maximum_filter1d(vol, size=L, axis=axis, origin=-half, mode='nearest')
    return vmin_neg, vmin_pos, vmax_neg, vmax_pos


def extract_tierA_features(img_s, sp_s, df_s, dt_s, c_j, span_z, mask_s, atl, rng_jit=None):
    p5, p50, p95 = np.percentile(img_s, [5, 50, 95])
    scale = max(float(p95 - p5), 1.0)
    img_norm = ((img_s - p50) / scale).astype(np.float32)

    inds = np.argwhere(mask_s)
    z_idx, y_idx, x_idx = inds[:, 0], inds[:, 1], inds[:, 2]
    Z, Y, X = img_s.shape
    base_val = img_norm[z_idx, y_idx, x_idx]

    g_dict, grad_dict = {}, {}
    for sig_mm in [1.0, 2.0, 4.0, 8.0, 14.0, 24.0]:
        g = gaussian_filter(img_norm, sigma=[sig_mm / s for s in sp_s])
        g_dict[sig_mm] = g
        if sig_mm in [1.0, 2.0, 4.0]:
            gz, gy, gx = np.gradient(g, *sp_s)
            grad_dict[sig_mm] = np.sqrt(gz**2 + gy**2 + gx**2)[z_idx, y_idx, x_idx]

    g1 = g_dict[1.0]
    g2 = g_dict[2.0]
    g24 = g_dict[24.0]
    var24 = gaussian_filter((img_norm - g24) ** 2, sigma=[24.0 / s for s in sp_s])
    std24 = np.sqrt(np.maximum(var24, 0.0) + 0.015)
    z_loc0 = ((img_norm - g24) / std24)[z_idx, y_idx, x_idx]
    z_loc1_vol = (g1 - g24) / std24
    z_loc2 = ((g2 - g24) / std24)[z_idx, y_idx, x_idx]

    y_med = np.percentile(g2, 50, axis=(0, 2))
    y_iqr = np.maximum(np.percentile(g2, 90, axis=(0, 2)) - np.percentile(g2, 10, axis=(0, 2)), 0.15)
    z_axial = (g2[z_idx, y_idx, x_idx] - y_med[y_idx]) / y_iqr[y_idx]

    bz1 = np.clip((z_idx / Z * 20).astype(int), 0, 19)
    by1 = np.clip((y_idx / Y * 32).astype(int), 0, 31)
    bx1 = np.clip((x_idx / X * 32).astype(int), 0, 31)
    f1 = bz1 * 1024 + by1 * 32 + bx1

    jz = (z_idx * sp_s[0] - c_j[0]) / span_z
    jy = (y_idx * sp_s[1] - c_j[1]) / span_z
    jx = (x_idx * sp_s[2] - c_j[2]) / span_z
    bz2 = np.clip(((jz + 0.70) / 1.40 * 24).astype(int), 0, 23)
    by2 = np.clip(((jy + 1.20) / 2.40 * 40).astype(int), 0, 39)
    bx2 = np.clip(((jx + 0.80) / 1.60 * 32).astype(int), 0, 31)
    f2 = bz2 * 1280 + by2 * 32 + bx2
    bzs = np.clip((np.abs(jz) / 0.70 * 16).astype(int), 0, 15)
    fs = bzs * 1280 + by2 * 32 + bx2

    pf_fov, pt_fov = atl[0][f1], atl[1][f1]
    pf_j, pt_j = atl[2][f2], atl[3][f2]
    pf_s, pt_s = atl[4][fs], atl[5][fs]
    p_fb = 0.45 * pf_j + 0.35 * pf_s + 0.20 * pf_fov
    p_tb = 0.45 * pt_j + 0.35 * pt_s + 0.20 * pt_fov
    log_ft = np.log((p_fb + 1e-4) / (p_tb + 1e-4))

    cont_cols = []
    g1_m = g1[z_idx, y_idx, x_idx]
    zl1_m = z_loc1_vol[z_idx, y_idx, x_idx]
    for v_vol, v_m in [(g1, g1_m), (z_loc1_vol, zl1_m)]:
        for d_mm in [18.0, 45.0]:
            mn_x, mp_x, mxn_x, mxp_x = [
                a[z_idx, y_idx, x_idx] for a in dir_min_max_1d(v_vol, 2, d_mm, sp_s[2])
            ]
            mn_z, mp_z, mxn_z, mxp_z = [
                a[z_idx, y_idx, x_idx] for a in dir_min_max_1d(v_vol, 0, d_mm, sp_s[0])
            ]
            mn_y, mp_y, _, _ = [
                a[z_idx, y_idx, x_idx] for a in dir_min_max_1d(v_vol, 1, d_mm, sp_s[1])
            ]
            encl_x = v_m - np.maximum(mn_x, mp_x)
            encl_z = v_m - np.maximum(mn_z, mp_z)
            encl_xz = v_m - 0.5 * (np.maximum(mn_x, mp_x) + np.maximum(mn_z, mp_z))
            cont_cols.extend([
                mn_x, mp_x, mxn_x, mxp_x, mn_z, mp_z, mxn_z, mxp_z, mn_y, mp_y, encl_x, encl_z, encl_xz
            ])

    dx_steps = [4.0, 8.0, 14.0, 22.0, 32.0, 44.0]
    dz_steps = [4.0, 9.0, 16.0, 25.0, 36.0]
    dy_steps = [5.0, 12.0, 22.0, 35.0]
    rx_neg = np.column_stack([
        g2[z_idx, y_idx, np.clip(x_idx - int(round(d / sp_s[2])), 0, X - 1)] for d in dx_steps
    ])
    rx_pos = np.column_stack([
        g2[z_idx, y_idx, np.clip(x_idx + int(round(d / sp_s[2])), 0, X - 1)] for d in dx_steps
    ])
    rz_neg = np.column_stack([
        g2[np.clip(z_idx - int(round(d / sp_s[0])), 0, Z - 1), y_idx, x_idx] for d in dz_steps
    ])
    rz_pos = np.column_stack([
        g2[np.clip(z_idx + int(round(d / sp_s[0])), 0, Z - 1), y_idx, x_idx] for d in dz_steps
    ])
    ry_neg = np.column_stack([
        g2[z_idx, np.clip(y_idx - int(round(d / sp_s[1])), 0, Y - 1), x_idx] for d in dy_steps
    ])
    ry_pos = np.column_stack([
        g2[z_idx, np.clip(y_idx + int(round(d / sp_s[1])), 0, Y - 1), x_idx] for d in dy_steps
    ])

    g2_m = g2[z_idx, y_idx, x_idx]
    zn = z_idx.astype(np.float32) / Z
    yn = y_idx.astype(np.float32) / Y
    xn = x_idx.astype(np.float32) / X
    cols = [
        base_val, g1_m, g2_m, g_dict[4.0][z_idx, y_idx, x_idx],
        g_dict[8.0][z_idx, y_idx, x_idx], g_dict[14.0][z_idx, y_idx, x_idx], g24[z_idx, y_idx, x_idx],
        base_val - g_dict[4.0][z_idx, y_idx, x_idx], base_val - g_dict[8.0][z_idx, y_idx, x_idx],
        g1_m - g2_m, g2_m - g_dict[4.0][z_idx, y_idx, x_idx],
        g_dict[4.0][z_idx, y_idx, x_idx] - g_dict[8.0][z_idx, y_idx, x_idx],
        grad_dict[1.0], grad_dict[2.0], grad_dict[4.0],
        z_loc0, zl1_m, z_loc2, z_axial, zn, yn, xn, np.abs(zn - 0.5),
        jz, jy, jx, np.abs(jz), np.sqrt(jy**2 + jx**2),
        pf_fov, pt_fov, pf_j, pt_j, pf_s, pt_s, p_fb, p_tb, log_ft,
    ] + cont_cols

    rng = np.random.RandomState(42)
    offsets = np.vstack([
        rng.uniform(-14.0, 14.0, size=(10, 3)),
        rng.uniform(-36.0, 36.0, size=(10, 3)),
    ])
    rsid = _compute_shifted_diffs_3d(img_norm, z_idx, y_idx, x_idx, offsets, sp_s)

    df_m = df_s[z_idx, y_idx, x_idx]
    dt_m = dt_s[z_idx, y_idx, x_idx]
    df_feat = np.clip(df_m, -7.0, 10.0)
    dt_feat = np.clip(dt_m, -7.0, 10.0)

    return np.hstack([
        np.column_stack(cols),
        rx_neg, rx_pos, rz_neg, rz_pos, ry_neg, ry_pos,
        rsid,
        np.column_stack([df_feat, dt_feat, df_feat - dt_feat]),
    ]).astype(np.float32)


def postprocess_bone_with_shaft_fill(mask, sp, c_j, span_z, is_femur=True):
    cc = _postprocess_bone_mask(mask)
    if not np.any(cc):
        return cc
    inds = np.argwhere(cc)
    z0, y0, x0 = np.maximum(inds.min(axis=0) - 1, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + 2, cc.shape)
    crop = cc[z0:z1, y0:y1, x0:x1].copy()
    for y_loc in range(crop.shape[1]):
        y_abs = y0 + y_loc
        jy = (y_abs * sp[1] - c_j[1]) / span_z
        if (is_femur and jy <= -0.16) or ((not is_femur) and jy >= 0.14):
            crop[:, y_loc, :] = binary_fill_holes(crop[:, y_loc, :])
    for z_loc in range(crop.shape[0]):
        crop[z_loc] = binary_fill_holes(crop[z_loc])
    crop = binary_fill_holes(crop)
    out = np.zeros_like(cc, dtype=bool)
    out[z0:z1, y0:y1, x0:x1] = crop
    return out


def predict_tierA_probs(image, spacing, tierA_or_hgb1, hgb_2=None, atl_full=None, bm_orig=None):
    """Run Tier-A global bone spatial-atlas + cortical enclosure prediction on a single volume."""
    img = np.asarray(image, dtype=np.float32)
    sp = tuple(float(s) for s in spacing)
    img_s = img[::1, ::2, ::2].astype(np.float32)
    sp_s = (sp[0], sp[1] * 2.0, sp[2] * 2.0)
    Z, Y, X = img_s.shape
    init_models = None
    if isinstance(tierA_or_hgb1, dict):
        atl = tierA_or_hgb1['atl_full']
        models_list = [tierA_or_hgb1['hgb_1'], tierA_or_hgb1['hgb_2']]
        if 'hgb_init_1' in tierA_or_hgb1 and 'hgb_init_2' in tierA_or_hgb1:
            init_models = [tierA_or_hgb1['hgb_init_1'], tierA_or_hgb1['hgb_init_2']]
        if hgb_2 is not None and isinstance(hgb_2, dict):
            bm_orig = hgb_2
    else:
        atl = atl_full
        models_list = [tierA_or_hgb1, hgb_2]
    zf = [img.shape[d] / img_s.shape[d] for d in range(3)]

    if bm_orig is None:
        bz1 = np.clip((np.arange(Z) / Z * 20).astype(int), 0, 19)[:, None, None]
        by1 = np.clip((np.arange(Y) / Y * 32).astype(int), 0, 31)[None, :, None]
        bx1 = np.clip((np.arange(X) / X * 32).astype(int), 0, 31)[None, None, :]
        f1 = (bz1 * 1024 + by1 * 32 + bx1).ravel()
        fb_0 = _postprocess_bone_mask(atl[0][f1].reshape(Z, Y, X) >= 0.35)
        tb_0 = _postprocess_bone_mask(atl[1][f1].reshape(Z, Y, X) >= 0.35)
        df_0, dt_0, c_j0, span_z0 = compute_joint_frame_strided(fb_0, tb_0, sp_s)
        _, f2_0, fs_0 = get_joint_bins(img_s.shape, sp_s, c_j0, span_z0)
        p_atl0 = np.maximum(atl[2][f2_0] + atl[3][f2_0], atl[4][fs_0] + atl[5][fs_0]).reshape(img_s.shape)
        if init_models is not None:
            cand0 = (df_0 <= 18.0) | (dt_0 <= 18.0) | (p_atl0 >= 0.005)
            if not np.any(cand0):
                cand0 = np.ones(img_s.shape, dtype=bool)
            X_0 = extract_tierA_features(img_s, sp_s, df_0, dt_0, c_j0, span_z0, cand0, atl, rng_jit=None)[:, :-3]
            p_0 = sum(m.predict_proba(X_0) for m in init_models) / len(init_models)
        else:
            cand0 = (df_0 <= 12.0) | (dt_0 <= 12.0) | (p_atl0 >= 0.02)
            if not np.any(cand0):
                cand0 = np.ones(img_s.shape, dtype=bool)
            X_0 = extract_tierA_features(img_s, sp_s, df_0, dt_0, c_j0, span_z0, cand0, atl, rng_jit=None)
            p_0 = sum(m.predict_proba(X_0) for m in models_list) / len(models_list)
        v_0 = np.zeros((*img_s.shape, 3), dtype=np.float32)
        v_0[..., 0] = 1.0
        v_0[cand0] = p_0
        for c in range(3):
            v_0[..., c] = gaussian_filter(v_0[..., c], sigma=[1.0 / s for s in sp_s])
        fb_s0 = _postprocess_bone_mask((v_0[..., 1] >= 0.45) & (v_0[..., 1] >= v_0[..., 2]))
        tb_s0 = _postprocess_bone_mask((v_0[..., 2] >= 0.45) & (v_0[..., 2] > v_0[..., 1]))
        if np.any(fb_s0) and np.any(tb_s0):
            _, _, c_j0, span_z0 = compute_joint_frame_strided(fb_s0, tb_s0, sp_s)
        pf_0_up = zoom(v_0[..., 1], zf, order=1)
        pt_0_up = zoom(v_0[..., 2], zf, order=1)
        bm_orig = {
            'femur': postprocess_bone_with_shaft_fill(
                (pf_0_up >= 0.47) & (pf_0_up >= pt_0_up), sp, c_j0, span_z0, is_femur=True
            ).astype(np.uint8),
            'tibia': postprocess_bone_with_shaft_fill(
                (pt_0_up >= 0.47) & (pt_0_up > pf_0_up), sp, c_j0, span_z0, is_femur=False
            ).astype(np.uint8),
        }

    fb_s = bm_orig['femur'][::1, ::2, ::2] > 0
    tb_s = bm_orig['tibia'][::1, ::2, ::2] > 0
    df_s, dt_s, c_j, span_z = compute_joint_frame_strided(fb_s, tb_s, sp_s)
    _, f2, fs = get_joint_bins(img_s.shape, sp_s, c_j, span_z)
    p_atl = np.maximum(atl[2][f2] + atl[3][f2], atl[4][fs] + atl[5][fs]).reshape(img_s.shape)
    cand = (df_s <= 10.0) | (dt_s <= 10.0) | (p_atl >= 0.02)
    if not np.any(cand):
        cand = np.ones(img_s.shape, dtype=bool)
    X_c = extract_tierA_features(img_s, sp_s, df_s, dt_s, c_j, span_z, cand, atl, rng_jit=None)
    p_pred = sum(m.predict_proba(X_c) for m in models_list) / len(models_list)

    pf_raw_s = 1.0 / (1.0 + np.exp(np.clip(df_s[cand] / 1.5, -15.0, 15.0)))
    pt_raw_s = 1.0 / (1.0 + np.exp(np.clip(dt_s[cand] / 1.5, -15.0, 15.0)))
    p0_raw_s = np.clip(1.0 - pf_raw_s - pt_raw_s, 0.0, 1.0)
    pr_raw_s = np.column_stack([p0_raw_s, pf_raw_s, pt_raw_s])

    disagree = np.max(np.abs(p_pred - pr_raw_s), axis=1, keepdims=True)
    w_raw = 0.15 * np.clip(1.0 - disagree / 0.45, 0.0, 1.0)
    pb_s = (1.0 - w_raw) * p_pred + w_raw * pr_raw_s

    vol_s = np.zeros((*img_s.shape, 3), dtype=np.float32)
    vol_s[..., 0] = 1.0
    vol_s[cand] = pb_s
    for c in range(3):
        vol_s[..., c] = gaussian_filter(vol_s[..., c], sigma=[1.0 / s for s in sp_s])

    pf_up = zoom(vol_s[..., 1], zf, order=1)
    pt_up = zoom(vol_s[..., 2], zf, order=1)

    df_orig = fast_signed_distance_transform(bm_orig['femur'] > 0, sp)
    dt_orig = fast_signed_distance_transform(bm_orig['tibia'] > 0, sp)
    pf_orig = 1.0 / (1.0 + np.exp(np.clip(df_orig / 0.9, -15.0, 15.0)))
    pt_orig = 1.0 / (1.0 + np.exp(np.clip(dt_orig / 0.9, -15.0, 15.0)))

    wf_b = 0.25 * np.clip(1.0 - np.abs(pf_up - pf_orig) / 0.38, 0.0, 1.0) * (np.abs(df_orig) <= 2.5)
    wt_b = 0.25 * np.clip(1.0 - np.abs(pt_up - pt_orig) / 0.38, 0.0, 1.0) * (np.abs(dt_orig) <= 2.5)
    pf_fin = ((1.0 - wf_b) * pf_up + wf_b * pf_orig).astype(np.float32)
    pt_fin = ((1.0 - wt_b) * pt_up + wt_b * pt_orig).astype(np.float32)
    return {'femur': pf_fin, 'tibia': pt_fin, 'c_j': c_j, 'span_z': span_z}


def predict_bones_from_probs(p_a, spacing):
    """Convert Tier-A bone probability dict into postprocessed binary femur/tibia masks."""
    sp = tuple(float(s) for s in spacing)
    pf_fin = p_a['femur']
    pt_fin = p_a['tibia']
    Z, Y, X = pf_fin.shape
    c_j = p_a.get('c_j')
    span_z = p_a.get('span_z')
    if c_j is None or span_z is None:
        sp_s = (sp[0], sp[1] * 2.0, sp[2] * 2.0)
        fb_s = _postprocess_bone_mask((pf_fin[::1, ::2, ::2] >= 0.47) & (pf_fin[::1, ::2, ::2] >= pt_fin[::1, ::2, ::2]))
        tb_s = _postprocess_bone_mask((pt_fin[::1, ::2, ::2] >= 0.47) & (pt_fin[::1, ::2, ::2] > pf_fin[::1, ::2, ::2]))
        if np.any(fb_s) and np.any(tb_s):
            _, _, c_j, span_z = compute_joint_frame_strided(fb_s, tb_s, sp_s)
        else:
            c_j = np.array([0.5 * Z * sp[0], 0.55 * Y * sp[1], 0.5 * X * sp[2]], dtype=np.float32)
            span_z = 65.0
    span_z = float(span_z)
    fb_mask = postprocess_bone_with_shaft_fill(
        (pf_fin >= 0.47) & (pf_fin >= pt_fin), sp, c_j, span_z, is_femur=True
    )
    tb_mask = postprocess_bone_with_shaft_fill(
        (pt_fin >= 0.47) & (pt_fin > pf_fin), sp, c_j, span_z, is_femur=False
    )
    return {'femur': fb_mask.astype(np.uint8), 'tibia': tb_mask.astype(np.uint8)}


# ==============================================================================
# TIER B: Full-Resolution Articular Subchondral Refiner (RefBone V1 & V2)
# ==============================================================================


def detect_medial_is_low_z(bm, sp):
    fb = bm['femur'] > 0
    tb = bm['tibia'] > 0
    if np.sum(fb) < 20 or np.sum(tb) < 20:
        return True
    zf, yf, _ = np.where(fb)
    zt, yt, xt = np.where(tb)
    y_top_f = np.percentile(yf, 15)
    y_bot_f = np.percentile(yf, 85)
    f_valg = (np.mean(zf[yf <= y_top_f]) - np.mean(zf[yf >= y_bot_f])) * sp[0]

    z_mid = 0.5 * (zt.min() + zt.max())
    tp_top = yt <= np.percentile(yt, 20)
    xt1 = xt[tp_top & (zt < z_mid)]
    xt2 = xt[tp_top & (zt >= z_mid)]
    if len(xt1) < 5 or len(xt2) < 5:
        return bool(f_valg >= 0.0)
    tb_dx = (
        (np.percentile(xt1, 95) - np.percentile(xt1, 5))
        - (np.percentile(xt2, 95) - np.percentile(xt2, 5))
    ) * sp[2]
    return bool((f_valg + 0.5 * tb_dx) >= 0.0)


def get_robust_bone_frame(bm_self, d_other, max_other, sp, med_low_z=True):
    sp_arr = np.array(sp, dtype=np.float32)
    art = (bm_self > 0) & (d_other <= max_other)
    if np.sum(art) < 100:
        art = bm_self > 0
    pts = np.argwhere(art).astype(np.float32) * sp_arr
    q10 = np.percentile(pts, 10, axis=0)
    q50 = np.percentile(pts, 50, axis=0)
    q90 = np.percentile(pts, 90, axis=0)
    scale = np.maximum(q90 - q10, 5.0)
    return q50, scale, med_low_z


def coords_to_bone_atlas_bins(coords_mm, q50, scale, med_low_z=True):
    rel = (coords_mm - q50) / scale
    uz_sym = np.abs(rel[:, 0])
    uz_ml = rel[:, 0] if med_low_z else -rel[:, 0]
    uy = rel[:, 1]
    ux = rel[:, 2]
    bz_sym = np.clip((uz_sym / 0.90 * 24).astype(int), 0, 23)
    bz_ml = np.clip(((uz_ml + 0.90) / 1.80 * 28).astype(int), 0, 27)
    by = np.clip(((uy + 0.90) / 1.80 * 32).astype(int), 0, 31)
    bx = np.clip(((ux + 0.90) / 1.80 * 24).astype(int), 0, 23)
    return (bz_sym, by, bx), (bz_ml, by, bx)


def extract_ray_feats_v2(img, sp, bm_pr, bone_name, bone_atlas_p, rng_jit=None):
    from .rf_seg import _compute_articular_landmarks_and_stats

    df_pr = fast_signed_distance_transform(bm_pr['femur'] > 0, sp)
    dt_pr = fast_signed_distance_transform(bm_pr['tibia'] > 0, sp)
    d_self = df_pr if bone_name == 'femur' else dt_pr
    d_other = dt_pr if bone_name == 'femur' else df_pr

    d_norm_ref = gaussian_filter(d_self, sigma=[2.5 / s for s in sp])
    nz, ny, nx = np.gradient(d_norm_ref, *sp)
    norm = np.sqrt(nz**2 + ny**2 + nx**2) + 1e-6
    nz, ny, nx = (nz / norm).astype(np.float32), (ny / norm).astype(np.float32), (nx / norm).astype(np.float32)

    max_other = 45.0 if bone_name == 'femur' else 22.0
    shell = (np.abs(d_self) <= 3.0) & (d_other <= max_other) & (d_other >= -2.5)
    if not np.any(shell):
        return shell, d_self, np.zeros((0, 108), dtype=np.float32), np.zeros((0, 156), dtype=np.float32)
    inds = np.argwhere(shell)
    z_idx, y_idx, x_idx = inds[:, 0], inds[:, 1], inds[:, 2]
    nz_m, ny_m, nx_m = nz[z_idx, y_idx, x_idx], ny[z_idx, y_idx, x_idx], nx[z_idx, y_idx, x_idx]

    img_f32 = img.astype(np.float32)
    img_f = gaussian_filter(img_f32, sigma=[0.4 / s for s in sp])
    img_lat = gaussian_filter(img_f32, sigma=[1.2 / sp[0], 0.35 / sp[1], 1.2 / sp[2]])
    img_s1 = gaussian_filter(img_f32, sigma=[1.0 / s for s in sp])
    img_s2 = gaussian_filter(img_f32, sigma=[2.0 / s for s in sp])

    z0, y0, x0 = np.maximum(inds.min(axis=0) - 6, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + 7, img.shape)
    ds_c = d_self[z0:z1, y0:y1, x0:x1]
    im_c = img_lat[z0:z1, y0:y1, x0:x1]
    sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
    w_bn = ((ds_c <= -1.4) & (ds_c >= -5.5)).astype(np.float32) + 1e-5
    w_bn_s = gaussian_filter(w_bn, sigma=sig_bn)
    i_bn_mean = gaussian_filter(im_c * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_var = gaussian_filter(((im_c - i_bn_mean) ** 2) * w_bn, sigma=sig_bn) / w_bn_s

    roi_vals = img_f[shell]
    p10, p50, p90 = np.percentile(roi_vals, [10, 50, 90])
    iqr = max(float(p90 - p10), 1.0)
    i_bn_std = np.maximum(np.sqrt(np.maximum(i_bn_var, 0.0)), max(0.04 * iqr, 2.0))
    z_bone_crop = (im_c - i_bn_mean) / i_bn_std

    z_max, y_max, x_max = img.shape
    cz_max, cy_max, cx_max = z_bone_crop.shape
    zc, yc, xc = z_idx - z0, y_idx - y0, x_idx - x0

    d_grid = np.linspace(-4.5, 4.5, 19, dtype=np.float32)
    N = len(z_idx)
    ray_v0 = np.empty((N, len(d_grid)), dtype=np.float32)
    ray_v1 = np.empty((N, len(d_grid)), dtype=np.float32)
    ray_zb = np.empty((N, len(d_grid)), dtype=np.float32)
    for k, d_val in enumerate(d_grid):
        sz = np.clip(np.round(z_idx + d_val * nz_m / sp[0]).astype(int), 0, z_max - 1)
        sy = np.clip(np.round(y_idx + d_val * ny_m / sp[1]).astype(int), 0, y_max - 1)
        sx = np.clip(np.round(x_idx + d_val * nx_m / sp[2]).astype(int), 0, x_max - 1)
        ray_v0[:, k] = img_f[sz, sy, sx]
        ray_v1[:, k] = img_s1[sz, sy, sx]

        szc = np.clip(sz - z0, 0, cz_max - 1)
        syc = np.clip(sy - y0, 0, cy_max - 1)
        sxc = np.clip(sx - x0, 0, cx_max - 1)
        ray_zb[:, k] = z_bone_crop[szc, syc, sxc]

    i_in = np.percentile(ray_v1[:, 0:7], 20, axis=1, keepdims=True)
    i_out = np.percentile(ray_v1[:, 6:17], 85, axis=1, keepdims=True)
    i_span = np.maximum(i_out - i_in, max(0.08 * iqr, 3.0))

    rn0 = (ray_v0 - i_in) / i_span
    rn1 = (ray_v1 - i_in) / i_span
    rg0 = np.diff(rn0, axis=1)
    rg1 = np.diff(rn1, axis=1)

    idx_peak_rise = (np.argmax(rg0[:, 5:13], axis=1).astype(np.float32) - 3.5) * 0.5
    val_peak_rise = np.max(rg0[:, 5:13], axis=1)
    idx_cort_min = (np.argmin(rn0[:, 5:12], axis=1).astype(np.float32) - 4.0) * 0.5
    val_cort_min = np.min(rn0[:, 5:12], axis=1)

    dy_steps = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    vy_lat = np.empty((N, 9), dtype=np.float32)
    vy_zb = np.empty((N, 9), dtype=np.float32)
    for k, dy in enumerate(dy_steps):
        syc = np.clip(yc + dy, 0, cy_max - 1)
        vy_lat[:, k] = (im_c[zc, syc, xc] - i_in[:, 0]) / i_span[:, 0]
        vy_zb[:, k] = z_bone_crop[zc, syc, xc]
    vyg_lat = np.diff(vy_lat, axis=1)

    rv0_iqr = (ray_v0[:, 4:15] - p50) / iqr
    rv1_iqr = (ray_v1[:, 4:15] - p50) / iqr
    v_s2 = (img_s2[z_idx, y_idx, x_idx] - p50) / iqr

    _, _, comp = _compute_articular_landmarks_and_stats(bm_pr[bone_name], d_self, d_other, max_other, sp)
    c1, s1, c2, s2, _, z_mid = comp
    coords_mm = inds.astype(np.float32) * np.array(sp, dtype=np.float32)
    in_1 = coords_mm[:, 0] < z_mid
    uz = np.where(in_1, (c1[0] - coords_mm[:, 0]) / s1[0], (coords_mm[:, 0] - c2[0]) / s2[0])
    uy = np.where(in_1, (coords_mm[:, 1] - c1[1]) / s1[1], (coords_mm[:, 1] - c2[1]) / s2[1])
    ux = np.where(in_1, (coords_mm[:, 2] - c1[2]) / s1[2], (coords_mm[:, 2] - c2[2]) / s2[2])

    max_other_frame = 45.0 if bone_name == 'femur' else 25.0
    med_low_z = detect_medial_is_low_z(bm_pr, sp)
    q50, scale, _ = get_robust_bone_frame(bm_pr[bone_name], d_other, max_other_frame, sp, med_low_z)
    bins_s, bins_m = coords_to_bone_atlas_bins(coords_mm, q50, scale, med_low_z)
    p_atl_s = bone_atlas_p[0][bins_s]
    p_atl_m = bone_atlas_p[1][bins_m]
    rel_f = (coords_mm - q50) / scale
    uz_ml = rel_f[:, 0] if med_low_z else -rel_f[:, 0]

    d_self_feat = d_self[z_idx, y_idx, x_idx]
    d_other_m = d_other[z_idx, y_idx, x_idx]

    X_v1 = np.hstack([
        np.column_stack([
            uz, uy, ux, np.abs(rel_f[:, 0]), rel_f[:, 1], rel_f[:, 2],
            p_atl_s, nz_m, ny_m, nx_m, v_s2, (i_out[:, 0] - i_in[:, 0]) / iqr,
        ]),
        rn0, rn1, rg0, rg1, rv0_iqr, rv1_iqr,
    ]).astype(np.float32)

    X_v2 = np.hstack([
        X_v1,
        np.column_stack([
            p_atl_m, uz_ml, d_self_feat, d_other_m,
            idx_peak_rise, val_peak_rise, idx_cort_min, val_cort_min,
            z_bone_crop[zc, yc, xc],
        ]),
        ray_zb[:, 3:16], vy_lat, vyg_lat, vy_zb,
    ]).astype(np.float32)
    return shell, d_self, X_v1, X_v2


def _refine_one_bone_v2(img, sp, bm_a, bone_name, m_d, return_v1=False, return_prob=False):
    if np.sum(bm_a['femur'] > 0) < 20 or np.sum(bm_a['tibia'] > 0) < 20:
        m0 = (bm_a[bone_name] > 0).astype(np.uint8)
        p0 = m0.astype(np.float32)
        if return_v1 and return_prob:
            return m0, m0, p0
        if return_v1:
            return m0, m0
        if return_prob:
            return m0, p0
        return m0
    shell, d_self, X1, X2 = extract_ray_feats_v2(
        img, sp, bm_a, bone_name, m_d['p_full'], rng_jit=None
    )
    if not np.any(shell):
        m0 = (bm_a[bone_name] > 0).astype(np.uint8)
        p0 = m0.astype(np.float32)
        if return_v1 and return_prob:
            return m0, m0, p0
        if return_v1:
            return m0, m0
        if return_prob:
            return m0, p0
        return m0
    with joblib.parallel_backend('threading'):
        p1 = m_d['et_v1'].predict_proba(X1)[:, 1]
        p_et2 = m_d['et_v2'].predict_proba(X2)[:, 1]
    p2 = (
        0.40 * m_d['hgb_v2_1'].predict_proba(X2)[:, 1]
        + 0.35 * m_d['hgb_v2_2'].predict_proba(X2)[:, 1]
        + 0.25 * p_et2
    )
    p_bl = 0.35 * p1 + 0.65 * p2
    joint_clf = m_d.get('joint_clf')
    joint_alpha = float(m_d.get('joint_alpha', 0.0))
    if joint_clf is not None and joint_alpha > 0.0:
        p_jb = joint_clf.predict_proba(X1)[:, 1]
        p_bl = (1.0 - joint_alpha) * p_bl + joint_alpha * p_jb
    p_orig = 1.0 / (1.0 + np.exp(np.clip(d_self / 0.4, -20.0, 20.0)))
    sig_s = [0.8 / sp[0], 0.35 / sp[1], 0.6 / sp[2]]
    volb = p_orig.copy()
    volb[shell] = p_bl
    volb_s = gaussian_filter(volb, sigma=sig_s)
    m_v2 = _postprocess_bone_mask(volb_s >= 0.50).astype(np.uint8)
    prob_v2 = np.clip(volb_s, 0.0, 1.0).astype(np.float32)
    if not return_v1:
        return (m_v2, prob_v2) if return_prob else m_v2
    volb1 = p_orig.copy()
    volb1[shell] = p1
    m_v1 = _postprocess_bone_mask(
        gaussian_filter(volb1, sigma=sig_s) >= 0.50
    ).astype(np.uint8)
    return (m_v2, m_v1, prob_v2) if return_prob else (m_v2, m_v1)


def refine_bone_v2_single(
    image, spacing, bm_self_or_dict, bm_other_or_models,
    bone_name=None, ref_v2_bone=None, return_v1=False, return_prob=False,
):
    """Run Tier-B V2 articular subchondral refiner on a single volume given Tier-A bone masks."""
    img = np.asarray(image, dtype=np.float32)
    sp = tuple(float(s) for s in spacing)
    if bone_name is not None:
        other_name = 'femur' if bone_name == 'tibia' else 'tibia'
        bm_a = {bone_name: bm_self_or_dict, other_name: bm_other_or_models}
        return _refine_one_bone_v2(
            img, sp, bm_a, bone_name, ref_v2_bone, return_v1=return_v1, return_prob=return_prob
        )
    bm_a = bm_self_or_dict
    v2_models = bm_other_or_models
    if not return_v1 and not return_prob:
        return {
            b: _refine_one_bone_v2(img, sp, bm_a, b, v2_models[b], return_v1=False)
            for b in ['femur', 'tibia']
        }
    res = {
        b: _refine_one_bone_v2(
            img, sp, bm_a, b, v2_models[b], return_v1=return_v1, return_prob=return_prob
        )
        for b in ['femur', 'tibia']
    }
    if return_v1 and return_prob:
        return (
            {b: res[b][0] for b in ['femur', 'tibia']},
            {b: res[b][1] for b in ['femur', 'tibia']},
            {b: res[b][2] for b in ['femur', 'tibia']},
        )
    return (
        {b: res[b][0] for b in ['femur', 'tibia']},
        {b: res[b][1] for b in ['femur', 'tibia']},
    )

