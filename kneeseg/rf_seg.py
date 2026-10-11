import os

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter, label
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesClassifier

from .features import compute_signed_distance_transforms, extract_features, fast_signed_distance_transform

# Shared in-memory cache during a training session to pass LOO atlases & OOF P1 probabilities
_SESSION_ATLAS_CACHE = {}
_SESSION_OOF_PROB_CACHE = {}


def _infer_default_spacing(shape):
    if shape[1] >= 1.8 * shape[0]:
        return (1.2, 0.4, 0.4)
    return (1.0, 1.0, 1.0)


def _vol_fingerprint(img):
    flat = img.ravel()
    step = max(1, len(flat) // 64)
    return (img.shape, float(flat[::step].sum()), float(flat[:100].mean()))


def _get_candidate_roi(df, dt):
    return (df > -0.8) & (dt > -0.8) & (((df <= 5.5) & (dt <= 55.0)) | ((dt <= 5.5) & (df <= 25.0)))


def _get_tibial_roi(df, dt):
    return (dt > -0.8) & (dt <= 5.5) & (df > -0.8) & (df <= 20.0)


def _compute_articular_landmarks_and_stats(bone_mask, sdt_self, sdt_other, max_other_dist, spacing):
    sp_arr = np.array(spacing, dtype=np.float32)
    art_mask = (bone_mask > 0) & (sdt_other <= max_other_dist)
    if np.sum(art_mask) < 50:
        art_mask = bone_mask > 0
    if not np.any(art_mask):
        art_pts = np.zeros((10, 3), dtype=np.float32)
    else:
        art_pts = np.argwhere(art_mask).astype(np.float32) * sp_arr
    c_min = art_pts.min(axis=0)
    c_max = art_pts.max(axis=0)
    c_mean = art_pts.mean(axis=0)

    surf_mask = art_mask & (sdt_self >= -1.5)
    if np.sum(surf_mask) < 20:
        surf_mask = art_mask
    surf_pts = np.argwhere(surf_mask).astype(np.float32) * sp_arr if np.any(surf_mask) else art_pts

    z_mid = np.median(surf_pts[:, 0])
    comp1 = surf_pts[surf_pts[:, 0] < z_mid]
    comp2 = surf_pts[surf_pts[:, 0] >= z_mid]
    if len(comp1) < 10:
        comp1 = surf_pts
    if len(comp2) < 10:
        comp2 = surf_pts
    c1, s1 = comp1.mean(axis=0), comp1.std(axis=0) + 1.5
    c2, s2 = comp2.mean(axis=0), comp2.std(axis=0) + 1.5
    spine = 0.5 * (c1 + c2)

    lms1, lms2 = [], []
    z_q1 = np.quantile(comp1[:, 0], [0.80, 0.50, 0.20])
    x_q1 = np.quantile(comp1[:, 2], [0.20, 0.50, 0.80])
    for zc in z_q1:
        for xc in x_q1:
            d2 = (comp1[:, 0] - zc) ** 2 + (comp1[:, 2] - xc) ** 2
            lms1.append(comp1[np.argmin(d2)])

    z_q2 = np.quantile(comp2[:, 0], [0.20, 0.50, 0.80])
    x_q2 = np.quantile(comp2[:, 2], [0.20, 0.50, 0.80])
    for zc in z_q2:
        for xc in x_q2:
            d2 = (comp2[:, 0] - zc) ** 2 + (comp2[:, 2] - xc) ** 2
            lms2.append(comp2[np.argmin(d2)])

    return (
        (c_min, c_max, c_mean),
        (np.array(lms1, dtype=np.float32), np.array(lms2, dtype=np.float32)),
        (c1, s1, c2, s2, spine, z_mid),
    )


def _compute_atlas_bins_from_stats(coords_mm, df_m, dt_m, f_stats, t_comp):
    c1_t, s1_t, c2_t, s2_t, _, z_mid_t = t_comp
    in_1t = coords_mm[:, 0] < z_mid_t
    uz_t = np.where(in_1t, (c1_t[0] - coords_mm[:, 0]) / s1_t[0], (coords_mm[:, 0] - c2_t[0]) / s2_t[0])
    ux_t = np.where(in_1t, (coords_mm[:, 2] - c1_t[2]) / s1_t[2], (coords_mm[:, 2] - c2_t[2]) / s2_t[2])

    bz_t = np.clip(((uz_t + 2.5) / 5.0 * 24).astype(int), 0, 23)
    bx_t = np.clip(((ux_t + 2.5) / 5.0 * 24).astype(int), 0, 23)
    bdt_t = np.clip((dt_m / 5.5 * 16).astype(int), 0, 15)
    bdf_t = np.clip((df_m / 18.0 * 16).astype(int), 0, 15)

    f_min, f_max, _ = f_stats
    rel_f = (coords_mm - f_min) / (f_max - f_min + 1e-6)
    z_sym_f = np.abs(rel_f[:, 0] - 0.5) * 2.0
    bz_f = np.clip((z_sym_f * 20).astype(int), 0, 19)
    by_f = np.clip((rel_f[:, 1] * 24).astype(int), 0, 23)
    bx_f = np.clip((rel_f[:, 2] * 24).astype(int), 0, 23)
    bdf_f = np.clip((df_m / 5.5 * 16).astype(int), 0, 15)
    return (bz_t, bx_t, bdt_t, bdf_t), (bz_f, by_f, bx_f, bdf_f)


def _make_atlas_priors(hists):
    ht_tot, ht_tc, ht_fc, hf_tot, hf_fc, hf_tc, h2d_base, h2d_tc, h2d_dt_sum = hists
    st_tot = gaussian_filter(ht_tot, sigma=0.8) + 2.0
    pt_tc = gaussian_filter(ht_tc, sigma=0.8) / st_tot
    pt_fc = gaussian_filter(ht_fc, sigma=0.8) / st_tot
    sf_tot = gaussian_filter(hf_tot, sigma=0.8) + 2.0
    pf_fc = gaussian_filter(hf_fc, sigma=0.8) / sf_tot
    pf_tc = gaussian_filter(hf_tc, sigma=0.8) / sf_tot
    p2d_tc = gaussian_filter(h2d_tc, sigma=0.8) / (gaussian_filter(h2d_base, sigma=0.8) + 2.0)
    ht_tc_2d = np.sum(ht_tc, axis=(2, 3))
    edt_2d = gaussian_filter(h2d_dt_sum, sigma=0.8) / (gaussian_filter(ht_tc_2d, sigma=0.8) + 2.0)
    return (pt_tc, pt_fc, pf_fc, pf_tc, p2d_tc, edt_2d)


def _compute_shifted_diffs(vol, z_idx, y_idx, x_idx, offsets_mm, spacing):
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


def _precompute_volume_maps(img, b_masks, dts, spacing, lbl=None):
    df, dt = dts['femur'], dts['tibia']
    roi = _get_candidate_roi(df, dt)
    roi_tib = _get_tibial_roi(df, dt)
    if not np.any(roi):
        roi = np.ones_like(img, dtype=bool)
        roi_tib = roi
    inds = np.argwhere(roi)
    margin = [int(np.ceil(8.0 / s)) for s in spacing]
    z0, y0, x0 = np.maximum(inds.min(axis=0) - margin, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + margin + 1, img.shape)

    roi_vals = img[roi].astype(np.float32)
    p10, p50, p90 = np.percentile(roi_vals, [10, 50, 90])
    iqr = max(float(p90 - p10), 1.0)
    img_norm = ((img.astype(np.float32) - p50) / iqr).astype(np.float32)
    img_crop = img_norm[z0:z1, y0:y1, x0:x1]

    gauss_crops = {}
    grad_crops = {}
    grad_10_vec = None
    for sig_mm in [0.5, 1.0, 2.0, 4.0, 8.0]:
        sig_vox = [sig_mm / s for s in spacing]
        g = gaussian_filter(img_crop, sigma=sig_vox)
        gauss_crops[sig_mm] = g
        if sig_mm in [0.5, 1.0, 2.0]:
            gz, gy, gx = np.gradient(g, *spacing)
            grad_crops[sig_mm] = np.sqrt(gz**2 + gy**2 + gx**2)
            if sig_mm == 1.0:
                grad_10_vec = (gz, gy, gx)

    df_crop = df[z0:z1, y0:y1, x0:x1]
    dt_crop = dt[z0:z1, y0:y1, x0:x1]
    gz_f, gy_f, gx_f = np.gradient(df_crop, *spacing)
    norm_f = np.sqrt(gz_f**2 + gy_f**2 + gx_f**2) + 1e-6
    gz_f, gy_f, gx_f = gz_f / norm_f, gy_f / norm_f, gx_f / norm_f

    gz_t, gy_t, gx_t = np.gradient(dt_crop, *spacing)
    norm_t = np.sqrt(gz_t**2 + gy_t**2 + gx_t**2) + 1e-6
    gz_t, gy_t, gx_t = gz_t / norm_t, gy_t / norm_t, gx_t / norm_t

    f_stats, f_lms, f_comp = _compute_articular_landmarks_and_stats(b_masks['femur'], df, dt, 45.0, spacing)
    t_stats, t_lms, t_comp = _compute_articular_landmarks_and_stats(b_masks['tibia'], dt, df, 25.0, spacing)

    u_tf = f_stats[2] - t_stats[2]
    u_tf = u_tf / (np.linalg.norm(u_tf) + 1e-6)

    atlas_hists = None
    if lbl is not None:
        df_r_all = df[inds[:, 0], inds[:, 1], inds[:, 2]]
        dt_r_all = dt[inds[:, 0], inds[:, 1], inds[:, 2]]
        pos_sdt = (df_r_all >= 0.0) & (dt_r_all >= 0.0)
        inds_atl = inds[pos_sdt]
        coords_roi = inds_atl.astype(np.float32) * np.array(spacing, dtype=np.float32)
        df_r = df_r_all[pos_sdt]
        dt_r = dt_r_all[pos_sdt]
        bins_t, bins_f = _compute_atlas_bins_from_stats(coords_roi, df_r, dt_r, f_stats, t_comp)
        y_r = lbl[inds_atl[:, 0], inds_atl[:, 1], inds_atl[:, 2]]
        ht_tot = np.zeros((24, 24, 16, 16), dtype=np.float32)
        ht_tc = np.zeros((24, 24, 16, 16), dtype=np.float32)
        ht_fc = np.zeros((24, 24, 16, 16), dtype=np.float32)
        np.add.at(ht_tot, bins_t, 1.0)
        np.add.at(ht_tc, tuple(b[y_r == 4] for b in bins_t), 1.0)
        np.add.at(ht_fc, tuple(b[y_r == 2] for b in bins_t), 1.0)

        hf_tot = np.zeros((20, 24, 24, 16), dtype=np.float32)
        hf_fc = np.zeros((20, 24, 24, 16), dtype=np.float32)
        hf_tc = np.zeros((20, 24, 24, 16), dtype=np.float32)
        np.add.at(hf_tot, bins_f, 1.0)
        np.add.at(hf_fc, tuple(b[y_r == 2] for b in bins_f), 1.0)
        np.add.at(hf_tc, tuple(b[y_r == 4] for b in bins_f), 1.0)

        bz_t, bx_t, _, _ = bins_t
        base_m = (dt_r >= 0.0) & (dt_r <= 1.5)
        tc_base_m = base_m & (y_r == 4)
        h2d_base = np.zeros((24, 24), dtype=np.float32)
        h2d_tc = np.zeros((24, 24), dtype=np.float32)
        h2d_dt_sum = np.zeros((24, 24), dtype=np.float32)
        np.add.at(h2d_base, (bz_t[base_m], bx_t[base_m]), 1.0)
        np.add.at(h2d_tc, (bz_t[tc_base_m], bx_t[tc_base_m]), 1.0)
        np.add.at(h2d_dt_sum, (bz_t[y_r == 4], bx_t[y_r == 4]), dt_r[y_r == 4])
        atlas_hists = (ht_tot, ht_tc, ht_fc, hf_tot, hf_fc, hf_tc, h2d_base, h2d_tc, h2d_dt_sum)

    return {
        'bbox': (z0, z1, y0, y1, x0, x1),
        'img_norm': img_norm,
        'gauss_crops': gauss_crops,
        'grad_crops': grad_crops,
        'grad_10_vec': grad_10_vec,
        'normals_f': (gz_f, gy_f, gx_f),
        'normals_t': (gz_t, gy_t, gx_t),
        'bone_stats': {'femur': f_stats, 'tibia': t_stats},
        'comp_stats': {'femur': f_comp, 'tibia': t_comp},
        'u_tf': u_tf,
        'landmarks': {'femur': f_lms, 'tibia': t_lms},
        'roi': roi,
        'roi_tib': roi_tib,
        'atlas_hists': atlas_hists,
    }


def _extract_from_precomputed(pre, dts, spacing, mask_3d, atlas_priors, prob_map=None, sdt_jitter_std=0.0, jitter_seed=42):
    z0, z1, y0, y1, x0, x1 = pre['bbox']
    mask_indices = np.argwhere(mask_3d)
    z_idx, y_idx, x_idx = mask_indices[:, 0], mask_indices[:, 1], mask_indices[:, 2]
    zc, yc, xc = z_idx - z0, y_idx - y0, x_idx - x0

    img_norm = pre['img_norm']
    base_val = img_norm[z_idx, y_idx, x_idx]

    g05 = pre['gauss_crops'][0.5][zc, yc, xc]
    g10 = pre['gauss_crops'][1.0][zc, yc, xc]
    g20 = pre['gauss_crops'][2.0][zc, yc, xc]
    g40 = pre['gauss_crops'][4.0][zc, yc, xc]
    g80 = pre['gauss_crops'][8.0][zc, yc, xc]

    cols = [
        base_val,
        g05,
        g10,
        g20,
        g40,
        g80,
        base_val - g40,
        base_val - g80,
        g05 - g10,
        g10 - g20,
        g20 - g40,
        g40 - g80,
        pre['grad_crops'][0.5][zc, yc, xc],
        pre['grad_crops'][1.0][zc, yc, xc],
        pre['grad_crops'][2.0][zc, yc, xc],
    ]

    df, dt = dts['femur'], dts['tibia']
    df_m = df[z_idx, y_idx, x_idx].copy()
    dt_m = dt[z_idx, y_idx, x_idx].copy()
    if sdt_jitter_std > 0.0:
        rng_j = np.random.RandomState(jitter_seed)
        df_m = df_m + rng_j.normal(0.0, sdt_jitter_std, size=df_m.shape).astype(np.float32)
        dt_m = dt_m + rng_j.normal(0.0, sdt_jitter_std, size=dt_m.shape).astype(np.float32)
    df_pos = np.maximum(df_m, 0.0)
    dt_pos = np.maximum(dt_m, 0.0)
    cols.extend([
        df_m,
        dt_m,
        df_m + dt_m,
        df_m - dt_m,
        np.minimum(df_m, dt_m),
        df_pos / (df_pos + dt_pos + 0.5),
        df_pos / (df_pos + dt_pos + 5.0),
    ])

    nz_f, ny_f, nx_f = (
        pre['normals_f'][0][zc, yc, xc],
        pre['normals_f'][1][zc, yc, xc],
        pre['normals_f'][2][zc, yc, xc],
    )
    nz_t, ny_t, nx_t = (
        pre['normals_t'][0][zc, yc, xc],
        pre['normals_t'][1][zc, yc, xc],
        pre['normals_t'][2][zc, yc, xc],
    )
    u_tf = pre['u_tf']
    n_dot = nz_f * nz_t + ny_f * ny_t + nx_f * nx_t
    nf_utf = nz_f * u_tf[0] + ny_f * u_tf[1] + nx_f * u_tf[2]
    nt_utf = nz_t * u_tf[0] + ny_t * u_tf[1] + nx_t * u_tf[2]
    foot_t_df = df_m - dt_m * n_dot
    foot_f_dt = dt_m - df_m * n_dot
    cols.extend([
        np.abs(nz_f),
        ny_f,
        nx_f,
        np.abs(nz_t),
        ny_t,
        nx_t,
        n_dot,
        nf_utf,
        nt_utf,
        foot_t_df,
        foot_f_dt,
    ])

    igz, igy, igx = (
        pre['grad_10_vec'][0][zc, yc, xc],
        pre['grad_10_vec'][1][zc, yc, xc],
        pre['grad_10_vec'][2][zc, yc, xc],
    )
    cols.append(igz * nz_f + igy * ny_f + igx * nx_f)
    cols.append(igz * nz_t + igy * ny_t + igx * nx_t)

    z_max, y_max, x_max = img_norm.shape
    for step_mm in [-2.5, -1.5, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0]:
        for nz, ny, nx in [(nz_f, ny_f, nx_f), (nz_t, ny_t, nx_t)]:
            sz = np.clip(np.round(z_idx + step_mm * nz / spacing[0]).astype(int), 0, z_max - 1)
            sy = np.clip(np.round(y_idx + step_mm * ny / spacing[1]).astype(int), 0, y_max - 1)
            sx = np.clip(np.round(x_idx + step_mm * nx / spacing[2]).astype(int), 0, x_max - 1)
            val = img_norm[sz, sy, sx]
            cols.append(val)
            cols.append(base_val - val)

    # Bone-surface-anchored ray features along bone normals
    for d_m, (nz, ny, nx) in [(dt_m, (nz_t, ny_t, nx_t)), (df_m, (nz_f, ny_f, nx_f))]:
        step_cortex = -(d_m + 0.6)
        step_base = -(d_m - 0.6)
        step_mid = -0.5 * d_m
        vals_surf = []
        for st_arr in [step_cortex, step_base, step_mid]:
            sz = np.clip(np.round(z_idx + st_arr * nz / spacing[0]).astype(int), 0, z_max - 1)
            sy = np.clip(np.round(y_idx + st_arr * ny / spacing[1]).astype(int), 0, y_max - 1)
            sx = np.clip(np.round(x_idx + st_arr * nx / spacing[2]).astype(int), 0, x_max - 1)
            v = img_norm[sz, sy, sx]
            vals_surf.append(v)
            cols.append(v)
        cols.append(vals_surf[1] - vals_surf[0])
        cols.append(base_val - vals_surf[0])
        cols.append(np.minimum(np.minimum(vals_surf[1], vals_surf[2]), base_val))

    coords_mm = mask_indices.astype(np.float32) * np.array(spacing, dtype=np.float32)

    PT_tc, PT_fc, PF_fc, PF_tc, P2D_tc, EDT_2d = atlas_priors
    bins_t, bins_f = _compute_atlas_bins_from_stats(
        coords_mm, df_m, dt_m, pre['bone_stats']['femur'], pre['comp_stats']['tibia']
    )
    pt_tc_v = PT_tc[bins_t]
    pt_fc_v = PT_fc[bins_t]
    pf_fc_v = PF_fc[bins_f]
    pf_tc_v = PF_tc[bins_f]

    bdt_m1 = np.clip(bins_t[2] - 1, 0, 15)
    bdt_p1 = np.clip(bins_t[2] + 1, 0, 15)
    pt_tc_m1 = PT_tc[bins_t[0], bins_t[1], bdt_m1, bins_t[3]]
    pt_tc_p1 = PT_tc[bins_t[0], bins_t[1], bdt_p1, bins_t[3]]
    pt_tc_max3 = np.maximum(np.maximum(pt_tc_m1, pt_tc_v), pt_tc_p1)
    pt_tc_mean3 = (pt_tc_m1 + pt_tc_v + pt_tc_p1) / 3.0

    bdf_m1 = np.clip(bins_f[3] - 1, 0, 15)
    bdf_p1 = np.clip(bins_f[3] + 1, 0, 15)
    pf_fc_m1 = PF_fc[bins_f[0], bins_f[1], bins_f[2], bdf_m1]
    pf_fc_p1 = PF_fc[bins_f[0], bins_f[1], bins_f[2], bdf_p1]
    pf_fc_max3 = np.maximum(np.maximum(pf_fc_m1, pf_fc_v), pf_fc_p1)
    pf_fc_mean3 = (pf_fc_m1 + pf_fc_v + pf_fc_p1) / 3.0

    p2d_v = P2D_tc[bins_t[0], bins_t[1]]
    edt_v = EDT_2d[bins_t[0], bins_t[1]]
    cols.extend([
        pt_tc_v,
        pt_fc_v,
        pf_fc_v,
        pf_tc_v,
        pt_tc_max3,
        pt_tc_mean3,
        pf_fc_max3,
        pf_fc_mean3,
        pt_tc_v - pt_fc_v,
        pf_fc_v - pf_tc_v,
        np.maximum(pf_fc_v, pt_fc_v) - pt_tc_v,
        p2d_v,
        edt_v,
        dt_m - 2.0 * edt_v,
        dt_m / (2.0 * edt_v + 0.5),
    ])

    lm_blocks = []
    rel_cols = []
    for b_name in ['femur', 'tibia']:
        c_min, c_max, c_mean = pre['bone_stats'][b_name]
        rel_box = (coords_mm - c_min) / (c_max - c_min + 1e-6)
        rel_ctr = (coords_mm - c_mean) / 40.0
        z_sym = np.abs(rel_box[:, 0] - 0.5)
        rel_cols.extend([rel_box, rel_ctr, z_sym[:, None]])

        lms1, lms2 = pre['landmarks'][b_name]
        d_lm1 = cdist(coords_mm, lms1).astype(np.float32)
        d_lm2 = cdist(coords_mm, lms2).astype(np.float32)
        lm_blocks.extend([np.minimum(d_lm1, d_lm2), np.maximum(d_lm1, d_lm2)])

        c1, s1, c2, s2, spine, _ = pre['comp_stats'][b_name]
        d1_zx = np.sqrt(((coords_mm[:, 0] - c1[0]) / s1[0]) ** 2 + ((coords_mm[:, 2] - c1[2]) / s1[2]) ** 2)
        d2_zx = np.sqrt(((coords_mm[:, 0] - c2[0]) / s2[0]) ** 2 + ((coords_mm[:, 2] - c2[2]) / s2[2]) ** 2)
        d1_3d = np.linalg.norm(coords_mm - c1, axis=1)
        d2_3d = np.linalg.norm(coords_mm - c2, axis=1)
        d_spine_z = np.abs(coords_mm[:, 0] - spine[0])
        d_spine_3d = np.linalg.norm(coords_mm - spine, axis=1)
        closer_y = np.where(d1_3d < d2_3d, coords_mm[:, 1] - c1[1], coords_mm[:, 1] - c2[1])
        closer_x = np.where(d1_3d < d2_3d, (coords_mm[:, 2] - c1[2]) / s1[2], (coords_mm[:, 2] - c2[2]) / s2[2])
        closer_z_signed = np.where(
            d1_3d < d2_3d, (c1[0] - coords_mm[:, 0]) / s1[0], (coords_mm[:, 0] - c2[0]) / s2[0]
        )
        closer_z_abs = np.abs(closer_z_signed)
        cols.extend([
            np.minimum(d1_zx, d2_zx),
            np.minimum(d1_3d, d2_3d),
            d_spine_z,
            d_spine_3d,
            closer_y,
            closer_x,
            closer_z_signed,
            closer_z_abs,
        ])

    rng = np.random.RandomState(42)
    offsets_all = np.vstack([
        rng.uniform(-4.5, 4.5, size=(20, 3)),
        rng.uniform(-18.0, 18.0, size=(20, 3)),
    ])
    rsid = _compute_shifted_diffs(img_norm, z_idx, y_idx, x_idx, offsets_all, spacing)

    blocks = [np.column_stack(cols), np.hstack(rel_cols)] + lm_blocks + [rsid]

    if prob_map is not None:
        rng_p = np.random.RandomState(123)
        offsets_p = np.vstack([
            rng_p.uniform(-3.5, 3.5, size=(15, 3)),
            rng_p.uniform(-12.0, 12.0, size=(15, 3)),
        ])
        p_cols = []
        p_blocks = []
        pf = prob_map[..., 1].astype(np.float32)
        pt = prob_map[..., 2].astype(np.float32)
        p_cols.extend([
            pf[z_idx, y_idx, x_idx],
            pt[z_idx, y_idx, x_idx],
            pf[z_idx, y_idx, x_idx] - pt[z_idx, y_idx, x_idx],
            pf[z_idx, y_idx, x_idx] + pt[z_idx, y_idx, x_idx],
        ])
        for p_ch in [pf, pt]:
            p_crop = p_ch[z0:z1, y0:y1, x0:x1]
            for sig_mm in [0.8, 1.8, 3.5]:
                pg = gaussian_filter(p_crop, sigma=[sig_mm / s for s in spacing])
                p_cols.append(pg[zc, yc, xc])
                if sig_mm == 1.8:
                    pgz, pgy, pgx = np.gradient(pg, *spacing)
                    p_cols.append(np.sqrt(pgz**2 + pgy**2 + pgx**2)[zc, yc, xc])
            for step_mm in [-1.5, -0.5, 0.5, 1.5, 2.5]:
                for nz, ny, nx in [(nz_f, ny_f, nx_f), (nz_t, ny_t, nx_t)]:
                    sz = np.clip(np.round(z_idx + step_mm * nz / spacing[0]).astype(int), 0, z_max - 1)
                    sy = np.clip(np.round(y_idx + step_mm * ny / spacing[1]).astype(int), 0, y_max - 1)
                    sx = np.clip(np.round(x_idx + step_mm * nx / spacing[2]).astype(int), 0, x_max - 1)
                    p_cols.append(p_ch[sz, sy, sx])
            for st_arr in [-(dt_m - 0.6), -0.5 * dt_m]:
                sz = np.clip(np.round(z_idx + st_arr * nz_t / spacing[0]).astype(int), 0, z_max - 1)
                sy = np.clip(np.round(y_idx + st_arr * ny_t / spacing[1]).astype(int), 0, y_max - 1)
                sx = np.clip(np.round(x_idx + st_arr * nx_t / spacing[2]).astype(int), 0, x_max - 1)
                p_cols.append(p_ch[sz, sy, sx])
            rspd = _compute_shifted_diffs(p_ch, z_idx, y_idx, x_idx, offsets_p, spacing)
            p_blocks.append(rspd)
        blocks.append(np.column_stack(p_cols))
        blocks.extend(p_blocks)

    return np.hstack(blocks)


def _keep_top_k_components(mask, k=1, min_size=150):
    labeled, n_comp = label(mask)
    if n_comp == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    top_labels = np.argsort(sizes)[::-1][:k]
    out = np.zeros_like(mask, dtype=bool)
    for l_idx in top_labels:
        if sizes[l_idx] >= min_size:
            out |= labeled == l_idx
    return out


class CartilageClassifier:
    def __init__(self, n_estimators=100, max_depth=25, n_jobs=-1, target_bones=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.target_bones = target_bones
        self.stage = 1
        self.clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=n_jobs,
            random_state=42,
        )
        self.clf_tib_spec = None
        self.atlas_priors = None

    def train(
        self,
        images,
        bone_masks,
        labels,
        landmarks_list=None,
        landmark_indices=None,
        prob_maps=None,
        dtype='float32',
        spacings=None,
    ):
        """
        Trains Semantic Context Forest for cartilage segmentation (Pass 1 or Pass 2).
        """
        # Fallback for small synthetic volumes or non-femur/tibia setups
        if (
            len(images) == 0
            or min(images[0].shape) < 32
            or 'femur' not in bone_masks[0]
            or 'tibia' not in bone_masks[0]
        ):
            self.stage = 0
            X_all, y_all = [], []
            for i, (img, b_masks, lbl) in enumerate(zip(images, bone_masks, labels)):
                dts = compute_signed_distance_transforms(b_masks)
                min_dt = np.min([np.abs(d) for d in dts.values()], axis=0)
                mask = min_dt.flatten() < 10
                lms = landmarks_list[i] if landmarks_list else None
                pm = prob_maps[i] if prob_maps else None
                X = extract_features(
                    img,
                    dts,
                    mask=mask,
                    landmarks_dict=lms,
                    landmark_indices=landmark_indices,
                    prob_map=pm,
                    sorted_bones_override=self.target_bones,
                    target_dtype=dtype,
                )
                y = lbl.flatten()[mask]
                if len(y) > 25000:
                    np.random.seed(42)
                    indices = np.random.choice(len(y), 25000, replace=False)
                    X, y = X[indices], y[indices]
                X_all.append(X)
                y_all.append(y)
            self.clf.fit(np.vstack(X_all), np.concatenate(y_all))
            return

        if spacings is None:
            spacings = [_infer_default_spacing(img.shape) for img in images]

        print(f"    Precomputing physical-space SDTs, normal fields, and 4D/2D Atlases for {len(images)} cases...")

        def _prep_case(idx):
            img, b_m, lbl, sp = images[idx], bone_masks[idx], labels[idx], spacings[idx]
            dts = {
                'femur': fast_signed_distance_transform(b_m['femur'], sp),
                'tibia': fast_signed_distance_transform(b_m['tibia'], sp),
            }
            pre = _precompute_volume_maps(img, b_m, dts, sp, lbl=lbl)
            return img, b_m, lbl, sp, dts, pre

        prepped = Parallel(n_jobs=min(16, max(1, len(images))), prefer='threads')(
            delayed(_prep_case)(i) for i in range(len(images))
        )

        tot_hists = [sum(item[5]['atlas_hists'][k] for item in prepped) for k in range(9)]
        self.atlas_priors = _make_atlas_priors(tot_hists)
        train_loo_atlases = []
        for item in prepped:
            ih = item[5]['atlas_hists']
            train_loo_atlases.append(_make_atlas_priors([tot_hists[k] - ih[k] for k in range(9)]))

        def _sample_case(idx, prob_map=None, tib_only=False, seed_offset=0):
            img, b_m, lbl, sp, dts, pre = prepped[idx]
            atlas_p = train_loo_atlases[idx]
            roi = pre['roi_tib'] if tib_only else pre['roi']
            dt = dts['tibia']
            y_clean = np.zeros_like(lbl, dtype=np.uint8)
            y_clean[lbl == 2] = 2
            y_clean[lbl == 4] = 4

            rng = np.random.RandomState(42 + idx + seed_offset)
            if tib_only:
                pos_t = np.argwhere(roi & (y_clean == 4))
                pos_f = np.argwhere(roi & (y_clean == 2))
                neg_b = np.argwhere(roi & (y_clean == 0))
                n_t = min(len(pos_t), 6000)
                idx_t = pos_t[rng.choice(len(pos_t), n_t, replace=False)] if n_t > 0 else pos_t
                n_f = min(len(pos_f), max(1, int(0.55 * max(n_t, 1000))))
                idx_f = pos_f[rng.choice(len(pos_f), n_f, replace=False)] if n_f > 0 else pos_f
                n_nb = min(len(neg_b), max(1, int(1.95 * max(n_t, 1000))))
                idx_nb = neg_b[rng.choice(len(neg_b), n_nb, replace=False)] if n_nb > 0 else neg_b
                arrs = [idx_t, idx_f, idx_nb]
            else:
                pos_f = np.argwhere(roi & (y_clean == 2))
                pos_t = np.argwhere(roi & (y_clean == 4))
                neg_tib = np.argwhere(roi & (y_clean == 0) & (dt <= 5.5))
                neg_fem = np.argwhere(roi & (y_clean == 0) & (dt > 5.5))
                n_f = min(len(pos_f), 6500)
                n_t = min(len(pos_t), 6000)
                idx_f = pos_f[rng.choice(len(pos_f), n_f, replace=False)] if n_f > 0 else pos_f
                idx_t = pos_t[rng.choice(len(pos_t), n_t, replace=False)] if n_t > 0 else pos_t
                n_nt = min(len(neg_tib), max(1, int(2.1 * max(n_t, 1000))))
                n_nf = min(len(neg_fem), max(1, int(2.2 * max(n_f, 1000))))
                idx_nt = neg_tib[rng.choice(len(neg_tib), n_nt, replace=False)] if n_nt > 0 else neg_tib
                idx_nf = neg_fem[rng.choice(len(neg_fem), n_nf, replace=False)] if n_nf > 0 else neg_fem
                arrs = [idx_f, idx_t, idx_nt, idx_nf]

            sample_mask = np.zeros_like(roi, dtype=bool)
            for arr in arrs:
                if len(arr) > 0:
                    sample_mask[arr[:, 0], arr[:, 1], arr[:, 2]] = True

            X = _extract_from_precomputed(
                pre, dts, sp, sample_mask, atlas_p, prob_map=prob_map, sdt_jitter_std=0.0, jitter_seed=42 + idx + seed_offset
            )
            y = y_clean[sample_mask]
            return X, y

        n_est = max(self.n_estimators, 100)
        m_depth = max(self.max_depth, 25)

        if prob_maps is None:
            # --- PASS 1: Global + Tibial Specialist + 2-Fold Out-of-Fold Probabilities ---
            self.stage = 1
            sampled_g = Parallel(n_jobs=min(16, max(1, len(prepped))), prefer='threads')(
                delayed(_sample_case)(i, None, False, 0) for i in range(len(prepped))
            )
            X_g = np.vstack([s[0] for s in sampled_g])
            y_g = np.concatenate([s[1] for s in sampled_g])

            sampled_t = Parallel(n_jobs=min(16, max(1, len(prepped))), prefer='threads')(
                delayed(_sample_case)(i, None, True, 500) for i in range(len(prepped))
            )
            X_t = np.vstack([s[0] for s in sampled_t])
            y_t = np.concatenate([s[1] for s in sampled_t])

            print(f"    Fitting Pass-1 Global Cartilage Forest on {X_g.shape} and Tibial Specialist on {X_t.shape}...")
            self.clf = ExtraTreesClassifier(
                n_estimators=n_est, max_depth=m_depth, min_samples_leaf=5, max_features='sqrt', n_jobs=self.n_jobs, random_state=42
            ).fit(X_g, y_g)
            self.clf_tib_spec = ExtraTreesClassifier(
                n_estimators=n_est, max_depth=m_depth, min_samples_leaf=5, max_features='sqrt', n_jobs=self.n_jobs, random_state=43
            ).fit(X_t, y_t)

            # Fit 2-fold out-of-fold models so Pass 2 gets unbiased Pass-1 probability maps
            if len(prepped) >= 4:
                mid = len(prepped) // 2
                ng_mid = sum(len(s[1]) for s in sampled_g[:mid])
                nt_mid = sum(len(s[1]) for s in sampled_t[:mid])
                clf_g_a = ExtraTreesClassifier(
                    n_estimators=65, max_depth=24, min_samples_leaf=8, max_features='sqrt', n_jobs=self.n_jobs, random_state=42
                ).fit(X_g[:ng_mid], y_g[:ng_mid])
                clf_g_b = ExtraTreesClassifier(
                    n_estimators=65, max_depth=24, min_samples_leaf=8, max_features='sqrt', n_jobs=self.n_jobs, random_state=43
                ).fit(X_g[ng_mid:], y_g[ng_mid:])
                clf_t_a = ExtraTreesClassifier(
                    n_estimators=65, max_depth=24, min_samples_leaf=8, max_features='sqrt', n_jobs=self.n_jobs, random_state=44
                ).fit(X_t[:nt_mid], y_t[:nt_mid])
                clf_t_b = ExtraTreesClassifier(
                    n_estimators=65, max_depth=24, min_samples_leaf=8, max_features='sqrt', n_jobs=self.n_jobs, random_state=45
                ).fit(X_t[nt_mid:], y_t[nt_mid:])

                print("    Precomputing 2-fold Out-of-Fold Pass-1 Cartilage Probability Maps...")
                _SESSION_OOF_PROB_CACHE.clear()
                for start in range(0, len(prepped), 10):
                    idxs = list(range(start, min(start + 10, len(prepped))))
                    Xs_g = Parallel(n_jobs=len(idxs), prefer='threads')(
                        delayed(_extract_from_precomputed)(
                            prepped[i][5], prepped[i][4], prepped[i][3], prepped[i][5]['roi'], train_loo_atlases[i]
                        )
                        for i in idxs
                    )
                    Xs_t = Parallel(n_jobs=len(idxs), prefer='threads')(
                        delayed(_extract_from_precomputed)(
                            prepped[i][5], prepped[i][4], prepped[i][3], prepped[i][5]['roi_tib'], train_loo_atlases[i]
                        )
                        for i in idxs
                    )
                    for i, X_rg, X_rt in zip(idxs, Xs_g, Xs_t):
                        img, b_m, lbl, sp, dts, pre = prepped[i]
                        cg = clf_g_b if i < mid else clf_g_a
                        ct = clf_t_b if i < mid else clf_t_a
                        pg = cg.predict_proba(X_rg)
                        pt = ct.predict_proba(X_rt)
                        pv = np.zeros((*img.shape, 3), dtype=np.float32)
                        pv[..., 0] = 1.0
                        pv[pre['roi']] = pg
                        pv[pre['roi_tib']] = 0.40 * pv[pre['roi_tib']] + 0.60 * pt
                        df, dt = dts['femur'], dts['tibia']
                        pv[df > 5.5, 1] = 0.0
                        pv[dt > 5.5, 2] = 0.0
                        _SESSION_OOF_PROB_CACHE[_vol_fingerprint(img)] = pv
        else:
            # --- PASS 2: Auto-Context Global + Tibial Specialist ---
            self.stage = 2
            _SESSION_OOF_PROB_CACHE.clear()
            sampled_g = Parallel(n_jobs=min(16, max(1, len(prepped))), prefer='threads')(
                delayed(_sample_case)(i, prob_maps[i], False, 100) for i in range(len(prepped))
            )
            X_g = np.vstack([s[0] for s in sampled_g])
            y_g = np.concatenate([s[1] for s in sampled_g])

            sampled_t = Parallel(n_jobs=min(16, max(1, len(prepped))), prefer='threads')(
                delayed(_sample_case)(i, prob_maps[i], True, 700) for i in range(len(prepped))
            )
            X_t = np.vstack([s[0] for s in sampled_t])
            y_t = np.concatenate([s[1] for s in sampled_t])

            print(f"    Fitting Pass-2 Global Cartilage Forest on {X_g.shape} and Tibial Specialist on {X_t.shape}...")
            self.clf = ExtraTreesClassifier(
                n_estimators=n_est, max_depth=m_depth, min_samples_leaf=5, max_features='sqrt', n_jobs=self.n_jobs, random_state=42
            ).fit(X_g, y_g)
            self.clf_tib_spec = ExtraTreesClassifier(
                n_estimators=n_est, max_depth=m_depth, min_samples_leaf=5, max_features='sqrt', n_jobs=self.n_jobs, random_state=43
            ).fit(X_t, y_t)

    def predict_proba_map(
        self,
        image,
        bone_masks,
        proximity_mm=20.0,
        landmarks_dict=None,
        landmark_indices=None,
        dtype='float32',
        spacing=None,
    ):
        """
        Predicts dense 3-class probability map for Auto-Context Pass 2.
        """
        fp = _vol_fingerprint(image)
        if fp in _SESSION_OOF_PROB_CACHE:
            return _SESSION_OOF_PROB_CACHE.pop(fp)

        if spacing is None:
            spacing = _infer_default_spacing(image.shape)

        if getattr(self, 'stage', 1) in ('oai_v6_p1', 'oai_v6_p2'):
            from .oai_subchondral_atlas import predict_oai_cartilage_p1
            m_d = self.meta if 'p1_folds' in self.meta else self.meta['m_v6']
            _, pv = predict_oai_cartilage_p1(image, bone_masks, spacing, m_d)
            return pv

        if getattr(self, 'stage', 1) in ('unified_v6_p1', 'unified_v6_p2'):
            from .oai_subchondral_atlas import predict_unified_cartilage_p1
            _, pv = predict_unified_cartilage_p1(
                image, bone_masks, spacing, self.meta, model_dir=getattr(self, 'model_dir', None)
            )
            return pv

        if min(image.shape) < 32 or getattr(self, 'stage', 1) == 0 or self.atlas_priors is None:
            dts = compute_signed_distance_transforms(bone_masks)
            min_dt = np.min([np.abs(d) for d in dts.values()], axis=0)
            mask = min_dt.flatten() < proximity_mm
            X = extract_features(
                image,
                dts,
                mask=mask,
                landmarks_dict=landmarks_dict,
                landmark_indices=landmark_indices,
                sorted_bones_override=self.target_bones,
                target_dtype=dtype,
            )
            y_prob_masked = self.clf.predict_proba(X)
            n_cls = getattr(self.clf, 'n_classes_', len(self.clf.classes_))
            y_prob = np.zeros((*image.shape, n_cls), dtype=np.float32)
            y_prob.reshape(-1, n_cls)[mask] = y_prob_masked
            return y_prob

        if getattr(self, 'stage', 1) in ('v6_p1', 'v6_p2'):
            _, pv = self._predict_v6_p1(image, bone_masks, spacing)
            return pv

        dts = {
            'femur': fast_signed_distance_transform(bone_masks['femur'], spacing),
            'tibia': fast_signed_distance_transform(bone_masks['tibia'], spacing),
        }
        pre = _precompute_volume_maps(image, bone_masks, dts, spacing, lbl=None)
        X_g = _extract_from_precomputed(pre, dts, spacing, pre['roi'], self.atlas_priors, prob_map=None)
        probs_g = self.clf.predict_proba(X_g)

        prob_vol = np.zeros((*image.shape, 3), dtype=np.float32)
        prob_vol[..., 0] = 1.0
        prob_vol[pre['roi']] = probs_g

        if self.clf_tib_spec is not None and np.any(pre['roi_tib']):
            X_t = _extract_from_precomputed(pre, dts, spacing, pre['roi_tib'], self.atlas_priors, prob_map=None)
            probs_t = self.clf_tib_spec.predict_proba(X_t)
            prob_vol[pre['roi_tib']] = 0.40 * prob_vol[pre['roi_tib']] + 0.60 * probs_t

        df, dt = dts['femur'], dts['tibia']
        prob_vol[df > 5.5, 1] = 0.0
        prob_vol[dt > 5.5, 2] = 0.0
        return prob_vol

    def predict(
        self,
        image,
        bone_masks,
        proximity_mm=20.0,
        landmarks_dict=None,
        landmark_indices=None,
        prob_map=None,
        dtype='float32',
        spacing=None,
        pass1_probs=None,
    ):
        """
        Predicts cartilage labels (2=Femoral Cartilage, 4=Tibial Cartilage, 6=Patellar Cartilage) and probabilities.
        """
        if prob_map is None and pass1_probs is not None:
            prob_map = pass1_probs

        if spacing is None:
            spacing = _infer_default_spacing(image.shape)

        if getattr(self, 'stage', 1) == 'oai_v6_p1':
            from .oai_subchondral_atlas import predict_oai_cartilage_p1
            return predict_oai_cartilage_p1(image, bone_masks, spacing, self.meta)

        if getattr(self, 'stage', 1) == 'oai_v6_p2':
            from .oai_subchondral_atlas import predict_oai_cartilage_p2
            return predict_oai_cartilage_p2(image, bone_masks, prob_map, spacing, self.meta)

        if getattr(self, 'stage', 1) == 'unified_v6_p1':
            from .oai_subchondral_atlas import predict_unified_cartilage_p1
            return predict_unified_cartilage_p1(
                image, bone_masks, spacing, self.meta, model_dir=getattr(self, 'model_dir', None)
            )

        if getattr(self, 'stage', 1) == 'unified_v6_p2':
            from .oai_subchondral_atlas import predict_unified_cartilage_p2
            return predict_unified_cartilage_p2(
                image, bone_masks, prob_map, spacing, self.meta, model_dir=getattr(self, 'model_dir', None)
            )

        if min(image.shape) < 32 or getattr(self, 'stage', 1) == 0 or self.atlas_priors is None:
            dts = compute_signed_distance_transforms(bone_masks)
            min_dt = np.min([np.abs(d) for d in dts.values()], axis=0)
            mask = min_dt.flatten() < proximity_mm
            X = extract_features(
                image,
                dts,
                mask=mask,
                landmarks_dict=landmarks_dict,
                landmark_indices=landmark_indices,
                prob_map=prob_map,
                sorted_bones_override=self.target_bones,
                target_dtype=dtype,
            )
            y_pred_masked = self.clf.predict(X)
            y_prob_masked = self.clf.predict_proba(X)
            y_pred = np.zeros(image.size, dtype=np.uint8)
            y_pred[mask] = y_pred_masked
            n_cls = getattr(self.clf, 'n_classes_', len(self.clf.classes_))
            y_prob = np.zeros((*image.shape, n_cls), dtype=np.float32)
            y_prob.reshape(-1, n_cls)[mask] = y_prob_masked
            return y_pred.reshape(image.shape), y_prob

        if getattr(self, 'stage', 1) == 'v6_p1':
            return self._predict_v6_p1(image, bone_masks, spacing)
        if getattr(self, 'stage', 1) == 'v6_p2':
            return self._predict_v6_p2(image, bone_masks, spacing, prob_map=prob_map)

        dts = {
            'femur': fast_signed_distance_transform(bone_masks['femur'], spacing),
            'tibia': fast_signed_distance_transform(bone_masks['tibia'], spacing),
        }
        pre = _precompute_volume_maps(image, bone_masks, dts, spacing, lbl=None)

        if prob_map is not None:
            p_sum = prob_map[..., 1] + prob_map[..., 2]
            p_sum_s = gaussian_filter(p_sum, sigma=[1.5 / s for s in spacing])
            act_roi = pre['roi'] & (p_sum_s >= 0.015)
            if not np.any(act_roi):
                act_roi = pre['roi']
        else:
            act_roi = pre['roi']

        X_g = _extract_from_precomputed(pre, dts, spacing, act_roi, self.atlas_priors, prob_map=prob_map)
        probs_g = self.clf.predict_proba(X_g)

        pv = np.zeros((*image.shape, 3), dtype=np.float32)
        pv[..., 0] = 1.0
        pv[act_roi] = probs_g

        act_tib = act_roi & pre['roi_tib']
        if self.clf_tib_spec is not None and np.any(act_tib):
            X_t = _extract_from_precomputed(pre, dts, spacing, act_tib, self.atlas_priors, prob_map=prob_map)
            probs_t = self.clf_tib_spec.predict_proba(X_t)
            pv[act_tib] = 0.35 * pv[act_tib] + 0.65 * probs_t

        df, dt = dts['femur'], dts['tibia']
        pv[df > 5.5, 1] = 0.0
        pv[dt > 5.5, 2] = 0.0

        p_f = gaussian_filter(pv[..., 1], sigma=[0.75 / spacing[0], 0.45 / spacing[1], 0.65 / spacing[2]])
        p_t = gaussian_filter(pv[..., 2], sigma=[0.85 / spacing[0], 0.45 / spacing[1], 0.75 / spacing[2]])

        pred_fc = _keep_top_k_components((p_f >= 0.48) & (p_f >= p_t), k=1, min_size=300)
        pred_tc = _keep_top_k_components((p_t >= 0.46) & (p_t > p_f), k=2, min_size=150)

        y_pred = np.zeros(image.shape, dtype=np.uint8)
        y_pred[pred_fc] = 2
        y_pred[pred_tc] = 4

        return y_pred, pv

    def _predict_v6_p1(self, image, bone_masks, sp):
        from .cartilage_v6 import predict_cartilage_v6_p1

        meta = getattr(self, 'meta', {})
        p1_models = meta.get('p1_folds')
        if p1_models is None and 'm_v6' in meta:
            p1_models = meta['m_v6']['p1_folds']
        return predict_cartilage_v6_p1(
            image,
            bone_masks,
            sp,
            p1_models,
            self.atlas_priors,
            model_dir=getattr(self, 'model_dir', None),
        )

    def _predict_v6_p2(self, image, bone_masks, sp, prob_map=None):
        from .cartilage_v6 import predict_cartilage_v6_p2

        meta = getattr(self, 'meta', {})
        return predict_cartilage_v6_p2(
            image,
            bone_masks,
            sp,
            meta,
            prob_map=prob_map,
            model_dir=getattr(self, 'model_dir', None),
        )

    def save(self, path):
        payload = dict(getattr(self, 'meta', {}))
        payload.update(
            {
                'clf': self.clf,
                'clf_tib_spec': self.clf_tib_spec,
                'atlas_priors': self.atlas_priors,
                'stage': getattr(self, 'stage', 1),
                'target_bones': self.target_bones,
            }
        )
        joblib.dump(payload, path)

    def load(self, path):
        self.model_dir = os.path.dirname(os.path.abspath(path))
        data = joblib.load(path)
        if isinstance(data, dict) and data.get('format_version') in (
            'oai_cartilage_v6_p1', 'oai_cartilage_v6_p2', 'unified_ski10_oai_cart_p1', 'unified_ski10_oai_cart_p2'
        ):
            fmt = data['format_version']
            self.clf = None
            self.clf_tib_spec = None
            self.meta = data
            self.target_bones = ['femur', 'tibia', 'patella']
            if fmt == 'oai_cartilage_v6_p1':
                self.stage = 'oai_v6_p1'
                self.atlas_priors = data.get('atlas_priors', {'oai': True})
            elif fmt == 'oai_cartilage_v6_p2':
                self.stage = 'oai_v6_p2'
                self.atlas_priors = data.get('m_v6', {}).get('atlas_priors', {'oai': True})
            elif fmt == 'unified_ski10_oai_cart_p1':
                self.stage = 'unified_v6_p1'
                self.atlas_priors = {'unified': True}
            else:
                self.stage = 'unified_v6_p2'
                self.atlas_priors = {'unified': True}
            return

        if isinstance(data, dict) and 'clf' in data:
            self.clf = data['clf']
            self.clf_tib_spec = data.get('clf_tib_spec')
            self.atlas_priors = data.get('atlas_priors')
            self.stage = data.get('stage', 1)
            self.meta = data
            if data.get('target_bones') is not None:
                self.target_bones = data['target_bones']
        else:
            self.clf = data
            self.clf_tib_spec = None
            self.atlas_priors = None
            self.stage = 0
            self.meta = {}
            if self.target_bones is None:
                classes = getattr(self.clf, 'classes_', [])
                n_feats = getattr(self.clf, 'n_features_in_', 0)
                if len(classes) >= 6 or n_feats in (38, 157):
                    self.target_bones = ['femur', 'tibia', 'patella']
                elif len(classes) == 5 or n_feats in (37, 122):
                    self.target_bones = ['femur', 'tibia']

