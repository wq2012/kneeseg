"""Self-contained V6/V5/V7 Multi-Atlas, Auto-Context, Stage-3, and Calibrator cartilage inference."""

import os

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

import joblib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from kneeseg.features import fast_signed_distance_transform


_D_GRID = np.array(
    [-3.6, -2.8, -2.0, -1.5, -1.1, -0.75, -0.4, 0.0, 0.4, 0.75, 1.1, 1.5, 2.0, 2.8, 3.6],
    dtype=np.float32,
)
_H_GRID = np.array([-1.0, -0.5, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.2], dtype=np.float32)
_DY_STEPS = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=int)
_R_STEPS = np.array([-6.0, -3.5, -1.8, 1.8, 3.5, 6.0], dtype=np.float32)

_SIG_T_CALIB = [0.35 / 1.5, 0.14 / 0.39, 0.25 / 0.39]
_PTB_0_CALIB = 0.12

# In-memory session cache so Pass-1 -> Pass-2 calls on the same volume in inference.py
# reuse Pass-1 fold predictions without recomputing them twice.
_CART_P1_SESSION_CACHE = {}


def fast_keep_top_k(mask, k=1, min_size=100):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    inds = np.argwhere(mask)
    z0, y0, x0 = inds.min(axis=0)
    z1, y1, x1 = inds.max(axis=0) + 1
    crop = mask[z0:z1, y0:y1, x0:x1]
    labeled, n_c = label(crop)
    if n_c == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    valid = np.where(counts >= min_size)[0]
    if len(valid) == 0:
        return np.zeros_like(mask, dtype=bool)
    top_k = valid[np.argsort(counts[valid])[::-1][:k]]
    out = np.zeros_like(mask, dtype=bool)
    out[z0:z1, y0:y1, x0:x1] = np.isin(labeled, top_k)
    return out


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


def compute_ml_frames(bm, df, dt, sp):
    sp_arr = np.array(sp, dtype=np.float32)
    med_low_z = detect_medial_is_low_z(bm, sp)
    tb = bm['tibia'] > 0
    inds_tb = np.argwhere(tb & (dt >= -2.0) & (df <= 32.0))
    if len(inds_tb) < 10:
        inds_tb = np.argwhere(tb)
    z0, y0, x0 = np.maximum(inds_tb.min(axis=0) - 2, 0)
    z1, y1, x1 = np.minimum(inds_tb.max(axis=0) + 3, tb.shape)
    dts = gaussian_filter(
        dt[z0:z1, y0:y1, x0:x1], sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]]
    )
    _, nyt, _ = np.gradient(dts, *sp)

    plat_sub = (
        tb[z0:z1, y0:y1, x0:x1]
        & (dt[z0:z1, y0:y1, x0:x1] >= -1.2)
        & (df[z0:z1, y0:y1, x0:x1] <= 24.0)
        & (nyt <= -0.45)
    )
    if np.sum(plat_sub) < 100:
        plat_sub = (
            tb[z0:z1, y0:y1, x0:x1]
            & (dt[z0:z1, y0:y1, x0:x1] >= -1.5)
            & (df[z0:z1, y0:y1, x0:x1] <= 25.0)
        )
    if np.sum(plat_sub) < 10:
        plat_sub = tb[z0:z1, y0:y1, x0:x1]
    pts = (np.argwhere(plat_sub) + np.array([z0, y0, x0])) * sp_arr
    p_min = np.percentile(pts, 2, axis=0)
    p_max = np.percentile(pts, 98, axis=0)
    p_mid = 0.5 * (p_min + p_max)
    p_span = np.maximum(p_max - p_min, 5.0)

    z_med_split = float(np.median(pts[:, 0]))
    pts_low = pts[pts[:, 0] < z_med_split]
    pts_high = pts[pts[:, 0] >= z_med_split]
    if len(pts_low) == 0:
        pts_low = pts
    if len(pts_high) == 0:
        pts_high = pts
    pts_M = pts_low if med_low_z else pts_high
    pts_L = pts_high if med_low_z else pts_low
    c_M, s_M = pts_M.mean(axis=0), pts_M.std(axis=0) + 1.2
    c_L, s_L = pts_L.mean(axis=0), pts_L.std(axis=0) + 1.2
    return med_low_z, (p_min, p_max, p_mid, p_span), (c_M, s_M, c_L, s_L, z_med_split)


def compute_atlas_bins_v6(coords_mm, dfs_m, dts_m, f_stats, t_comp, ml_info):
    med_low_z, (p_min, p_max, p_mid, p_span), (c_M, s_M, c_L, s_L, z_med_split) = ml_info

    c1_t, s1_t, c2_t, s2_t, spine_t, z_mid_t = t_comp
    in_1t = coords_mm[:, 0] < z_mid_t
    uz_t = np.where(in_1t, (c1_t[0] - coords_mm[:, 0]) / s1_t[0], (coords_mm[:, 0] - c2_t[0]) / s2_t[0])
    uy_t = np.where(in_1t, (coords_mm[:, 1] - c1_t[1]) / s1_t[1], (coords_mm[:, 1] - c2_t[1]) / s2_t[1])
    ux_t = np.where(in_1t, (coords_mm[:, 2] - c1_t[2]) / s1_t[2], (coords_mm[:, 2] - c2_t[2]) / s2_t[2])
    bz_old = np.clip(((uz_t + 2.8) / 5.6 * 26).astype(int), 0, 25)
    bx_old = np.clip(((ux_t + 2.8) / 5.6 * 26).astype(int), 0, 25)
    bdt_16 = np.clip(((dts_m + 0.5) / 6.0 * 16).astype(int), 0, 15)
    bdf_18 = np.clip(((dfs_m + 0.5) / 26.0 * 18).astype(int), 0, 17)

    u_ml = (coords_mm[:, 0] - p_mid[0]) / p_span[0]
    if not med_low_z:
        u_ml = -u_ml
    u_ap = (coords_mm[:, 2] - p_mid[2]) / p_span[2]
    u_y = (coords_mm[:, 1] - p_mid[1]) / p_span[1]
    bz_ml = np.clip(((u_ml + 0.62) / 1.24 * 34).astype(int), 0, 33)
    bx_ml = np.clip(((u_ap + 0.62) / 1.24 * 26).astype(int), 0, 25)
    by_ml = np.clip(((u_y + 0.85) / 1.70 * 18).astype(int), 0, 17)

    bz_wide = np.clip(((u_ml + 0.82) / 1.64 * 32).astype(int), 0, 31)
    bx_wide = np.clip(((u_ap + 0.85) / 1.70 * 28).astype(int), 0, 27)

    in_low = coords_mm[:, 0] < z_med_split
    in_M = in_low if med_low_z else (~in_low)
    is_lat = (~in_M).astype(int)
    dir_M = -1.0 if med_low_z else 1.0
    dir_L = 1.0 if med_low_z else -1.0
    uz_p = np.where(in_M, dir_M * (coords_mm[:, 0] - c_M[0]) / s_M[0], dir_L * (coords_mm[:, 0] - c_L[0]) / s_L[0])
    uy_p = np.where(in_M, (coords_mm[:, 1] - c_M[1]) / s_M[1], (coords_mm[:, 1] - c_L[1]) / s_L[1])
    ux_p = np.where(in_M, (coords_mm[:, 2] - c_M[2]) / s_M[2], (coords_mm[:, 2] - c_L[2]) / s_L[2])
    bz_p = np.clip(((uz_p + 3.0) / 6.0 * 24).astype(int), 0, 23)
    bx_p = np.clip(((ux_p + 3.2) / 6.4 * 24).astype(int), 0, 23)

    f_min, f_max, _ = f_stats
    rel_f = (coords_mm - f_min) / (f_max - f_min + 1e-6)
    z_sym_f = np.abs(rel_f[:, 0] - 0.5) * 2.0
    bz_fo = np.clip((z_sym_f / 1.25 * 22).astype(int), 0, 21)
    by_f = np.clip(((rel_f[:, 1] + 0.15) / 1.30 * 26).astype(int), 0, 25)
    bx_f = np.clip(((rel_f[:, 2] + 0.15) / 1.30 * 26).astype(int), 0, 25)
    bdf_f = np.clip(((dfs_m + 0.5) / 6.0 * 16).astype(int), 0, 15)

    rel_fz_ml = rel_f[:, 0] if med_low_z else (1.0 - rel_f[:, 0])
    bz_fml = np.clip(((rel_fz_ml + 0.10) / 1.20 * 34).astype(int), 0, 33)
    is_lat_f = (rel_fz_ml >= 0.5).astype(int)
    uz_fsep = np.where(is_lat_f == 1, (rel_fz_ml - 0.5) * 2.0, (0.5 - rel_fz_ml) * 2.0)
    bz_fsep = np.clip((uz_fsep / 1.25 * 22).astype(int), 0, 21)

    bins = {
        't_old': (bz_old, bx_old, bdt_16, bdf_18),
        't_ml': (bz_ml, bx_ml, bdt_16, bdf_18),
        't_mly': (bz_ml, by_ml, bx_ml, bdt_16),
        't_sep': (is_lat, bz_p, bx_p, bdt_16, bdf_18),
        't_wide': (bz_wide, bx_wide, bdt_16, bdf_18),
        'f_old': (bz_fo, by_f, bx_f, bdf_f),
        'f_ml': (bz_fml, by_f, bx_f, bdf_f),
        'f_sep': (is_lat_f, bz_fsep, by_f, bx_f, bdf_f),
    }
    coords = {
        'uz_t': uz_t, 'uy_t': uy_t, 'ux_t': ux_t,
        'u_ml': u_ml, 'u_ap': u_ap, 'u_y': u_y,
        'is_lat': is_lat.astype(np.float32),
        'uz_p': uz_p, 'uy_p': uy_p, 'ux_p': ux_p,
        'spine_z': spine_t[0],
        'rel_f': rel_f, 'rel_fz_ml': rel_fz_ml, 'z_sym_f': z_sym_f,
    }
    return bins, coords


def build_geometry_from_dts(
    img, sp, df, dt, df_alt, dt_alt, df_raw, dt_raw,
    bone_name, atlas_p, f_stats, f_comp, t_stats, t_comp, ml_info,
    use_atlas_box_for_femur=True,
):
    if bone_name == 'tibia':
        shell = (dt >= -0.95) & (dt <= 4.6) & (df >= -1.0) & (df <= 25.0)
    elif use_atlas_box_for_femur:
        raw_shell = (df >= -0.95) & (df <= 4.7) & (dt >= -1.0) & (dt <= 56.0)
        if np.any(raw_shell):
            raw_inds = np.argwhere(raw_shell)
            coords_raw = raw_inds.astype(np.float32) * np.array(sp, dtype=np.float32)
            dfs_raw = df[raw_inds[:, 0], raw_inds[:, 1], raw_inds[:, 2]]
            dts_raw = dt[raw_inds[:, 0], raw_inds[:, 1], raw_inds[:, 2]]
            bins_r, c_dict_r = compute_atlas_bins_v6(
                coords_raw, dfs_raw, dts_raw, f_stats, t_comp, ml_info
            )
            rel_f_r = c_dict_r['rel_f']
            in_box = (
                (rel_f_r[:, 0] >= -0.10) & (rel_f_r[:, 0] <= 1.10)
                & (rel_f_r[:, 1] >= -0.12) & (rel_f_r[:, 1] <= 1.06)
                & (rel_f_r[:, 2] >= -0.12) & (rel_f_r[:, 2] <= 1.10)
            )
            p_ens = (
                0.45 * atlas_p['pf_fml_fc'][bins_r['f_ml']]
                + 0.35 * atlas_p['pf_sep_fc'][bins_r['f_sep']]
                + 0.20 * atlas_p['pf_old_fc'][bins_r['f_old']]
            )
            p3d = atlas_p['p3d_fml_fc'][bins_r['f_ml'][0], bins_r['f_ml'][1], bins_r['f_ml'][2]]
            keep = in_box & ((p_ens > 0.002) | (p3d > 0.004) | (dts_raw <= 42.0))
            shell = np.zeros_like(raw_shell, dtype=bool)
            if np.any(keep):
                kept_inds = raw_inds[keep]
                shell[kept_inds[:, 0], kept_inds[:, 1], kept_inds[:, 2]] = True
            else:
                shell = raw_shell
        else:
            shell = raw_shell
    else:
        shell = (df >= -0.95) & (df <= 4.7) & (dt >= -1.0) & (dt <= 9.0)

    if not np.any(shell):
        shell = (np.abs(df if bone_name == 'femur' else dt) <= 5.0)
    if not np.any(shell):
        shell = np.zeros(img.shape, dtype=bool)
        shell[0, 0, 0] = True

    all_inds = np.argwhere(shell)
    margin = [int(np.ceil(6.0 / s)) for s in sp]
    z0, y0, x0 = np.maximum(all_inds.min(axis=0) - margin, 0)
    z1, y1, x1 = np.minimum(all_inds.max(axis=0) + margin + 1, img.shape)

    df_crop = df[z0:z1, y0:y1, x0:x1]
    dt_crop = dt[z0:z1, y0:y1, x0:x1]
    df_alt_crop = df_alt[z0:z1, y0:y1, x0:x1]
    dt_alt_crop = dt_alt[z0:z1, y0:y1, x0:x1]
    df_raw_crop = df_raw[z0:z1, y0:y1, x0:x1]
    dt_raw_crop = dt_raw[z0:z1, y0:y1, x0:x1]

    dfs_crop = gaussian_filter(df_crop, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
    dts_crop = gaussian_filter(dt_crop, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])

    d_self_crop = df_crop if bone_name == 'femur' else dt_crop
    d_other_crop = dt_crop if bone_name == 'femur' else df_crop
    d_alt_self_crop = df_alt_crop if bone_name == 'femur' else dt_alt_crop
    d_raw_self_crop = df_raw_crop if bone_name == 'femur' else dt_raw_crop
    ds_self_crop = dfs_crop if bone_name == 'femur' else dts_crop
    ds_other_crop = dts_crop if bone_name == 'femur' else dfs_crop

    nz, ny, nx = np.gradient(ds_self_crop, *sp)
    norm = np.sqrt(nz**2 + ny**2 + nx**2) + 1e-6
    nz, ny, nx = (nz / norm).astype(np.float32), (ny / norm).astype(np.float32), (nx / norm).astype(np.float32)

    nzo, nyo, nxo = np.gradient(ds_other_crop, *sp)
    normo = np.sqrt(nzo**2 + nyo**2 + nxo**2) + 1e-6
    nzo, nyo, nxo = (nzo / normo).astype(np.float32), (nyo / normo).astype(np.float32), (nxo / normo).astype(np.float32)

    img_f32 = img[z0:z1, y0:y1, x0:x1].astype(np.float32)
    m_low = (d_self_crop <= -1.5) & (d_self_crop >= -5.5)
    m_high = (d_self_crop >= 0.4) & (d_self_crop <= 2.0) & (
        d_other_crop <= (12.0 if bone_name == 'femur' else 10.0)
    )
    i_low = float(np.median(img_f32[m_low])) if np.any(m_low) else float(np.percentile(img_f32, 15))
    if np.any(m_high):
        vals_h = img_f32[m_high]
        i_high = float(np.percentile(vals_h, 75))
        i_med_h = float(np.median(vals_h))
        iqr_h = max(float(np.percentile(vals_h, 75) - np.percentile(vals_h, 25)), 1.0)
    else:
        i_high = float(np.percentile(img_f32, 85))
        i_med_h = i_high
        iqr_h = 40.0

    inorm = (img_f32 - i_low) / max(i_high - i_low, 1.0)
    z_cart = (img_f32 - i_med_h) / iqr_h

    img_lat_raw = gaussian_filter(img_f32, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
    sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
    w_bn = ((d_self_crop <= -1.4) & (d_self_crop >= -5.5)).astype(np.float32) + 1e-5
    w_bn_s = gaussian_filter(w_bn, sigma=sig_bn)
    i_bn_mean = gaussian_filter(img_lat_raw * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_var = gaussian_filter(((img_lat_raw - i_bn_mean) ** 2) * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_std = np.maximum(np.sqrt(np.maximum(i_bn_var, 0.0)), max(0.05 * (i_high - i_low), 2.5))
    z_bone_crop = ((img_lat_raw - i_bn_mean) / i_bn_std).astype(np.float32)

    return {
        'bone_name': bone_name,
        'shell': shell,
        'bbox': (z0, z1, y0, y1, x0, x1),
        'sp': sp,
        'img_f32': img_f32,
        'inorm': inorm,
        'z_cart': z_cart,
        'z_bone_crop': z_bone_crop,
        'i_low': i_low,
        'i_high': i_high,
        'df_crop': df_crop,
        'dt_crop': dt_crop,
        'dfs_crop': dfs_crop,
        'dts_crop': dts_crop,
        'd_self_crop': d_self_crop,
        'd_other_crop': d_other_crop,
        'd_alt_self_crop': d_alt_self_crop,
        'd_raw_self_crop': d_raw_self_crop,
        'ds_self_crop': ds_self_crop,
        'ds_other_crop': ds_other_crop,
        'normals_self': (nz, ny, nx),
        'normals_other': (nzo, nyo, nxo),
        'f_stats': f_stats,
        'f_comp': f_comp,
        't_stats': t_stats,
        't_comp': t_comp,
        'ml_info': ml_info,
        'atlas_p': atlas_p,
    }


def compute_articular_landmarks_and_stats(bone_mask, d_self, d_other, max_other_mm, spacing):
    from .rf_seg import _compute_articular_landmarks_and_stats
    return _compute_articular_landmarks_and_stats(bone_mask, d_self, d_other, max_other_mm, spacing)


def prepare_geometry_v6(
    img, sp, bm_ref, bm_alt, bm_raw, bone_name, atlas_p,
    dts_pre=None, landmarks_pre=None, use_atlas_box_for_femur=True,
):
    if dts_pre is not None:
        df, dt, df_alt, dt_alt, df_raw, dt_raw = dts_pre
    else:
        df = fast_signed_distance_transform(bm_ref['femur'] > 0, sp)
        dt = fast_signed_distance_transform(bm_ref['tibia'] > 0, sp)
        df_alt = df if (bm_alt['femur'] is bm_ref['femur']) else fast_signed_distance_transform(bm_alt['femur'] > 0, sp)
        dt_alt = dt if (bm_alt['tibia'] is bm_ref['tibia']) else fast_signed_distance_transform(bm_alt['tibia'] > 0, sp)
        df_raw = df if (bm_raw['femur'] is bm_ref['femur']) else fast_signed_distance_transform(bm_raw['femur'] > 0, sp)
        dt_raw = dt if (bm_raw['tibia'] is bm_ref['tibia']) else fast_signed_distance_transform(bm_raw['tibia'] > 0, sp)

    if landmarks_pre is not None:
        f_stats, f_comp, t_stats, t_comp, ml_info = landmarks_pre
    else:
        f_stats, _, f_comp = compute_articular_landmarks_and_stats(bm_ref['femur'], df, dt, 45.0, sp)
        t_stats, _, t_comp = compute_articular_landmarks_and_stats(bm_ref['tibia'], dt, df, 25.0, sp)
        ml_info = compute_ml_frames(bm_ref, df, dt, sp)

    return build_geometry_from_dts(
        img, sp, df, dt, df_alt, dt_alt, df_raw, dt_raw,
        bone_name, atlas_p, f_stats, f_comp, t_stats, t_comp, ml_info,
        use_atlas_box_for_femur=use_atlas_box_for_femur,
    )


def add_v7_fields(geo):
    z0, z1, y0, y1, x0, x1 = geo['bbox']
    sp = geo['sp']
    img_f32 = geo['img_f32']
    d_self_crop = geo['d_self_crop']
    d_other_crop = geo['d_other_crop']
    i_low, i_high = geo['i_low'], geo['i_high']
    img_f0_raw = gaussian_filter(img_f32, sigma=[0.35 / s for s in sp])
    sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
    w_bn = ((d_self_crop <= -1.4) & (d_self_crop >= -5.5)).astype(np.float32) + 1e-5
    w_bn_s = gaussian_filter(w_bn, sigma=sig_bn)
    i_bn_mean = gaussian_filter(img_f0_raw * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_var = gaussian_filter(((img_f0_raw - i_bn_mean) ** 2) * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_std = np.maximum(np.sqrt(np.maximum(i_bn_var, 0.0)), max(0.05 * (i_high - i_low), 2.5))
    z_bone_sharp = ((img_f0_raw - i_bn_mean) / i_bn_std).astype(np.float32)
    m_low = (d_self_crop <= -1.5) & (d_self_crop >= -5.5)
    m_high = (d_self_crop >= 0.4) & (d_self_crop <= 1.9) & (d_other_crop <= 9.5)
    zb_med_glob = max(float(np.median(z_bone_sharp[m_high])), 1.0) if np.any(m_high) else 2.5
    z_med_split = geo['ml_info'][2][4]
    z_mm = (np.arange(z0, z1, dtype=np.float32) * sp[0])[:, None, None]
    w_high_z = np.clip(0.5 + (z_mm - z_med_split) / 4.0, 0.0, 1.0).astype(np.float32)
    comp_masks = [(z_mm < z_med_split), (z_mm >= z_med_split)]
    i_low_c, i_high_c, i_med_c, iqr_c, zb_med_c = [], [], [], [], []
    for cm in comp_masks:
        ml_c = m_low & cm
        mh_c = m_high & cm
        il = float(np.median(img_f32[ml_c])) if np.sum(ml_c) >= 30 else i_low
        if np.sum(mh_c) >= 30:
            vh = img_f32[mh_c]
            ih = float(np.percentile(vh, 75))
            im = float(np.median(vh))
            iq = max(float(np.percentile(vh, 75) - np.percentile(vh, 25)), 1.0)
            zm = max(float(np.median(z_bone_sharp[mh_c])), 1.0)
        else:
            ih = i_high
            im = float(np.median(img_f32[m_high])) if np.any(m_high) else i_high
            iq = 25.0
            zm = zb_med_glob
        i_low_c.append(il)
        i_high_c.append(ih)
        i_med_c.append(im)
        iqr_c.append(iq)
        zb_med_c.append(zm)
    geo['z_bone_sharp'] = z_bone_sharp
    geo['zb_med_glob'] = zb_med_glob
    geo['w_high_z'] = w_high_z
    geo['comp_masks'] = comp_masks
    geo['comp_stats'] = (i_low_c, i_high_c, i_med_c, iqr_c, zb_med_c)
    return geo


def extract_features_v6(geo, target_mask, pm_self=None, pm_other=None, rng_jit=None, leak_free_jit=True):
    bone_name = geo['bone_name']
    z0, z1, y0, y1, x0, x1 = geo['bbox']
    sp = geo['sp']
    inds = np.argwhere(target_mask)
    zc, yc, xc = inds[:, 0] - z0, inds[:, 1] - y0, inds[:, 2] - x0
    coords_mm = inds.astype(np.float32) * np.array(sp, dtype=np.float32)
    N = len(zc)

    d_raw_m = geo['d_self_crop'][zc, yc, xc]
    d_alt_m = geo['d_alt_self_crop'][zc, yc, xc]
    d_st2_m = geo['d_raw_self_crop'][zc, yc, xc]
    d_s_m = geo['ds_self_crop'][zc, yc, xc]
    do_raw_m = geo['d_other_crop'][zc, yc, xc]
    do_s_m = geo['ds_other_crop'][zc, yc, xc]

    shift_ref_alt = d_raw_m - d_alt_m
    shift_ref_st2 = d_raw_m - d_st2_m

    d_raw_feat = d_raw_m
    d_alt_feat = d_alt_m
    d_s_feat = d_s_m
    d_s_anchor = d_s_m

    dfs_for_bins = d_s_anchor if bone_name == 'femur' else do_s_m
    dts_for_bins = d_s_anchor if bone_name == 'tibia' else do_s_m
    ap = geo['atlas_p']
    bins, c_dict = compute_atlas_bins_v6(
        coords_mm, dfs_for_bins, dts_for_bins, geo['f_stats'], geo['t_comp'], geo['ml_info']
    )

    pt_old_tc = ap['pt_old_tc'][bins['t_old']]
    pt_old_fc = ap['pt_old_fc'][bins['t_old']]
    pt_ml_tc = ap['pt_ml_tc'][bins['t_ml']]
    pt_ml_fc = ap['pt_ml_fc'][bins['t_ml']]
    pt_mly_tc = ap['pt_mly_tc'][bins['t_mly']]
    pt_sep_tc = ap['pt_sep_tc'][bins['t_sep']]
    pt_sep_fc = ap['pt_sep_fc'][bins['t_sep']]
    pt_wide_tc = ap['pt_wide_tc'][bins['t_wide']]

    pt_ens_tc = 0.32 * pt_ml_tc + 0.30 * pt_sep_tc + 0.20 * pt_mly_tc + 0.18 * pt_wide_tc
    pt_ens_fc = 0.50 * pt_ml_fc + 0.50 * pt_sep_fc
    bdt_m1, bdt_p1 = np.clip(bins['t_ml'][2] - 1, 0, 15), np.clip(bins['t_ml'][2] + 1, 0, 15)
    pt_ml_max3 = np.maximum(
        np.maximum(ap['pt_ml_tc'][bins['t_ml'][0], bins['t_ml'][1], bdt_m1, bins['t_ml'][3]], pt_ml_tc),
        ap['pt_ml_tc'][bins['t_ml'][0], bins['t_ml'][1], bdt_p1, bins['t_ml'][3]],
    )
    p2d_ml_v = ap['p2d_ml_tc'][bins['t_ml'][0], bins['t_ml'][1]]
    edt_ml_v = ap['edt_ml_2d'][bins['t_ml'][0], bins['t_ml'][1]]
    p2d_sep_v = ap['p2d_sep_tc'][bins['t_sep'][0], bins['t_sep'][1], bins['t_sep'][2]]
    edt_sep_v = ap['edt_sep_2d'][bins['t_sep'][0], bins['t_sep'][1], bins['t_sep'][2]]
    p2d_wide_v = ap['p2d_wide_tc'][bins['t_wide'][0], bins['t_wide'][1]]
    edt_wide_v = ap['edt_wide_2d'][bins['t_wide'][0], bins['t_wide'][1]]
    p2d_ens = 0.40 * p2d_ml_v + 0.35 * p2d_sep_v + 0.25 * p2d_wide_v
    edt_ens = 0.40 * edt_ml_v + 0.35 * edt_sep_v + 0.25 * edt_wide_v

    pf_old_fc = ap['pf_old_fc'][bins['f_old']]
    pf_old_tc = ap['pf_old_tc'][bins['f_old']]
    pf_fml_fc = ap['pf_fml_fc'][bins['f_ml']]
    pf_fml_tc = ap['pf_fml_tc'][bins['f_ml']]
    pf_sep_fc = ap['pf_sep_fc'][bins['f_sep']]
    pf_sep_tc = ap['pf_sep_tc'][bins['f_sep']]
    pf_ens_fc = 0.45 * pf_fml_fc + 0.35 * pf_sep_fc + 0.20 * pf_old_fc
    pf_ens_tc = 0.50 * pf_fml_tc + 0.50 * pf_sep_tc

    bdf_m1, bdf_p1 = np.clip(bins['f_ml'][3] - 1, 0, 15), np.clip(bins['f_ml'][3] + 1, 0, 15)
    pf_fml_max3 = np.maximum(
        np.maximum(ap['pf_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2], bdf_m1], pf_fml_fc),
        ap['pf_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2], bdf_p1],
    )
    p3d_fml_v = ap['p3d_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2]]
    edf_fml_v = ap['edf_fml_3d'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2]]

    img_f32 = geo['img_f32']
    i_low = geo['i_low']
    i_high = geo['i_high']
    z_cart = geo['z_cart']
    z_bone = geo['z_bone_crop']

    if '_img_f0_0' not in geo:
        inorm_0 = geo['inorm']
        geo['_base_span'] = max(i_high - i_low, 1.0)
        geo['_abs_z'] = gaussian_filter(np.abs(z_cart), sigma=[1.0 / sp[0], 0.35 / sp[1], 1.0 / sp[2]])
        geo['_img_f0_0'] = gaussian_filter(inorm_0, sigma=[0.35 / s for s in sp])
        geo['_img_lat_0'] = gaussian_filter(inorm_0, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
        geo['_img_s1_0'] = gaussian_filter(inorm_0, sigma=[1.0 / s for s in sp])
        geo['_img_s2_0'] = gaussian_filter(inorm_0, sigma=[2.0 / s for s in sp])
        geo['_img_s4_0'] = gaussian_filter(inorm_0, sigma=[4.0 / s for s in sp])
        geo['_img_s8_0'] = gaussian_filter(inorm_0, sigma=[8.0 / s for s in sp])
        gz0_0, gy0_0, gx0_0 = np.gradient(geo['_img_f0_0'], *sp)
        geo['_g0_0'] = (gz0_0, gy0_0, gx0_0, np.sqrt(gz0_0**2 + gy0_0**2 + gx0_0**2))
        gz1_0, gy1_0, gx1_0 = np.gradient(geo['_img_s1_0'], *sp)
        geo['_g1_0'] = (gz1_0, gy1_0, gx1_0, np.sqrt(gz1_0**2 + gy1_0**2 + gx1_0**2))

    if pm_self is not None:
        pm_crop = pm_self[z0:z1, y0:y1, x0:x1]
        m_p1_core = pm_crop[..., 2] >= 0.55
        if np.sum(m_p1_core) >= 50:
            i_high = float(np.median(img_f32[m_p1_core]))

    scale = float(geo['_base_span'] / max(i_high - i_low, 1.0))
    abs_z = geo['_abs_z']
    if abs(scale - 1.0) < 1e-7:
        img_f0 = geo['_img_f0_0']
        img_lat = geo['_img_lat_0']
        img_s1 = geo['_img_s1_0']
        img_s2 = geo['_img_s2_0']
        img_s4 = geo['_img_s4_0']
        img_s8 = geo['_img_s8_0']
        gz0, gy0, gx0, gmag0 = geo['_g0_0']
        gz1, gy1, gx1, gmag1 = geo['_g1_0']
    else:
        img_f0 = geo['_img_f0_0'] * scale
        img_lat = geo['_img_lat_0'] * scale
        img_s1 = geo['_img_s1_0'] * scale
        img_s2 = geo['_img_s2_0'] * scale
        img_s4 = geo['_img_s4_0'] * scale
        img_s8 = geo['_img_s8_0'] * scale
        gz0_0, gy0_0, gx0_0, gmag0_0 = geo['_g0_0']
        gz1_0, gy1_0, gx1_0, gmag1_0 = geo['_g1_0']
        gz0, gy0, gx0, gmag0 = gz0_0 * scale, gy0_0 * scale, gx0_0 * scale, gmag0_0 * scale
        gz1, gy1, gx1, gmag1 = gz1_0 * scale, gy1_0 * scale, gx1_0 * scale, gmag1_0 * scale

    nz, ny, nx = geo['normals_self']
    nzo, nyo, nxo = geo['normals_other']
    nz_m, ny_m, nx_m = nz[zc, yc, xc], ny[zc, yc, xc], nx[zc, yc, xc]
    nzo_m, nyo_m, nxo_m = nzo[zc, yc, xc], nyo[zc, yc, xc], nxo[zc, yc, xc]
    cz_max, cy_max, cx_max = img_f32.shape

    K = len(_D_GRID)
    rv0 = np.empty((N, K), dtype=np.float32)
    rv_lat = np.empty((N, K), dtype=np.float32)
    rv1 = np.empty((N, K), dtype=np.float32)
    rv_zb = np.empty((N, K), dtype=np.float32)
    for k, d_val in enumerate(_D_GRID):
        sz = np.clip(np.round(zc + d_val * nz_m / sp[0]).astype(int), 0, cz_max - 1)
        sy = np.clip(np.round(yc + d_val * ny_m / sp[1]).astype(int), 0, cy_max - 1)
        sx = np.clip(np.round(xc + d_val * nx_m / sp[2]).astype(int), 0, cx_max - 1)
        rv0[:, k] = img_f0[sz, sy, sx]
        rv_lat[:, k] = img_lat[sz, sy, sx]
        rv1[:, k] = img_s1[sz, sy, sx]
        rv_zb[:, k] = z_bone[sz, sy, sx]

    r_in = np.partition(rv_lat[:, 0:6], 1, axis=1)[:, 1:2]
    _s_out = np.partition(rv_lat[:, 5:13], (5, 6), axis=1)
    r_out = 0.05 * _s_out[:, 5:6] + 0.95 * _s_out[:, 6:7]
    r_span = np.maximum(r_out - r_in, 0.15)
    rn0 = (rv0 - r_in) / r_span
    rn_lat = (rv_lat - r_in) / r_span
    rg0 = np.diff(rv0, axis=1)
    rg_lat = np.diff(rv_lat, axis=1)
    rgn_lat = np.diff(rn_lat, axis=1)
    rlap_lat = np.diff(rg_lat, axis=1)
    rg_zb = np.diff(rv_zb, axis=1)

    idx_rise = np.argmax(rg_lat[:, 2:11], axis=1).astype(np.float32) - 4.0
    max_rise = np.max(rg_lat[:, 2:11], axis=1)
    idx_drop = np.argmin(rg_lat[:, 5:14], axis=1).astype(np.float32) - 2.0
    min_drop = np.min(rg_lat[:, 5:14], axis=1)
    idx_cort_min = np.argmin(rv_zb[:, 3:10], axis=1).astype(np.float32) - 4.0
    val_cort_min = np.min(rv_zb[:, 3:10], axis=1)
    idx_rise_zb = np.argmax(rg_zb[:, 2:11], axis=1).astype(np.float32) - 4.0
    max_rise_zb = np.max(rg_zb[:, 2:11], axis=1)

    vy_lat = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
    vy_zb = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
    for k, dy_px in enumerate(_DY_STEPS):
        sy = np.clip(yc + dy_px, 0, cy_max - 1)
        vy_lat[:, k] = img_lat[zc, sy, xc]
        vy_zb[:, k] = z_bone[zc, sy, xc]
    vyn_lat = (vy_lat - r_in) / r_span
    vyg_lat = np.diff(vy_lat, axis=1)

    vh_lat = np.empty((N, len(_H_GRID)), dtype=np.float32)
    vh_zb = np.empty((N, len(_H_GRID)), dtype=np.float32)
    for k, h_val in enumerate(_H_GRID):
        st = h_val - d_s_anchor
        sz = np.clip(np.round(zc + st * nz_m / sp[0]).astype(int), 0, cz_max - 1)
        sy = np.clip(np.round(yc + st * ny_m / sp[1]).astype(int), 0, cy_max - 1)
        sx = np.clip(np.round(xc + st * nx_m / sp[2]).astype(int), 0, cx_max - 1)
        vh_lat[:, k] = img_lat[sz, sy, sx]
        vh_zb[:, k] = z_bone[sz, sy, sx]
    gh_lat = np.diff(vh_lat, axis=1)

    comp = geo['f_comp'] if bone_name == 'femur' else geo['t_comp']
    c1, s1, c2, s2, spine, z_mid = comp
    in_1 = coords_mm[:, 0] < z_mid
    uz = np.where(in_1, (c1[0] - coords_mm[:, 0]) / s1[0], (coords_mm[:, 0] - c2[0]) / s2[0])
    uy = np.where(in_1, (coords_mm[:, 1] - c1[1]) / s1[1], (coords_mm[:, 1] - c2[1]) / s2[1])
    ux = np.where(in_1, (coords_mm[:, 2] - c1[2]) / s1[2], (coords_mm[:, 2] - c2[2]) / s2[2])
    d_zx = np.sqrt(uz**2 + ux**2)
    d_spine_z = np.abs(coords_mm[:, 0] - spine[0])

    dz_rad = coords_mm[:, 0] - np.where(in_1, c1[0], c2[0])
    dx_rad = coords_mm[:, 2] - np.where(in_1, c1[2], c2[2])
    r_norm = np.sqrt(dz_rad**2 + dx_rad**2) + 1e-6
    u_rz = (dz_rad / r_norm).astype(np.float32)
    u_rx = (dx_rad / r_norm).astype(np.float32)
    rad_vlat = np.empty((N, len(_R_STEPS)), dtype=np.float32)
    rad_zb = np.empty((N, len(_R_STEPS)), dtype=np.float32)
    for k, r_mm in enumerate(_R_STEPS):
        sz = np.clip(np.round(zc + r_mm * u_rz / sp[0]).astype(int), 0, cz_max - 1)
        sx = np.clip(np.round(xc + r_mm * u_rx / sp[2]).astype(int), 0, cx_max - 1)
        rad_vlat[:, k] = img_lat[sz, yc, sx]
        rad_zb[:, k] = z_bone[sz, yc, sx]
    rad_diff = np.diff(rad_vlat, axis=1)

    c_min, c_max, _ = geo['f_stats'] if bone_name == 'femur' else geo['t_stats']
    rel_box = (coords_mm - c_min) / (c_max - c_min + 1e-6)
    z_sym = np.abs(rel_box[:, 0] - 0.5)

    v0_m = img_f0[zc, yc, xc]
    vlat_m = img_lat[zc, yc, xc]
    vs1_m = img_s1[zc, yc, xc]
    vs2_m = img_s2[zc, yc, xc]
    vs4_m = img_s4[zc, yc, xc]
    vs8_m = img_s8[zc, yc, xc]
    z_m = z_cart[zc, yc, xc]
    absz_m = abs_z[zc, yc, xc]
    zb_m = z_bone[zc, yc, xc]
    gn0 = gz0[zc, yc, xc] * nz_m + gy0[zc, yc, xc] * ny_m + gx0[zc, yc, xc] * nx_m
    gn1 = gz1[zc, yc, xc] * nz_m + gy1[zc, yc, xc] * ny_m + gx1[zc, yc, xc] * nx_m

    uz_p, uy_p, ux_p = c_dict['uz_p'], c_dict['uy_p'], c_dict['ux_p']
    d_zx_p = np.sqrt(uz_p**2 + ux_p**2)

    base_cols = np.column_stack([
        v0_m, vlat_m, vs1_m, vs2_m, vs4_m, vs8_m, z_m, absz_m, zb_m,
        v0_m - vlat_m, v0_m - vs1_m, vs1_m - vs2_m, vs2_m - vs4_m, v0_m - vs4_m, v0_m - vs8_m,
        gmag0[zc, yc, xc], gmag1[zc, yc, xc], gn0, gn1, gy0[zc, yc, xc], gy1[zc, yc, xc],
        d_raw_feat, d_alt_feat, 0.5 * (d_raw_feat + d_alt_feat), shift_ref_alt, shift_ref_st2, d_s_feat,
        do_raw_m, do_s_m,
        d_s_feat + do_s_m, d_s_feat - do_s_m,
        np.maximum(d_s_feat, 0.0) / (np.maximum(d_s_feat, 0.0) + np.maximum(do_s_m, 0.0) + 0.5),
        nz_m, ny_m, nx_m, nzo_m, nyo_m, nxo_m,
        nz_m * nzo_m + ny_m * nyo_m + nx_m * nxo_m,
        c_dict['is_lat'], c_dict['u_ml'], c_dict['u_ap'], c_dict['u_y'],
        uz_p, uy_p, ux_p, d_zx_p, c_dict['rel_fz_ml'],
        uz, uy, ux, np.abs(uz), d_zx, d_spine_z,
        rel_box[:, 0], rel_box[:, 1], rel_box[:, 2], z_sym,
        pt_ens_tc, pt_ens_fc, pt_old_tc, pt_old_fc, pt_ml_tc, pt_sep_tc, pt_mly_tc, pt_wide_tc, pt_ml_max3,
        pt_ens_tc - pt_ens_fc, pt_ens_tc - pt_old_tc, pt_wide_tc - pt_ml_tc,
        p2d_ens, p2d_wide_v, edt_ens, edt_wide_v, d_s_feat - 2.0 * edt_ens, d_s_feat / (2.0 * edt_ens + 0.5),
        pf_ens_fc, pf_ens_tc, pf_old_fc, pf_old_tc, pf_fml_fc, pf_sep_fc, pf_fml_max3,
        pf_ens_fc - pf_ens_tc, pf_ens_fc - pf_old_fc,
        p3d_fml_v, edf_fml_v, d_s_feat - 2.0 * edf_fml_v, d_s_feat / (2.0 * edf_fml_v + 0.5),
        r_in[:, 0], r_out[:, 0], r_out[:, 0] - r_in[:, 0],
        idx_rise, max_rise, idx_drop, min_drop,
        idx_cort_min, val_cort_min, idx_rise_zb, max_rise_zb,
    ])

    blocks = [
        base_cols,
        rv0, rv_lat, rv1, rv_zb, rn0, rn_lat, rg0, rg_lat, rgn_lat, rlap_lat, rg_zb,
        vy_lat, vyn_lat, vyg_lat, vy_zb,
        vh_lat, gh_lat, vh_zb,
        rad_vlat, rad_diff, rad_zb,
    ]

    if pm_self is not None:
        ps_crop = pm_self[z0:z1, y0:y1, x0:x1].astype(np.float32)
        if pm_other is not None:
            po_crop = pm_other[z0:z1, y0:y1, x0:x1].astype(np.float32)
            p_bg_c = 0.75 * ps_crop[..., 0] + 0.25 * po_crop[..., 0]
            p_bn_c = ps_crop[..., 1]
            p_obn_c = po_crop[..., 1]
            p_ct_c = 0.80 * ps_crop[..., 2] + 0.20 * po_crop[..., 3]
            p_ot_c = 0.60 * ps_crop[..., 3] + 0.40 * po_crop[..., 2]
        else:
            p_bg_c = ps_crop[..., 0]
            p_bn_c = ps_crop[..., 1]
            p_obn_c = np.zeros_like(p_bn_c)
            p_ct_c = ps_crop[..., 2]
            p_ot_c = ps_crop[..., 3]

        sig_tan = [3.6 / sp[0], 1.5 / sp[1], 3.6 / sp[2]]
        sig_bn = [4.5 / sp[0], 2.5 / sp[1], 4.5 / sp[2]]
        w_ct = np.clip(p_ct_c - 0.30, 0.0, 1.0) ** 2 + 1e-5
        w_ct_s = gaussian_filter(w_ct, sigma=sig_tan)
        i_sheet_loc = gaussian_filter(img_f32 * w_ct, sigma=sig_tan) / w_ct_s
        i_var_loc = gaussian_filter(((img_f32 - i_sheet_loc) ** 2) * w_ct, sigma=sig_tan) / w_ct_s
        i_std_loc = np.maximum(np.sqrt(np.maximum(i_var_loc, 0.0)), 4.0)

        w_bn = ((geo['d_self_crop'] <= -1.2) & (geo['d_self_crop'] >= -4.8)).astype(np.float32) + 1e-5
        i_bone_loc = gaussian_filter(img_f32 * w_bn, sigma=sig_bn) / gaussian_filter(w_bn, sigma=sig_bn)
        in_loc_vol = (img_f32 - i_bone_loc) / np.maximum(i_sheet_loc - i_bone_loc, 5.0)
        z_loc_vol = (img_f32 - i_sheet_loc) / i_std_loc
        in_loc_lat = gaussian_filter(in_loc_vol, sigma=[1.2 / sp[0], 0.25 / sp[1], 1.2 / sp[2]])

        p_ct_lat = gaussian_filter(p_ct_c, sigma=[1.4 / sp[0], 0.35 / sp[1], 1.4 / sp[2]])
        p_ct_lat_wide = gaussian_filter(p_ct_c, sigma=[2.8 / sp[0], 0.40 / sp[1], 2.8 / sp[2]])
        p_ct_s1 = gaussian_filter(p_ct_c, sigma=[1.0 / s for s in sp])
        p_ct_s2 = gaussian_filter(p_ct_c, sigma=[2.2 / s for s in sp])
        p_bn_s1 = gaussian_filter(p_bn_c, sigma=[1.0 / s for s in sp])
        p_bg_s1 = gaussian_filter(p_bg_c, sigma=[1.0 / s for s in sp])
        p_ot_s1 = gaussian_filter(p_ot_c, sigma=[1.0 / s for s in sp])

        pgz, pgy, pgx = np.gradient(p_ct_s1, *sp)
        pg_norm = pgz[zc, yc, xc] * nz_m + pgy[zc, yc, xc] * ny_m + pgx[zc, yc, xc] * nx_m

        valid_c = (geo['d_self_crop'] >= -0.6).astype(np.float32)
        c_above_vol = np.cumsum(p_ct_c * valid_c, axis=1) * sp[1]
        t_col_2d = c_above_vol[:, -1:, :]
        c_below_vol = np.maximum(0.0, t_col_2d - c_above_vol)
        frac_above_vol = c_above_vol / (t_col_2d + 0.25)

        w_ny = np.clip(np.abs(ny), 0.20, 1.0)
        c_above_norm = np.cumsum(p_ct_c * valid_c * w_ny, axis=1) * sp[1]
        t_col_norm = c_above_norm[:, -1:, :]
        c_below_norm = np.maximum(0.0, t_col_norm - c_above_norm)

        t_sheet_s2 = gaussian_filter(t_col_2d, sigma=[2.2 / sp[0], 0.0, 2.2 / sp[2]])
        t_sheet_s4 = gaussian_filter(t_col_2d, sigma=[4.5 / sp[0], 0.0, 4.5 / sp[2]])
        pmax_col_2d = np.max(p_ct_c, axis=1, keepdims=True)
        pmax_sheet_s2 = gaussian_filter(pmax_col_2d, sigma=[2.2 / sp[0], 0.0, 2.2 / sp[2]])
        pmax_sheet_s4 = gaussian_filter(pmax_col_2d, sigma=[4.5 / sp[0], 0.0, 4.5 / sp[2]])

        p_ray_steps = np.array([-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.2], dtype=np.float32)
        pr_ct = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        pr_bn = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        pr_bg = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        pr_ot = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        pr_inloc = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        for k, st_val in enumerate(p_ray_steps):
            sz = np.clip(np.round(zc + st_val * nz_m / sp[0]).astype(int), 0, cz_max - 1)
            sy = np.clip(np.round(yc + st_val * ny_m / sp[1]).astype(int), 0, cy_max - 1)
            sx = np.clip(np.round(xc + st_val * nx_m / sp[2]).astype(int), 0, cx_max - 1)
            pr_ct[:, k] = p_ct_lat[sz, sy, sx]
            pr_bn[:, k] = p_bn_c[sz, sy, sx]
            pr_bg[:, k] = p_bg_c[sz, sy, sx]
            pr_ot[:, k] = p_ot_s1[sz, sy, sx]
            pr_inloc[:, k] = in_loc_lat[sz, sy, sx]

        c_in_ray = np.sum(pr_ct[:, 0:5], axis=1) * 0.40
        c_out_ray = np.sum(pr_ct[:, 5:10], axis=1) * 0.42
        t_ray_tot = c_in_ray + c_out_ray
        f_in_ray = c_in_ray / (t_ray_tot + 0.20)

        py_ct = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
        py_bn = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
        for k, dy_px in enumerate(_DY_STEPS):
            sy = np.clip(yc + dy_px, 0, cy_max - 1)
            py_ct[:, k] = p_ct_lat[zc, sy, xc]
            py_bn[:, k] = p_bn_c[zc, sy, xc]

        ph_ct = np.empty((N, len(_H_GRID)), dtype=np.float32)
        ph_ot = np.empty((N, len(_H_GRID)), dtype=np.float32)
        for k, h_val in enumerate(_H_GRID):
            st = h_val - d_s_anchor
            sz = np.clip(np.round(zc + st * nz_m / sp[0]).astype(int), 0, cz_max - 1)
            sy = np.clip(np.round(yc + st * ny_m / sp[1]).astype(int), 0, cy_max - 1)
            sx = np.clip(np.round(xc + st * nx_m / sp[2]).astype(int), 0, cx_max - 1)
            ph_ct[:, k] = p_ct_lat[sz, sy, sx]
            ph_ot[:, k] = p_ot_s1[sz, sy, sx]
        col_psum = np.sum(ph_ct[:, 2:], axis=1) * 0.45
        col_pmax = np.max(ph_ct[:, 2:], axis=1)
        col_otsum = np.sum(ph_ot[:, 2:], axis=1) * 0.45

        t1z = 1.0 - nz_m * nz_m
        t1y = -nz_m * ny_m
        t1x = -nz_m * nx_m
        t1_norm = np.sqrt(t1z**2 + t1y**2 + t1x**2) + 1e-6
        t1z, t1y, t1x = t1z / t1_norm, t1y / t1_norm, t1x / t1_norm
        t2z = ny_m * t1x - nx_m * t1y
        t2y = nx_m * t1z - nz_m * t1x
        t2x = nz_m * t1y - ny_m * t1z
        tan_offsets = [
            (-3.6, 0.0), (-1.8, 0.0), (1.8, 0.0), (3.6, 0.0),
            (0.0, -3.6), (0.0, -1.8), (0.0, 1.8), (0.0, 3.6),
        ]
        ptan_ct = np.empty((N, len(tan_offsets)), dtype=np.float32)
        for k, (a1, a2) in enumerate(tan_offsets):
            sz = np.clip(np.round(zc + (a1 * t1z + a2 * t2z) / sp[0]).astype(int), 0, cz_max - 1)
            sy = np.clip(np.round(yc + (a1 * t1y + a2 * t2y) / sp[1]).astype(int), 0, cy_max - 1)
            sx = np.clip(np.round(xc + (a1 * t1x + a2 * t2x) / sp[2]).astype(int), 0, cx_max - 1)
            ptan_ct[:, k] = p_ct_lat[sz, sy, sx]

        c_ab_m = c_above_vol[zc, yc, xc]
        c_bl_m = c_below_vol[zc, yc, xc]
        f_ab_m = frac_above_vol[zc, yc, xc]
        c_abn_m = c_above_norm[zc, yc, xc]
        c_bln_m = c_below_norm[zc, yc, xc]
        t_col_m = t_col_2d[zc, 0, xc]
        t_sh2_m = t_sheet_s2[zc, 0, xc]
        t_sh4_m = t_sheet_s4[zc, 0, xc]
        pm_col_m = pmax_col_2d[zc, 0, xc]
        pm_sh2_m = pmax_sheet_s2[zc, 0, xc]
        pm_sh4_m = pmax_sheet_s4[zc, 0, xc]

        p_scalar_cols = np.column_stack([
            in_loc_vol[zc, yc, xc], in_loc_lat[zc, yc, xc], z_loc_vol[zc, yc, xc], np.abs(z_loc_vol[zc, yc, xc]),
            ps_crop[zc, yc, xc, 2], p_obn_c[zc, yc, xc],
            p_bg_c[zc, yc, xc], p_bn_c[zc, yc, xc], p_ct_c[zc, yc, xc], p_ot_c[zc, yc, xc],
            p_ct_c[zc, yc, xc] + p_ot_c[zc, yc, xc],
            p_ct_lat[zc, yc, xc], p_ct_lat_wide[zc, yc, xc], p_ct_s1[zc, yc, xc], p_ct_s2[zc, yc, xc],
            p_bn_s1[zc, yc, xc], p_bg_s1[zc, yc, xc], p_ot_s1[zc, yc, xc],
            p_ct_c[zc, yc, xc] - p_bn_c[zc, yc, xc],
            p_ct_c[zc, yc, xc] - p_bg_c[zc, yc, xc],
            p_ct_c[zc, yc, xc] - p_ot_c[zc, yc, xc],
            pg_norm, col_psum, col_pmax, col_otsum,
            d_s_feat - col_psum, d_s_feat / (col_psum + 0.4),
            c_ab_m, c_bl_m, f_ab_m, c_abn_m, c_bln_m,
            c_in_ray, c_out_ray, t_ray_tot, f_in_ray,
            t_col_m, t_sh2_m, t_sh4_m, t_col_m - t_sh2_m, c_bl_m - t_sh2_m,
            pm_col_m, pm_sh2_m, pm_sh4_m, pm_col_m - pm_sh2_m,
        ])
        blocks.extend([p_scalar_cols, pr_ct, pr_bn, pr_bg, pr_ot, pr_inloc, py_ct, py_bn, ph_ct, ph_ot, ptan_ct])

    return np.hstack(blocks).astype(np.float32, copy=False)


def extract_features_tibia_v7(geo, target_mask, pm_self=None, pm_other=None, rng_jit=None):
    z0, z1, y0, y1, x0, x1 = geo['bbox']
    sp = geo['sp']
    inds = np.argwhere(target_mask)
    zc, yc, xc = inds[:, 0] - z0, inds[:, 1] - y0, inds[:, 2] - x0
    coords_mm = inds.astype(np.float32) * np.array(sp, dtype=np.float32)
    N = len(zc)

    d_raw_m = geo['d_self_crop'][zc, yc, xc]
    d_alt_m = geo['d_alt_self_crop'][zc, yc, xc]
    d_st2_m = geo['d_raw_self_crop'][zc, yc, xc]
    d_s_m = geo['ds_self_crop'][zc, yc, xc]
    do_raw_m = geo['d_other_crop'][zc, yc, xc]
    do_s_m = geo['ds_other_crop'][zc, yc, xc]

    shift_ref_alt = d_raw_m - d_alt_m
    shift_ref_st2 = d_raw_m - d_st2_m

    d_raw_feat = d_raw_m
    d_alt_feat = d_alt_m
    d_s_feat = d_s_m
    d_s_anchor = d_s_m

    ap = geo['atlas_p']
    bins, c_dict = compute_atlas_bins_v6(
        coords_mm, do_s_m, d_s_anchor, geo['f_stats'], geo['t_comp'], geo['ml_info']
    )

    pt_old_tc = ap['pt_old_tc'][bins['t_old']]
    pt_old_fc = ap['pt_old_fc'][bins['t_old']]
    pt_ml_tc = ap['pt_ml_tc'][bins['t_ml']]
    pt_ml_fc = ap['pt_ml_fc'][bins['t_ml']]
    pt_mly_tc = ap['pt_mly_tc'][bins['t_mly']]
    pt_sep_tc = ap['pt_sep_tc'][bins['t_sep']]
    pt_sep_fc = ap['pt_sep_fc'][bins['t_sep']]
    pt_wide_tc = ap['pt_wide_tc'][bins['t_wide']]

    pt_ens_tc = 0.32 * pt_ml_tc + 0.30 * pt_sep_tc + 0.20 * pt_mly_tc + 0.18 * pt_wide_tc
    pt_ens_fc = 0.50 * pt_ml_fc + 0.50 * pt_sep_fc
    bdt_m1, bdt_p1 = np.clip(bins['t_ml'][2] - 1, 0, 15), np.clip(bins['t_ml'][2] + 1, 0, 15)
    pt_ml_max3 = np.maximum(
        np.maximum(ap['pt_ml_tc'][bins['t_ml'][0], bins['t_ml'][1], bdt_m1, bins['t_ml'][3]], pt_ml_tc),
        ap['pt_ml_tc'][bins['t_ml'][0], bins['t_ml'][1], bdt_p1, bins['t_ml'][3]],
    )
    p2d_ml_v = ap['p2d_ml_tc'][bins['t_ml'][0], bins['t_ml'][1]]
    edt_ml_v = ap['edt_ml_2d'][bins['t_ml'][0], bins['t_ml'][1]]
    p2d_sep_v = ap['p2d_sep_tc'][bins['t_sep'][0], bins['t_sep'][1], bins['t_sep'][2]]
    edt_sep_v = ap['edt_sep_2d'][bins['t_sep'][0], bins['t_sep'][1], bins['t_sep'][2]]
    p2d_wide_v = ap['p2d_wide_tc'][bins['t_wide'][0], bins['t_wide'][1]]
    edt_wide_v = ap['edt_wide_2d'][bins['t_wide'][0], bins['t_wide'][1]]
    p2d_ens = 0.40 * p2d_ml_v + 0.35 * p2d_sep_v + 0.25 * p2d_wide_v
    edt_ens = 0.40 * edt_ml_v + 0.35 * edt_sep_v + 0.25 * edt_wide_v

    pf_old_fc = ap['pf_old_fc'][bins['f_old']]
    pf_old_tc = ap['pf_old_tc'][bins['f_old']]
    pf_fml_fc = ap['pf_fml_fc'][bins['f_ml']]
    pf_fml_tc = ap['pf_fml_tc'][bins['f_ml']]
    pf_sep_fc = ap['pf_sep_fc'][bins['f_sep']]
    pf_sep_tc = ap['pf_sep_tc'][bins['f_sep']]
    pf_ens_fc = 0.45 * pf_fml_fc + 0.35 * pf_sep_fc + 0.20 * pf_old_fc
    pf_ens_tc = 0.50 * pf_fml_tc + 0.50 * pf_sep_tc

    bdf_m1, bdf_p1 = np.clip(bins['f_ml'][3] - 1, 0, 15), np.clip(bins['f_ml'][3] + 1, 0, 15)
    pf_fml_max3 = np.maximum(
        np.maximum(ap['pf_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2], bdf_m1], pf_fml_fc),
        ap['pf_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2], bdf_p1],
    )
    p3d_fml_v = ap['p3d_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2]]
    edf_fml_v = ap['edf_fml_3d'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2]]

    img_f32 = geo['img_f32']
    i_low = geo['i_low']
    i_high = geo['i_high']
    z_bone = geo['z_bone_crop']
    z_bone_sharp = geo['z_bone_sharp']
    w_high_z = geo['w_high_z']
    comp_masks = geo['comp_masks']
    i_low_c, i_high_c, i_med_c, iqr_c, zb_med_c = [list(x) for x in geo['comp_stats']]
    zb_med_glob = geo['zb_med_glob']

    if pm_self is not None:
        pm_crop = pm_self[z0:z1, y0:y1, x0:x1]
        m_p1_core = pm_crop[..., 2] >= 0.52
        if np.sum(m_p1_core) >= 50:
            i_high = float(np.median(img_f32[m_p1_core]))
            zb_med_glob = max(float(np.median(z_bone_sharp[m_p1_core])), 1.0)
        for c_i, cm in enumerate(comp_masks):
            mc = m_p1_core & cm
            if np.sum(mc) >= 25:
                vh = img_f32[mc]
                i_high_c[c_i] = float(np.percentile(vh, 65))
                i_med_c[c_i] = float(np.median(vh))
                iqr_c[c_i] = max(float(np.percentile(vh, 75) - np.percentile(vh, 25)), 1.0)
                zb_med_c[c_i] = max(float(np.median(z_bone_sharp[mc])), 1.0)

    if '_img_f0_0' not in geo:
        inorm_0 = geo['inorm']
        geo['_base_span'] = max(geo['i_high'] - geo['i_low'], 1.0)
        geo['_abs_z'] = gaussian_filter(np.abs(geo['z_cart']), sigma=[1.0 / sp[0], 0.35 / sp[1], 1.0 / sp[2]])
        geo['_img_f0_0'] = gaussian_filter(inorm_0, sigma=[0.35 / s for s in sp])
        geo['_img_s06_0'] = gaussian_filter(inorm_0, sigma=[0.65 / s for s in sp])
        geo['_img_lat_0'] = gaussian_filter(inorm_0, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
        geo['_img_s1_0'] = gaussian_filter(inorm_0, sigma=[1.0 / s for s in sp])
        geo['_img_s2_0'] = gaussian_filter(inorm_0, sigma=[2.0 / s for s in sp])
        geo['_img_s4_0'] = gaussian_filter(inorm_0, sigma=[4.0 / s for s in sp])
        geo['_img_s8_0'] = gaussian_filter(inorm_0, sigma=[8.0 / s for s in sp])
        gz0_0, gy0_0, gx0_0 = np.gradient(geo['_img_f0_0'], *sp)
        geo['_g0_0'] = (gz0_0, gy0_0, gx0_0, np.sqrt(gz0_0**2 + gy0_0**2 + gx0_0**2))
        gz1_0, gy1_0, gx1_0 = np.gradient(geo['_img_s1_0'], *sp)
        geo['_g1_0'] = (gz1_0, gy1_0, gx1_0, np.sqrt(gz1_0**2 + gy1_0**2 + gx1_0**2))
    elif '_img_s06_0' not in geo:
        geo['_img_s06_0'] = gaussian_filter(geo['inorm'], sigma=[0.65 / s for s in sp])

    scale = float(geo['_base_span'] / max(i_high - i_low, 1.0))
    z_cart = geo['z_cart']
    abs_z = geo['_abs_z']

    i_low_2d = (1.0 - w_high_z) * i_low_c[0] + w_high_z * i_low_c[1]
    i_high_2d = (1.0 - w_high_z) * i_high_c[0] + w_high_z * i_high_c[1]
    i_med_2d = (1.0 - w_high_z) * i_med_c[0] + w_high_z * i_med_c[1]
    iqr_2d = (1.0 - w_high_z) * iqr_c[0] + w_high_z * iqr_c[1]
    zb_med_2d = (1.0 - w_high_z) * zb_med_c[0] + w_high_z * zb_med_c[1]

    inorm_comp = ((img_f32 - i_low_2d) / np.maximum(i_high_2d - i_low_2d, 1.0)).astype(np.float32)
    z_cart_comp = ((img_f32 - i_med_2d) / np.maximum(iqr_2d, 1.0)).astype(np.float32)
    zb_norm_glob = (z_bone_sharp / zb_med_glob).astype(np.float32)
    zb_norm_comp = (z_bone_sharp / np.maximum(zb_med_2d, 1.0)).astype(np.float32)

    if abs(scale - 1.0) < 1e-7:
        img_f0 = geo['_img_f0_0']
        img_s06 = geo['_img_s06_0']
        img_lat = geo['_img_lat_0']
        img_s1 = geo['_img_s1_0']
        img_s2 = geo['_img_s2_0']
        img_s4 = geo['_img_s4_0']
        img_s8 = geo['_img_s8_0']
        gz0, gy0, gx0, gmag0 = geo['_g0_0']
        gz1, gy1, gx1, gmag1 = geo['_g1_0']
    else:
        img_f0 = geo['_img_f0_0'] * scale
        img_s06 = geo['_img_s06_0'] * scale
        img_lat = geo['_img_lat_0'] * scale
        img_s1 = geo['_img_s1_0'] * scale
        img_s2 = geo['_img_s2_0'] * scale
        img_s4 = geo['_img_s4_0'] * scale
        img_s8 = geo['_img_s8_0'] * scale
        gz0_0, gy0_0, gx0_0, gmag0_0 = geo['_g0_0']
        gz1_0, gy1_0, gx1_0, gmag1_0 = geo['_g1_0']
        gz0, gy0, gx0, gmag0 = gz0_0 * scale, gy0_0 * scale, gx0_0 * scale, gmag0_0 * scale
        gz1, gy1, gx1, gmag1 = gz1_0 * scale, gy1_0 * scale, gx1_0 * scale, gmag1_0 * scale

    icomp_f0 = gaussian_filter(inorm_comp, sigma=[0.35 / s for s in sp])
    icomp_s06 = gaussian_filter(inorm_comp, sigma=[0.65 / s for s in sp])
    gzb, gyb, gxb = np.gradient(zb_norm_comp, *sp)
    gmag_zb = np.sqrt(gzb**2 + gyb**2 + gxb**2)

    nz, ny, nx = geo['normals_self']
    nzo, nyo, nxo = geo['normals_other']
    nz_m, ny_m, nx_m = nz[zc, yc, xc], ny[zc, yc, xc], nx[zc, yc, xc]
    nzo_m, nyo_m, nxo_m = nzo[zc, yc, xc], nyo[zc, yc, xc], nxo[zc, yc, xc]
    cz_max, cy_max, cx_max = img_f32.shape

    K = len(_D_GRID)
    rv0 = np.empty((N, K), dtype=np.float32)
    rv_s06 = np.empty((N, K), dtype=np.float32)
    rv_lat = np.empty((N, K), dtype=np.float32)
    rv1 = np.empty((N, K), dtype=np.float32)
    rv_zb = np.empty((N, K), dtype=np.float32)
    rv_zbn = np.empty((N, K), dtype=np.float32)
    for k, d_val in enumerate(_D_GRID):
        sz = np.clip(np.round(zc + d_val * nz_m / sp[0]).astype(int), 0, cz_max - 1)
        sy = np.clip(np.round(yc + d_val * ny_m / sp[1]).astype(int), 0, cy_max - 1)
        sx = np.clip(np.round(xc + d_val * nx_m / sp[2]).astype(int), 0, cx_max - 1)
        rv0[:, k] = img_f0[sz, sy, sx]
        rv_s06[:, k] = icomp_s06[sz, sy, sx]
        rv_lat[:, k] = img_lat[sz, sy, sx]
        rv1[:, k] = img_s1[sz, sy, sx]
        rv_zb[:, k] = z_bone[sz, sy, sx]
        rv_zbn[:, k] = zb_norm_comp[sz, sy, sx]

    r_in = np.partition(rv_lat[:, 0:6], 1, axis=1)[:, 1:2]
    _s_out = np.partition(rv_lat[:, 5:13], (5, 6), axis=1)
    r_out = 0.05 * _s_out[:, 5:6] + 0.95 * _s_out[:, 6:7]
    r_span = np.maximum(r_out - r_in, 0.15)
    rn0 = (rv0 - r_in) / r_span
    rn_lat = (rv_lat - r_in) / r_span
    rg0 = np.diff(rv0, axis=1)
    rg_s06 = np.diff(rv_s06, axis=1)
    rg_lat = np.diff(rv_lat, axis=1)
    rgn_lat = np.diff(rn_lat, axis=1)
    rlap_lat = np.diff(rg_lat, axis=1)
    rg_zbn = np.diff(rv_zbn, axis=1)

    idx_rise = np.argmax(rg_lat[:, 2:11], axis=1).astype(np.float32) - 4.0
    max_rise = np.max(rg_lat[:, 2:11], axis=1)
    idx_drop = np.argmin(rg_lat[:, 5:14], axis=1).astype(np.float32) - 2.0
    min_drop = np.min(rg_lat[:, 5:14], axis=1)

    idx_rise_0 = np.argmax(rg0[:, 2:11], axis=1).astype(np.float32) - 4.0
    max_rise_0 = np.max(rg0[:, 2:11], axis=1)
    idx_drop_0 = np.argmin(rg0[:, 5:14], axis=1).astype(np.float32) - 2.0
    min_drop_0 = np.min(rg0[:, 5:14], axis=1)
    idx_cort_min = np.argmin(rv_zbn[:, 3:10], axis=1).astype(np.float32) - 4.0
    val_cort_min = np.min(rv_zbn[:, 3:10], axis=1)
    idx_rise_zb = np.argmax(rg_zbn[:, 2:11], axis=1).astype(np.float32) - 4.0
    max_rise_zb = np.max(rg_zbn[:, 2:11], axis=1)
    idx_drop_zb = np.argmin(rg_zbn[:, 5:14], axis=1).astype(np.float32) - 2.0
    min_drop_zb = np.min(rg_zbn[:, 5:14], axis=1)

    vy_0 = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
    vy_lat = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
    vy_zb = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
    vy_zbn = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
    for k, dy_px in enumerate(_DY_STEPS):
        sy = np.clip(yc + dy_px, 0, cy_max - 1)
        vy_0[:, k] = icomp_f0[zc, sy, xc]
        vy_lat[:, k] = img_lat[zc, sy, xc]
        vy_zb[:, k] = z_bone[zc, sy, xc]
        vy_zbn[:, k] = zb_norm_comp[zc, sy, xc]
    vyn_lat = (vy_lat - r_in) / r_span
    vyg_0 = np.diff(vy_0, axis=1)
    vyg_lat = np.diff(vy_lat, axis=1)
    vyg_zbn = np.diff(vy_zbn, axis=1)

    vh_0 = np.empty((N, len(_H_GRID)), dtype=np.float32)
    vh_lat = np.empty((N, len(_H_GRID)), dtype=np.float32)
    vh_zbn = np.empty((N, len(_H_GRID)), dtype=np.float32)
    for k, h_val in enumerate(_H_GRID):
        st = h_val - d_s_anchor
        sz = np.clip(np.round(zc + st * nz_m / sp[0]).astype(int), 0, cz_max - 1)
        sy = np.clip(np.round(yc + st * ny_m / sp[1]).astype(int), 0, cy_max - 1)
        sx = np.clip(np.round(xc + st * nx_m / sp[2]).astype(int), 0, cx_max - 1)
        vh_0[:, k] = icomp_f0[sz, sy, sx]
        vh_lat[:, k] = img_lat[sz, sy, sx]
        vh_zbn[:, k] = zb_norm_comp[sz, sy, sx]
    gh_lat = np.diff(vh_lat, axis=1)
    gh_0 = np.diff(vh_0, axis=1)

    comp = geo['t_comp']
    c1, s1, c2, s2, spine, z_mid = comp
    in_1 = coords_mm[:, 0] < z_mid
    uz = np.where(in_1, (c1[0] - coords_mm[:, 0]) / s1[0], (coords_mm[:, 0] - c2[0]) / s2[0])
    uy = np.where(in_1, (coords_mm[:, 1] - c1[1]) / s1[1], (coords_mm[:, 1] - c2[1]) / s2[1])
    ux = np.where(in_1, (coords_mm[:, 2] - c1[2]) / s1[2], (coords_mm[:, 2] - c2[2]) / s2[2])
    d_zx = np.sqrt(uz**2 + ux**2)
    d_spine_z = np.abs(coords_mm[:, 0] - spine[0])

    dz_rad = coords_mm[:, 0] - np.where(in_1, c1[0], c2[0])
    dx_rad = coords_mm[:, 2] - np.where(in_1, c1[2], c2[2])
    r_norm = np.sqrt(dz_rad**2 + dx_rad**2) + 1e-6
    u_rz = (dz_rad / r_norm).astype(np.float32)
    u_rx = (dx_rad / r_norm).astype(np.float32)
    rad_vlat = np.empty((N, len(_R_STEPS)), dtype=np.float32)
    rad_zbn = np.empty((N, len(_R_STEPS)), dtype=np.float32)
    for k, r_mm in enumerate(_R_STEPS):
        sz = np.clip(np.round(zc + r_mm * u_rz / sp[0]).astype(int), 0, cz_max - 1)
        sx = np.clip(np.round(xc + r_mm * u_rx / sp[2]).astype(int), 0, cx_max - 1)
        rad_vlat[:, k] = img_lat[sz, yc, sx]
        rad_zbn[:, k] = zb_norm_comp[sz, yc, sx]
    rad_diff = np.diff(rad_vlat, axis=1)

    c_min, c_max, _ = geo['t_stats']
    rel_box = (coords_mm - c_min) / (c_max - c_min + 1e-6)
    z_sym = np.abs(rel_box[:, 0] - 0.5)

    v0_m = img_f0[zc, yc, xc]
    vs06_m = img_s06[zc, yc, xc]
    vlat_m = img_lat[zc, yc, xc]
    vs1_m = img_s1[zc, yc, xc]
    vs2_m = img_s2[zc, yc, xc]
    vs4_m = img_s4[zc, yc, xc]
    vs8_m = img_s8[zc, yc, xc]
    ic0_m = icomp_f0[zc, yc, xc]
    ics06_m = icomp_s06[zc, yc, xc]
    z_m = z_cart[zc, yc, xc]
    zc_comp_m = z_cart_comp[zc, yc, xc]
    absz_m = abs_z[zc, yc, xc]
    zb_m = z_bone[zc, yc, xc]
    zbs_m = z_bone_sharp[zc, yc, xc]
    zbng_m = zb_norm_glob[zc, yc, xc]
    zbnc_m = zb_norm_comp[zc, yc, xc]

    gn0 = gz0[zc, yc, xc] * nz_m + gy0[zc, yc, xc] * ny_m + gx0[zc, yc, xc] * nx_m
    gn1 = gz1[zc, yc, xc] * nz_m + gy1[zc, yc, xc] * ny_m + gx1[zc, yc, xc] * nx_m
    gn_zb = gzb[zc, yc, xc] * nz_m + gyb[zc, yc, xc] * ny_m + gxb[zc, yc, xc] * nx_m

    uz_p, uy_p, ux_p = c_dict['uz_p'], c_dict['uy_p'], c_dict['ux_p']
    d_zx_p = np.sqrt(uz_p**2 + ux_p**2)

    base_cols = np.column_stack([
        v0_m, vs06_m, vlat_m, vs1_m, vs2_m, vs4_m, vs8_m,
        ic0_m, ics06_m, z_m, zc_comp_m, absz_m, zb_m, zbs_m, zbng_m, zbnc_m,
        v0_m - vlat_m, v0_m - vs1_m, vs1_m - vs2_m, vs2_m - vs4_m, v0_m - vs4_m, v0_m - vs8_m,
        ic0_m - ics06_m, zbnc_m - zbng_m,
        gmag0[zc, yc, xc], gmag1[zc, yc, xc], gmag_zb[zc, yc, xc],
        gn0, gn1, gn_zb, gy0[zc, yc, xc], gy1[zc, yc, xc], gyb[zc, yc, xc],
        d_raw_feat, d_alt_feat, 0.5 * (d_raw_feat + d_alt_feat), shift_ref_alt, shift_ref_st2, d_s_feat,
        do_raw_m, do_s_m,
        d_s_feat + do_s_m, d_s_feat - do_s_m,
        np.maximum(d_s_feat, 0.0) / (np.maximum(d_s_feat, 0.0) + np.maximum(do_s_m, 0.0) + 0.5),
        nz_m, ny_m, nx_m, nzo_m, nyo_m, nxo_m,
        nz_m * nzo_m + ny_m * nyo_m + nx_m * nxo_m,
        c_dict['is_lat'], c_dict['u_ml'], c_dict['u_ap'], c_dict['u_y'],
        uz_p, uy_p, ux_p, d_zx_p, c_dict['rel_fz_ml'],
        uz, uy, ux, np.abs(uz), d_zx, d_spine_z,
        rel_box[:, 0], rel_box[:, 1], rel_box[:, 2], z_sym,
        pt_ens_tc, pt_ens_fc, pt_old_tc, pt_old_fc, pt_ml_tc, pt_sep_tc, pt_mly_tc, pt_wide_tc, pt_ml_max3,
        pt_ens_tc - pt_ens_fc, pt_ens_tc - pt_old_tc, pt_wide_tc - pt_ml_tc,
        p2d_ens, p2d_wide_v, edt_ens, edt_wide_v, d_s_feat - 2.0 * edt_ens, d_s_feat / (2.0 * edt_ens + 0.5),
        pf_ens_fc, pf_ens_tc, pf_old_fc, pf_old_tc, pf_fml_fc, pf_sep_fc, pf_fml_max3,
        pf_ens_fc - pf_ens_tc, pf_ens_fc - pf_old_fc,
        p3d_fml_v, edf_fml_v, d_s_feat - 2.0 * edf_fml_v, d_s_feat / (2.0 * edf_fml_v + 0.5),
        r_out[:, 0] - r_in[:, 0],
        idx_rise, max_rise, idx_drop, min_drop,
        idx_rise_0, max_rise_0, idx_drop_0, min_drop_0,
        idx_cort_min, val_cort_min, idx_rise_zb, max_rise_zb, idx_drop_zb, min_drop_zb,
    ])

    blocks = [
        base_cols,
        rv0, rv_s06, rv_lat, rv1, rv_zbn, rn0, rn_lat, rg0, rg_s06, rg_lat, rgn_lat, rlap_lat, rg_zbn,
        vy_0, vy_lat, vyn_lat, vyg_0, vyg_lat, vy_zbn, vyg_zbn,
        vh_0, vh_lat, gh_0, gh_lat, vh_zbn,
        rad_vlat, rad_diff, rad_zbn,
    ]

    if pm_self is not None:
        ps_crop = pm_self[z0:z1, y0:y1, x0:x1].astype(np.float32)
        if pm_other is not None:
            po_crop = pm_other[z0:z1, y0:y1, x0:x1].astype(np.float32)
            p_bg_c = 0.75 * ps_crop[..., 0] + 0.25 * po_crop[..., 0]
            p_bn_c = ps_crop[..., 1]
            p_obn_c = po_crop[..., 1]
            p_ct_c = 0.80 * ps_crop[..., 2] + 0.20 * po_crop[..., 3]
            p_ot_c = 0.60 * ps_crop[..., 3] + 0.40 * po_crop[..., 2]
        else:
            p_bg_c = ps_crop[..., 0]
            p_bn_c = ps_crop[..., 1]
            p_obn_c = np.zeros_like(p_bn_c)
            p_ct_c = ps_crop[..., 2]
            p_ot_c = ps_crop[..., 3]

        zc_p, yc_p, xc_p = zc, yc, xc

        sig_tan = [3.6 / sp[0], 1.5 / sp[1], 3.6 / sp[2]]
        sig_bn = [4.5 / sp[0], 2.5 / sp[1], 4.5 / sp[2]]
        w_ct = np.clip(p_ct_c - 0.30, 0.0, 1.0) ** 2 + 1e-5
        w_ct_s = gaussian_filter(w_ct, sigma=sig_tan)
        i_sheet_loc = gaussian_filter(img_f32 * w_ct, sigma=sig_tan) / w_ct_s
        i_var_loc = gaussian_filter(((img_f32 - i_sheet_loc) ** 2) * w_ct, sigma=sig_tan) / w_ct_s
        i_std_loc = np.maximum(np.sqrt(np.maximum(i_var_loc, 0.0)), 4.0)

        w_bn = ((geo['d_self_crop'] <= -1.2) & (geo['d_self_crop'] >= -4.8)).astype(np.float32) + 1e-5
        i_bone_loc = gaussian_filter(img_f32 * w_bn, sigma=sig_bn) / gaussian_filter(w_bn, sigma=sig_bn)
        in_loc_vol = (img_f32 - i_bone_loc) / np.maximum(i_sheet_loc - i_bone_loc, 5.0)
        z_loc_vol = (img_f32 - i_sheet_loc) / i_std_loc
        in_loc_lat = gaussian_filter(in_loc_vol, sigma=[1.0 / sp[0], 0.25 / sp[1], 1.0 / sp[2]])

        p_ct_sharp = gaussian_filter(p_ct_c, sigma=[0.45 / s for s in sp])
        p_ct_lat = gaussian_filter(p_ct_c, sigma=[1.4 / sp[0], 0.35 / sp[1], 1.4 / sp[2]])
        p_ct_lat_wide = gaussian_filter(p_ct_c, sigma=[2.8 / sp[0], 0.40 / sp[1], 2.8 / sp[2]])
        p_ct_s1 = gaussian_filter(p_ct_c, sigma=[1.0 / s for s in sp])
        p_ct_s2 = gaussian_filter(p_ct_c, sigma=[2.2 / s for s in sp])
        p_bn_s1 = gaussian_filter(p_bn_c, sigma=[1.0 / s for s in sp])
        p_bg_s1 = gaussian_filter(p_bg_c, sigma=[1.0 / s for s in sp])
        p_ot_s1 = gaussian_filter(p_ot_c, sigma=[1.0 / s for s in sp])

        pgz, pgy, pgx = np.gradient(p_ct_s1, *sp)
        pg_norm = pgz[zc_p, yc_p, xc_p] * nz_m + pgy[zc_p, yc_p, xc_p] * ny_m + pgx[zc_p, yc_p, xc_p] * nx_m

        valid_c = (geo['d_self_crop'] >= -0.6).astype(np.float32)
        c_above_vol = np.cumsum(p_ct_c * valid_c, axis=1) * sp[1]
        t_col_2d = c_above_vol[:, -1:, :]
        c_below_vol = np.maximum(0.0, t_col_2d - c_above_vol)
        frac_above_vol = c_above_vol / (t_col_2d + 0.25)

        w_ny = np.clip(np.abs(ny), 0.20, 1.0)
        c_above_norm = np.cumsum(p_ct_c * valid_c * w_ny, axis=1) * sp[1]
        t_col_norm = c_above_norm[:, -1:, :]
        c_below_norm = np.maximum(0.0, t_col_norm - c_above_norm)

        t_sheet_s2 = gaussian_filter(t_col_2d, sigma=[2.2 / sp[0], 0.0, 2.2 / sp[2]])
        t_sheet_s4 = gaussian_filter(t_col_2d, sigma=[4.5 / sp[0], 0.0, 4.5 / sp[2]])
        pmax_col_2d = np.max(p_ct_c, axis=1, keepdims=True)
        pmax_sheet_s2 = gaussian_filter(pmax_col_2d, sigma=[2.2 / sp[0], 0.0, 2.2 / sp[2]])
        pmax_sheet_s4 = gaussian_filter(pmax_col_2d, sigma=[4.5 / sp[0], 0.0, 4.5 / sp[2]])

        w_col = np.maximum(p_ct_c * valid_c - 0.12, 0.0)
        w_sum_2d = np.sum(w_col, axis=1) + 1e-4
        yy_col = np.arange(cy_max, dtype=np.float32)[None, :, None]
        y_p1_mean = np.sum(w_col * yy_col, axis=1) / w_sum_2d
        y_p1_var = np.sum(w_col * ((yy_col - y_p1_mean[:, None, :]) ** 2), axis=1) / w_sum_2d
        y_p1_std = np.sqrt(np.maximum(y_p1_var, 0.25))
        w_sh = (pmax_col_2d[:, 0, :] >= 0.30).astype(np.float32) + 1e-4
        w_sh_s2 = gaussian_filter(w_sh, sigma=[2.2 / sp[0], 2.2 / sp[2]])
        y_p1_mean_s2 = gaussian_filter(y_p1_mean * w_sh, sigma=[2.2 / sp[0], 2.2 / sp[2]]) / w_sh_s2
        y_p1_std_s2 = gaussian_filter(y_p1_std * w_sh, sigma=[2.2 / sp[0], 2.2 / sp[2]]) / w_sh_s2

        m_t1_40 = pmax_col_2d[:, 0, :] >= 0.40
        m_t1_28 = pmax_col_2d[:, 0, :] >= 0.28
        sd_t1_40 = (
            fast_signed_distance_transform(m_t1_40, (sp[0], sp[2]))
            if (np.any(m_t1_40) and not np.all(m_t1_40))
            else np.zeros_like(y_p1_mean)
        )
        sd_t1_28 = (
            fast_signed_distance_transform(m_t1_28, (sp[0], sp[2]))
            if (np.any(m_t1_28) and not np.all(m_t1_28))
            else np.zeros_like(y_p1_mean)
        )

        if pm_other is not None:
            pf_full_zx = pm_other[z0:z1, :, x0:x1, 2].astype(np.float32)
            pf_col_max = np.max(pf_full_zx, axis=1)
            pf_col_sum = np.sum(pf_full_zx, axis=1) * sp[1]
            pf_col_s2 = gaussian_filter(pf_col_max, sigma=[2.0 / sp[0], 2.0 / sp[2]])
            pf_col_s4 = gaussian_filter(pf_col_max, sigma=[4.0 / sp[0], 4.0 / sp[2]])
            pf_sum_s2 = gaussian_filter(pf_col_sum, sigma=[2.0 / sp[0], 2.0 / sp[2]])
            has_fc_col = pf_col_max >= 0.30
            y_from_bot = np.argmax(pf_full_zx[:, ::-1, :] >= 0.30, axis=1)
            y_fc_bot_2d = np.where(has_fc_col, (pf_full_zx.shape[1] - 1) - y_from_bot, -20).astype(np.float32)
            m_fc_foot = pf_col_s2 >= 0.30
            sd_fc_foot = (
                fast_signed_distance_transform(m_fc_foot, (sp[0], sp[2]))
                if (np.any(m_fc_foot) and not np.all(m_fc_foot))
                else np.zeros_like(pf_col_s2)
            )
        else:
            pf_col_max = np.zeros((cz_max, cx_max), dtype=np.float32)
            pf_col_sum = np.zeros((cz_max, cx_max), dtype=np.float32)
            pf_col_s2 = np.zeros((cz_max, cx_max), dtype=np.float32)
            pf_col_s4 = np.zeros((cz_max, cx_max), dtype=np.float32)
            pf_sum_s2 = np.zeros((cz_max, cx_max), dtype=np.float32)
            has_fc_col = np.zeros((cz_max, cx_max), dtype=bool)
            y_fc_bot_2d = np.full((cz_max, cx_max), -20.0, dtype=np.float32)
            sd_fc_foot = np.zeros((cz_max, cx_max), dtype=np.float32)

        y_glob_p = yc_p + y0
        dy_to_fc_mm = np.where(
            has_fc_col[zc_p, xc_p], (y_glob_p - y_fc_bot_2d[zc_p, xc_p]) * sp[1], 22.0
        ).astype(np.float32)

        p_ray_steps = np.array([-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.2], dtype=np.float32)
        pr_ct = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        pr_ct_sh = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        pr_bn = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        pr_bg = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        pr_ot = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        pr_inloc = np.empty((N, len(p_ray_steps)), dtype=np.float32)
        for k, st_val in enumerate(p_ray_steps):
            sz_p = np.clip(np.round(zc_p + st_val * nz_m / sp[0]).astype(int), 0, cz_max - 1)
            sy_p = np.clip(np.round(yc_p + st_val * ny_m / sp[1]).astype(int), 0, cy_max - 1)
            sx_p = np.clip(np.round(xc_p + st_val * nx_m / sp[2]).astype(int), 0, cx_max - 1)
            pr_ct[:, k] = p_ct_lat[sz_p, sy_p, sx_p]
            pr_ct_sh[:, k] = p_ct_sharp[sz_p, sy_p, sx_p]
            pr_bn[:, k] = p_bn_c[sz_p, sy_p, sx_p]
            pr_bg[:, k] = p_bg_c[sz_p, sy_p, sx_p]
            pr_ot[:, k] = p_ot_s1[sz_p, sy_p, sx_p]
            pr_inloc[:, k] = in_loc_lat[sz_p, sy_p, sx_p]

        c_in_ray = np.sum(pr_ct[:, 0:5], axis=1) * 0.40
        c_out_ray = np.sum(pr_ct[:, 5:10], axis=1) * 0.42
        t_ray_tot = c_in_ray + c_out_ray
        f_in_ray = c_in_ray / (t_ray_tot + 0.20)

        c_in_ray_sh = np.sum(pr_ct_sh[:, 0:5], axis=1) * 0.40
        c_out_ray_sh = np.sum(pr_ct_sh[:, 5:10], axis=1) * 0.42
        t_ray_tot_sh = c_in_ray_sh + c_out_ray_sh
        f_in_ray_sh = c_in_ray_sh / (t_ray_tot_sh + 0.20)

        py_ct = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
        py_bn = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
        for k, dy_px in enumerate(_DY_STEPS):
            sy_p = np.clip(yc_p + dy_px, 0, cy_max - 1)
            py_ct[:, k] = p_ct_lat[zc_p, sy_p, xc_p]
            py_bn[:, k] = p_bn_c[zc_p, sy_p, xc_p]

        ph_ct = np.empty((N, len(_H_GRID)), dtype=np.float32)
        ph_ot = np.empty((N, len(_H_GRID)), dtype=np.float32)
        for k, h_val in enumerate(_H_GRID):
            st = h_val - d_s_m
            sz_p = np.clip(np.round(zc + st * nz_m / sp[0]).astype(int), 0, cz_max - 1)
            sy_p = np.clip(np.round(yc + st * ny_m / sp[1]).astype(int), 0, cy_max - 1)
            sx_p = np.clip(np.round(xc + st * nx_m / sp[2]).astype(int), 0, cx_max - 1)
            ph_ct[:, k] = p_ct_lat[sz_p, sy_p, sx_p]
            ph_ot[:, k] = p_ot_s1[sz_p, sy_p, sx_p]
        col_psum = np.sum(ph_ct[:, 2:], axis=1) * 0.45
        col_pmax = np.max(ph_ct[:, 2:], axis=1)
        col_otsum = np.sum(ph_ot[:, 2:], axis=1) * 0.45

        t1z = 1.0 - nz_m * nz_m
        t1y = -nz_m * ny_m
        t1x = -nz_m * nx_m
        t1_norm = np.sqrt(t1z**2 + t1y**2 + t1x**2) + 1e-6
        t1z, t1y, t1x = t1z / t1_norm, t1y / t1_norm, t1x / t1_norm
        t2z = ny_m * t1x - nx_m * t1y
        t2y = nx_m * t1z - nz_m * t1x
        t2x = nz_m * t1y - ny_m * t1z
        tan_offsets = [
            (-3.6, 0.0), (-1.8, 0.0), (1.8, 0.0), (3.6, 0.0),
            (0.0, -3.6), (0.0, -1.8), (0.0, 1.8), (0.0, 3.6),
        ]
        ptan_ct = np.empty((N, len(tan_offsets)), dtype=np.float32)
        for k, (a1, a2) in enumerate(tan_offsets):
            sz_p = np.clip(np.round(zc_p + (a1 * t1z + a2 * t2z) / sp[0]).astype(int), 0, cz_max - 1)
            sy_p = np.clip(np.round(yc_p + (a1 * t1y + a2 * t2y) / sp[1]).astype(int), 0, cy_max - 1)
            sx_p = np.clip(np.round(xc_p + (a1 * t1x + a2 * t2x) / sp[2]).astype(int), 0, cx_max - 1)
            ptan_ct[:, k] = p_ct_lat[sz_p, sy_p, sx_p]

        c_ab_m = c_above_vol[zc_p, yc_p, xc_p]
        c_bl_m = c_below_vol[zc_p, yc_p, xc_p]
        f_ab_m = frac_above_vol[zc_p, yc_p, xc_p]
        c_abn_m = c_above_norm[zc_p, yc_p, xc_p]
        c_bln_m = c_below_norm[zc_p, yc_p, xc_p]
        t_col_m = t_col_2d[zc_p, 0, xc_p]
        t_sh2_m = t_sheet_s2[zc_p, 0, xc_p]
        t_sh4_m = t_sheet_s4[zc_p, 0, xc_p]
        pm_col_m = pmax_col_2d[zc_p, 0, xc_p]
        pm_sh2_m = pmax_sheet_s2[zc_p, 0, xc_p]
        pm_sh4_m = pmax_sheet_s4[zc_p, 0, xc_p]

        dy_p1_mean_mm = (yc_p - y_p1_mean[zc_p, xc_p]) * sp[1]
        dy_p1_mean_s2_mm = (yc_p - y_p1_mean_s2[zc_p, xc_p]) * sp[1]
        z_y_p1 = (yc_p - y_p1_mean_s2[zc_p, xc_p]) / np.maximum(y_p1_std_s2[zc_p, xc_p], 0.8)

        p_scalar_cols = np.column_stack([
            in_loc_vol[zc, yc, xc], in_loc_lat[zc, yc, xc], z_loc_vol[zc, yc, xc], np.abs(z_loc_vol[zc, yc, xc]),
            ps_crop[zc_p, yc_p, xc_p, 2], p_obn_c[zc_p, yc_p, xc_p],
            p_bg_c[zc_p, yc_p, xc_p], p_bn_c[zc_p, yc_p, xc_p], p_ct_c[zc_p, yc_p, xc_p], p_ot_c[zc_p, yc_p, xc_p],
            p_ct_sharp[zc_p, yc_p, xc_p],
            p_ct_c[zc_p, yc_p, xc_p] + p_ot_c[zc_p, yc_p, xc_p],
            p_ct_lat[zc_p, yc_p, xc_p], p_ct_lat_wide[zc_p, yc_p, xc_p], p_ct_s1[zc_p, yc_p, xc_p], p_ct_s2[zc_p, yc_p, xc_p],
            p_bn_s1[zc_p, yc_p, xc_p], p_bg_s1[zc_p, yc_p, xc_p], p_ot_s1[zc_p, yc_p, xc_p],
            p_ct_c[zc_p, yc_p, xc_p] - p_bn_c[zc_p, yc_p, xc_p],
            p_ct_c[zc_p, yc_p, xc_p] - p_bg_c[zc_p, yc_p, xc_p],
            p_ct_c[zc_p, yc_p, xc_p] - p_ot_c[zc_p, yc_p, xc_p],
            pg_norm, col_psum, col_pmax, col_otsum,
            d_s_feat - col_psum, d_s_feat / (col_psum + 0.4),
            c_ab_m, c_bl_m, f_ab_m, c_abn_m, c_bln_m,
            c_in_ray, c_out_ray, t_ray_tot, f_in_ray,
            c_in_ray_sh, c_out_ray_sh, t_ray_tot_sh, f_in_ray_sh,
            t_col_m, t_sh2_m, t_sh4_m, t_col_m - t_sh2_m, c_bl_m - t_sh2_m,
            pm_col_m, pm_sh2_m, pm_sh4_m, pm_col_m - pm_sh2_m,
            dy_p1_mean_mm, dy_p1_mean_s2_mm, z_y_p1,
            sd_t1_40[zc_p, xc_p], sd_t1_28[zc_p, xc_p],
            pf_col_max[zc_p, xc_p], pf_col_sum[zc_p, xc_p], pf_col_s2[zc_p, xc_p], pf_col_s4[zc_p, xc_p], pf_sum_s2[zc_p, xc_p],
            dy_to_fc_mm, sd_fc_foot[zc_p, xc_p],
        ])
        blocks.extend([
            p_scalar_cols, pr_ct, pr_ct_sh, pr_bn, pr_bg, pr_ot, pr_inloc,
            py_ct, py_bn, ph_ct, ph_ot, ptan_ct,
        ])

    return np.hstack(blocks).astype(np.float32, copy=False)


def shift_ax(arr, s, ax):
    res = np.empty_like(arr)
    if s > 0:
        if ax == 0:
            res[:s] = arr[:1]; res[s:] = arr[:-s]
        elif ax == 1:
            res[:, :s] = arr[:, :1]; res[:, s:] = arr[:, :-s]
        else:
            res[:, :, :s] = arr[:, :, :1]; res[:, :, s:] = arr[:, :, :-s]
    elif s < 0:
        s = -s
        if ax == 0:
            res[:-s] = arr[s:]; res[-s:] = arr[-1:]
        elif ax == 1:
            res[:, :-s] = arr[:, s:]; res[:, -s:] = arr[:, -1:]
        else:
            res[:, :, :-s] = arr[:, :, s:]; res[:, :, -s:] = arr[:, :, -1:]
    else:
        res[:] = arr
    return res


def extract_clean_s3_case(p):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        lbl = np.array(p['lbl_c'], dtype=np.uint8)
        sp = p['sp']
        fov_y, fov_z, p99 = float(p['fov_y']), float(p['fov_z']), float(p['p99'])

        img_f0 = np.array(p['img_f0'], dtype=np.float32)
        img_lat = np.array(p['img_lat'], dtype=np.float32)
        zb_raw = np.array(p['zb'], dtype=np.float32)

        dt2 = np.array(p['dt2'], dtype=np.float32)
        dt0 = np.array(p['dt0'], dtype=np.float32)
        df2 = np.array(p['df2'], dtype=np.float32)
        nyt = np.array(p['nyt'], dtype=np.float32)
        a_new = np.array(p['a_new_tc'], dtype=np.float32)
        edt_ens = np.array(p['edt_ens'], dtype=np.float32)

        bm2 = dt2 <= 0.0
        d_in_3d = distance_transform_edt(bm2, sampling=sp).astype(np.float32)
        d_out_3d = distance_transform_edt(~bm2, sampling=sp).astype(np.float32)
        sdt_3d_true = d_out_3d - (d_in_3d - 0.5 * sp[1])

        d_in_2d = distance_transform_edt(bm2, sampling=(1000.0, sp[1], sp[2])).astype(np.float32)
        d_out_2d = distance_transform_edt(~bm2, sampling=(1000.0, sp[1], sp[2])).astype(np.float32)
        sdt_2d_yx = d_out_2d - (d_in_2d - 0.5 * sp[1])

        pt_bg = np.array(p['pt_bg'], dtype=np.float32)
        ptb = np.array(p['ptb'], dtype=np.float32)
        pt = np.array(p['pt'], dtype=np.float32)
        pf_bg = np.array(p['pf_bg'], dtype=np.float32)
        pfb = np.array(p['pfb'], dtype=np.float32)
        pf = np.array(p['pf'], dtype=np.float32)

        pt6 = np.array(p['pt6'], dtype=np.float32)
        ptb6 = np.array(p['ptb6'], dtype=np.float32)

        st = np.array(p['st'], dtype=np.float32)
        sf = np.array(p['sf'], dtype=np.float32)
        st_part = np.array(p['st_part'], dtype=np.float32)
        sf_part = np.array(p['sf_part'], dtype=np.float32)
        dth_t = float(p['dth_t'])
        st_adj = st - dth_t
        st_hpen = st - 0.05 * np.clip(dt2 - 0.8, 0.0, 2.5)
        s_bot = pt / np.maximum(pt + ptb, 1e-4)
        pred0 = np.array(p['pred_tc'], dtype=bool)

        core_c = pt >= 0.80
        zb_med_c = float(np.median(zb_raw[core_c])) if np.any(core_c) else 15.0
        zb_norm = zb_raw / max(zb_med_c, 3.0)

        sig_tan = [3.2 / sp[0], 1.2 / sp[1], 3.2 / sp[2]]
        w_ct = (np.clip(pt - 0.40, 0.0, 1.0) ** 2) + 1e-5
        i_sheet = gaussian_filter(img_lat * w_ct, sigma=sig_tan) / gaussian_filter(w_ct, sigma=sig_tan)
        w_bn = ((dt2 <= -1.2) & (dt2 >= -4.5)).astype(np.float32) + 1e-5
        i_bone = gaussian_filter(img_lat * w_bn, sigma=sig_tan) / gaussian_filter(w_bn, sigma=sig_tan)
        denom_i = np.maximum(i_sheet - i_bone, 5.0)
        inorm_f0 = (img_f0 - i_bone) / denom_i
        inorm_lat = (img_lat - i_bone) / denom_i

        gz0, gy0, gx0 = np.gradient(inorm_f0, *sp)
        gmag0 = np.sqrt(gz0**2 + gy0**2 + gx0**2)
        gyy0 = np.gradient(gy0, sp[1], axis=1)
        gxx0 = np.gradient(gx0, sp[2], axis=2)

        dts = gaussian_filter(dt2, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
        nzt, nyt_g, nxt = np.gradient(dts, *sp)
        norm_n = np.sqrt(nzt**2 + nyt_g**2 + nxt**2) + 1e-6
        nzt_u = (nzt / norm_n).astype(np.float32)
        nyt_u = (nyt_g / norm_n).astype(np.float32)
        nxt_u = (nxt / norm_n).astype(np.float32)
        gn0 = gz0 * nzt_u + gy0 * nyt_u + gx0 * nxt_u

        nz, ny, nx = lbl.shape
        yy = np.arange(ny, dtype=np.float32)[None, :, None]

        d_out_px = distance_transform_edt(~pred0).astype(np.float32)
        d_in_px = distance_transform_edt(pred0).astype(np.float32)
        sd_pred_px = d_out_px - d_in_px
        d_out_mm = distance_transform_edt(~pred0, sampling=sp).astype(np.float32)
        d_in_mm = distance_transform_edt(pred0, sampling=sp).astype(np.float32)
        sd_pred_mm = d_out_mm - d_in_mm

        w_c = np.maximum(pt**2, 1e-5)
        w_c_sum = np.sum(w_c, axis=1, keepdims=True)
        y_mean_col = np.sum(w_c * yy, axis=1, keepdims=True) / w_c_sum
        y_std_col = np.sqrt(
            np.maximum(np.sum(w_c * ((yy - y_mean_col) ** 2), axis=1, keepdims=True) / w_c_sum, 0.25)
        )

        thick_soft_mm = np.sum(pt, axis=1, keepdims=True) * sp[1]
        thick_hard_mm = np.sum(pred0, axis=1, keepdims=True).astype(np.float32) * sp[1]
        pt_col_max = np.max(pt, axis=1, keepdims=True)

        pt_row_max = np.max(pt, axis=2, keepdims=True)
        pt_row_max_s = gaussian_filter(
            pt_row_max[:, :, 0], sigma=[1.2 / sp[0], 1.2 / sp[1]]
        )[:, :, None]
        thick_x_mm = np.sum(pt * (np.abs(nyt_u) < 0.65), axis=2, keepdims=True) * sp[2]

        i_cart_col = np.sum(w_c * img_lat, axis=1, keepdims=True) / w_c_sum
        w_bcol = ((dt2 <= -0.4) & (dt2 >= -3.5)).astype(np.float32) + 1e-5
        i_bone_col = np.sum(w_bcol * img_lat, axis=1, keepdims=True) / np.sum(w_bcol, axis=1, keepdims=True)
        z_col_cart = (img_lat - i_cart_col) / np.maximum(i_cart_col - i_bone_col, 3.0)
        z_col_bone = (img_lat - i_bone_col) / np.maximum(i_cart_col - i_bone_col, 3.0)
        z_col_f0 = (img_f0 - i_bone_col) / np.maximum(i_cart_col - i_bone_col, 3.0)

        pr_col = np.any(pred0, axis=1, keepdims=True)
        y_top_raw = np.min(np.where(pred0, yy, float(ny)), axis=1, keepdims=True)
        y_bot_raw = np.max(np.where(pred0, yy, -1.0), axis=1, keepdims=True)
        w_col2d = pr_col[:, 0, :].astype(np.float32) + 1e-4
        sig_2d = [1.5 / sp[0], 1.5 / sp[2]]
        w_col_s = gaussian_filter(w_col2d, sigma=sig_2d)[:, None, :]
        y_top_smooth = (
            gaussian_filter(np.where(pr_col[:, 0, :], y_top_raw[:, 0, :], 0.0) * w_col2d, sigma=sig_2d)[:, None, :]
        ) / w_col_s
        y_bot_smooth = (
            gaussian_filter(np.where(pr_col[:, 0, :], y_bot_raw[:, 0, :], 0.0) * w_col2d, sigma=sig_2d)[:, None, :]
        ) / w_col_s

        dy_top_raw = (yy - y_top_raw) * sp[1]
        dy_bot_raw = (yy - y_bot_raw) * sp[1]
        dy_top_sm = (yy - y_top_smooth) * sp[1]
        dy_bot_sm = (yy - y_bot_smooth) * sp[1]
        dy_mean = (yy - y_mean_col) * sp[1]
        zy_col = (yy - y_mean_col) / y_std_col
        rel_y_soft = dy_mean / np.maximum(thick_soft_mm, 0.8)

        in0_ym3 = shift_ax(img_f0, 3, 1); in0_ym2 = shift_ax(img_f0, 2, 1)
        in0_ym1_raw = shift_ax(img_f0, 1, 1); in0_yp1_raw = shift_ax(img_f0, -1, 1)
        in0_yp2 = shift_ax(img_f0, -2, 1); in0_yp3 = shift_ax(img_f0, -3, 1)
        i_loc_hi = np.maximum(in0_ym2, in0_ym3)
        i_loc_lo = np.minimum(in0_yp2, in0_yp3)
        span_loc = np.maximum(i_loc_hi - i_loc_lo, 5.0)
        p_step_0 = np.clip((img_f0 - i_loc_lo) / span_loc, -0.5, 1.5)
        p_step_ym1 = np.clip((in0_ym1_raw - i_loc_lo) / span_loc, -0.5, 1.5)
        p_step_yp1 = np.clip((in0_yp1_raw - i_loc_lo) / span_loc, -0.5, 1.5)

        pt_ym1 = shift_ax(pt, 1, 1); pt_yp1 = shift_ax(pt, -1, 1)
        pt_ym2 = shift_ax(pt, 2, 1); pt_yp2 = shift_ax(pt, -2, 1)
        ptb_ym1 = shift_ax(ptb, 1, 1); ptb_yp1 = shift_ax(ptb, -1, 1)
        pbg_ym1 = shift_ax(pt_bg, 1, 1); pbg_yp1 = shift_ax(pt_bg, -1, 1)
        pf_ym1 = shift_ax(pf, 1, 1); pf_yp1 = shift_ax(pf, -1, 1)
        st_ym1 = shift_ax(st, 1, 1); st_yp1 = shift_ax(st, -1, 1)
        st_ym2 = shift_ax(st, 2, 1); st_yp2 = shift_ax(st, -2, 1)
        zbn_ym1 = shift_ax(zb_norm, 1, 1); zbn_yp1 = shift_ax(zb_norm, -1, 1)
        in0_ym1 = shift_ax(inorm_f0, 1, 1); in0_yp1 = shift_ax(inorm_f0, -1, 1)

        st_xm1 = shift_ax(st, 1, 2); st_xp1 = shift_ax(st, -1, 2)
        st_xm2 = shift_ax(st, 2, 2); st_xp2 = shift_ax(st, -2, 2)
        pt_xm1 = shift_ax(pt, 1, 2); pt_xp1 = shift_ax(pt, -1, 2)
        ptb_xm1 = shift_ax(ptb, 1, 2); ptb_xp1 = shift_ax(ptb, -1, 2)
        in0_xm1 = shift_ax(inorm_f0, 1, 2); in0_xp1 = shift_ax(inorm_f0, -1, 2)

        st_zm1 = shift_ax(st, 1, 0); st_zp1 = shift_ax(st, -1, 0)
        pt_zm1 = shift_ax(pt, 1, 0); pt_zp1 = shift_ax(pt, -1, 0)
        in0_zm1 = shift_ax(inorm_f0, 1, 0); in0_zp1 = shift_ax(inorm_f0, -1, 0)

        st_s1 = gaussian_filter(st, sigma=[0.8 / sp[0], 0.5 / sp[1], 0.8 / sp[2]])
        st_s2 = gaussian_filter(st, sigma=[1.8 / sp[0], 1.0 / sp[1], 1.8 / sp[2]])
        pt_lat = gaussian_filter(pt, sigma=[1.5 / sp[0], 0.2 / sp[1], 1.5 / sp[2]])
        sbot_lat = gaussian_filter(s_bot, sigma=[2.0 / sp[0], 0.4 / sp[1], 2.0 / sp[2]])

        m_bot_vol = (dt2 >= -0.45) & (dt2 <= 0.55) & (np.maximum(pt_ym1, pt_ym2) >= 0.50)
        m_top_vol = ~m_bot_vol

        dv20_all = float(np.mean((dt2 - dt0)[m_bot_vol])) if np.any(m_bot_vol) else 0.0
        rcb_bot_all = float(np.mean(s_bot[m_bot_vol])) if np.any(m_bot_vol) else 0.5
        m_top_pr = m_top_vol & (st >= 0.42)
        st_top_mean = float(np.mean(st[m_top_pr])) if np.any(m_top_pr) else 0.55
        m_wb = (dt2 >= 0.0) & (dt2 <= 2.5) & (a_new >= 0.35)
        js_med = float(np.median((df2 + dt2)[m_wb])) if np.any(m_wb) else 4.0
        m_sup = (dt2 >= 1.1) & (dt2 <= 2.2) & (pt >= 0.50)
        m_mid = (dt2 >= 0.1) & (dt2 <= 0.9) & (pt >= 0.50)
        isup_imid = (
            float(np.mean(inorm_f0[m_sup]) - np.mean(inorm_f0[m_mid]))
            if (np.any(m_sup) and np.any(m_mid))
            else 0.0
        )
        m_cor = (dt2 >= 0.4) & (dt2 <= 1.4) & (a_new >= 0.45)
        st_cor = float(np.mean(st[m_cor])) if np.any(m_cor) else 0.85
        log_p99 = float(np.log10(max(p99, 10.0)))
        low_p99 = float(p99 < 250.0)

        case_ctx = np.array([
            log_p99, low_p99, (fov_y - 136.0) / 10.0, (fov_z - 110.0) / 8.0,
            dv20_all * 10.0, rcb_bot_all,
            st_top_mean - st_cor, (js_med - 4.0) / 1.5, isup_imid * 5.0, st_cor,
        ], dtype=np.float32)

        cand = ((st >= 0.16) | (st_adj >= 0.16) | (d_out_px <= 2.01)) & (dt2 >= -1.2) & (dt2 <= 5.5)
        if not np.any(cand):
            return np.zeros((0, 132), dtype=np.float32), np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=bool), cand, case_ctx

        m = cand
        zs_m, ys_m, xs_m = np.where(m)
        nzm, nym, nxm = nzt_u[m], nyt_u[m], nxt_u[m]

        ray_steps = [-1.2, -0.8, -0.4, 0.4, 0.8, 1.2, 1.6]
        nr_st = np.empty((len(zs_m), len(ray_steps)), dtype=np.float32)
        nr_in0 = np.empty((len(zs_m), len(ray_steps)), dtype=np.float32)
        nr_zbn = np.empty((len(zs_m), len(ray_steps)), dtype=np.float32)
        for k, d_r in enumerate(ray_steps):
            rz = np.clip(np.round(zs_m + d_r * nzm / sp[0]).astype(int), 0, nz - 1)
            ry = np.clip(np.round(ys_m + d_r * nym / sp[1]).astype(int), 0, ny - 1)
            rx = np.clip(np.round(xs_m + d_r * nxm / sp[2]).astype(int), 0, nx - 1)
            nr_st[:, k] = st[rz, ry, rx]
            nr_in0[:, k] = inorm_f0[rz, ry, rx]
            nr_zbn[:, k] = zb_norm[rz, ry, rx]

        def bcast(arr):
            return np.broadcast_to(arr, lbl.shape)[m]

        X_loc = np.hstack([
            np.column_stack([
                sd_pred_px[m], sd_pred_mm[m], dy_top_raw[m], dy_bot_raw[m], bcast(thick_hard_mm),
                st[m], st_hpen[m], s_bot[m], sf[m], st_part[m], sf_part[m],
                st[m] - st_s2[m], s_bot[m] - sbot_lat[m],
                pt[m], ptb[m], pt_bg[m], pt6[m], ptb6[m],
                pf[m], pfb[m], pf_bg[m],
                pt[m] / np.maximum(pt[m] + pt_bg[m], 1e-4),
                dt2[m], dt0[m], df2[m], dt2[m] - dt0[m],
                sdt_3d_true[m], sdt_2d_yx[m], d_in_3d[m], d_in_2d[m], sdt_2d_yx[m] - dt2[m],
                nyt[m], a_new[m], edt_ens[m],
                zb_norm[m], z_col_cart[m], z_col_bone[m],
                bcast(thick_soft_mm), bcast(pt_col_max),
                bcast(thick_soft_mm) / np.maximum(2.0 * edt_ens[m], 0.8),
                bcast(pt_row_max), bcast(pt_row_max_s), bcast(thick_x_mm),
                dy_top_sm[m], dy_bot_sm[m],
                dy_top_raw[m] - dy_top_sm[m], dy_bot_raw[m] - dy_bot_sm[m],
                dy_mean[m], zy_col[m], rel_y_soft[m],
                pt_ym1[m], pt_yp1[m], pt_ym2[m], pt_yp2[m],
                ptb_ym1[m], ptb_yp1[m], pbg_ym1[m], pbg_yp1[m],
                pf_ym1[m], pf_yp1[m],
                st_ym1[m], st_yp1[m], st_ym2[m], st_yp2[m],
                zbn_ym1[m], zbn_yp1[m], zb_norm[m] - zbn_ym1[m], zb_norm[m] - zbn_yp1[m],
                p_step_0[m], p_step_ym1[m], p_step_yp1[m], p_step_ym1[m] - p_step_yp1[m],
                st_s1[m], st_s2[m], st[m] - st_s1[m], pt_lat[m],
                inorm_f0[m], inorm_lat[m], inorm_f0[m] - inorm_lat[m], z_col_f0[m],
                gy0[m], gx0[m], gz0[m], gmag0[m], gn0[m], gyy0[m], gxx0[m],
                nzt_u[m], nyt_u[m], nxt_u[m],
                st_xm1[m], st_xp1[m], st_xm2[m], st_xp2[m],
                pt_xm1[m], pt_xp1[m], ptb_xm1[m], ptb_xp1[m],
                in0_xm1[m] - inorm_f0[m], in0_xp1[m] - inorm_f0[m],
                st_zm1[m], st_zp1[m], pt_zm1[m], pt_zp1[m],
                in0_zm1[m] - inorm_f0[m], in0_zp1[m] - inorm_f0[m],
                in0_ym1[m] - inorm_f0[m], in0_yp1[m] - inorm_f0[m],
            ]),
            nr_st, nr_in0, nr_zbn,
            np.column_stack([
                st[m] - st_cor, st_ym1[m] - st_cor, st_yp1[m] - st_cor,
            ]),
        ]).astype(np.float32)

        is_mbot_m = m_bot_vol[m]
        y = (lbl[m] == 4).astype(np.uint8)
        return X_loc, y, is_mbot_m, cand, case_ctx


def extract_full_oof_s3_case(p, m_override=None, pre_clean=None):
    p_eff = dict(p)
    if m_override is not None:
        for k, v in m_override.items():
            p_eff[k] = v
    if pre_clean is not None:
        X_loc, y, _, m, ctx = pre_clean
    else:
        X_loc, y, _, m, ctx = extract_clean_s3_case(p_eff)
    dt2 = np.array(p_eff['dt2'], dtype=np.float32)
    pt = np.array(p_eff['pt'], dtype=np.float32)
    ptb = np.array(p_eff['ptb'], dtype=np.float32)
    pt6 = np.array(p_eff.get('pt6', pt), dtype=np.float32)
    ptb6 = np.array(p_eff.get('ptb6', ptb), dtype=np.float32)
    p1_tc6 = np.array(p_eff.get('p1_tc6', pt6), dtype=np.float32)
    p1_tb6 = np.array(p_eff.get('p1_tb6', ptb6), dtype=np.float32)
    p2h_tc6 = np.array(p_eff.get('p2h_tc6', pt6), dtype=np.float32)
    p2h_tb6 = np.array(p_eff.get('p2h_tb6', ptb6), dtype=np.float32)
    p1_tc7 = np.array(p_eff.get('p1_tc7', p1_tc6), dtype=np.float32)
    p2h_tc7 = np.array(p_eff.get('p2h_tc7', p2h_tc6), dtype=np.float32)
    p2h_tb7 = np.array(p_eff.get('p2h_tb7', p2h_tb6), dtype=np.float32)
    p2c_tc7 = np.array(p_eff.get('p2c_tc7', pt6), dtype=np.float32)
    p2r_tc7 = np.array(p_eff.get('p2r_tc7', p2c_tc7), dtype=np.float32)
    st = np.array(p_eff['st'], dtype=np.float32)
    part = np.array(p_eff['st_part']) > np.array(p_eff['sf_part'])

    X_oof_extra = np.column_stack([
        p1_tc6[m], p1_tb6[m], p2h_tc6[m], p2h_tb6[m],
        p1_tc7[m], p2h_tc7[m], p2h_tb7[m], p2c_tc7[m], p2r_tc7[m],
        (p2h_tc6 - p1_tc6)[m], (p2h_tb6 - p1_tb6)[m],
        (pt6 - p1_tc6)[m], (ptb6 - p1_tb6)[m],
        (p2h_tc7 - p1_tc7)[m], (p2c_tc7 - p1_tc7)[m],
        (p2r_tc7 - p2h_tc6)[m], (p2c_tc7 - pt6)[m], (p2c_tc7 - pt)[m],
        shift_ax(p1_tc6, 1, 1)[m], shift_ax(p1_tc6, -1, 1)[m],
        shift_ax(p2h_tc6, 1, 1)[m], shift_ax(p2h_tc6, -1, 1)[m],
        shift_ax(p2c_tc7, 1, 1)[m], shift_ax(p2c_tc7, -1, 1)[m],
    ]).astype(np.float32)

    X_g = np.hstack([X_loc[:, 5:], X_oof_extra]).astype(np.float32)

    sub = (ptb >= 0.12) & (pt >= 0.15) & part
    bnd = (st >= 0.35) & (st <= 0.65) & part
    sup = (dt2 >= 0.8) & (st >= 0.35) & (st <= 0.65) & part
    ctx_extra = np.array([
        float(np.mean((p2h_tc6 - p1_tc6)[sub])) * 10.0 if np.any(sub) else 0.0,
        float(np.mean((p2h_tb6 - p1_tb6)[sub])) * 10.0 if np.any(sub) else 0.0,
        float(np.mean((p2h_tc6 - p1_tc6)[bnd])) * 10.0 if np.any(bnd) else 0.0,
        float(np.mean((p2h_tc6 - p1_tc6)[sup])) * 10.0 if np.any(sup) else 0.0,
        float(np.mean((p2r_tc7 - p2h_tc6)[sub])) * 10.0 if np.any(sub) else 0.0,
        float(p_eff['dth_t']) * 10.0,
    ], dtype=np.float32)
    ctx_full = np.concatenate([ctx, ctx_extra])
    return X_g, y, m, ctx_full


def predict_full_oof_s3_case(p, f3_models, m_override=None, pre_clean=None):
    X_g, _, m, ctx_full = extract_full_oof_s3_case(p, m_override=m_override, pre_clean=pre_clean)
    n_vox = len(X_g)
    ctx_mat = np.broadcast_to(ctx_full[None, :], (n_vox, len(ctx_full)))
    X_c = np.hstack([X_g, ctx_mat])
    models_g = f3_models['models_g']
    models_c = f3_models['models_c']
    pred_g = np.zeros(n_vox, dtype=np.float32)
    pred_c = np.zeros(n_vox, dtype=np.float32)
    for mg, mc in zip(models_g, models_c):
        pred_g += (1.0 / len(models_g)) * mg.predict_proba(X_g)[:, 1].astype(np.float32)
        pred_c += (1.0 / len(models_c)) * mc.predict_proba(X_c)[:, 1].astype(np.float32)
    return pred_g, pred_c, m


def extract_pure_oof_case(p, cand):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        lbl = np.array(p['lbl_c'], dtype=np.uint8)
        sp = p['sp']

        img_f0 = np.array(p['img_f0'], dtype=np.float32)
        img_lat = np.array(p['img_lat'], dtype=np.float32)
        zb_raw = np.array(p['zb'], dtype=np.float32)

        dt2 = np.array(p['dt2'], dtype=np.float32)
        dt0 = np.array(p['dt0'], dtype=np.float32)
        df2 = np.array(p['df2'], dtype=np.float32)
        nyt = np.array(p['nyt'], dtype=np.float32)
        a_new = np.array(p['a_new_tc'], dtype=np.float32)
        edt_ens = np.array(p['edt_ens'], dtype=np.float32)

        bm2 = dt2 <= 0.0
        d_in_3d = distance_transform_edt(bm2, sampling=sp).astype(np.float32)
        d_out_3d = distance_transform_edt(~bm2, sampling=sp).astype(np.float32)
        sdt_3d_true = d_out_3d - (d_in_3d - 0.5 * sp[1])

        d_in_2d = distance_transform_edt(bm2, sampling=(1000.0, sp[1], sp[2])).astype(np.float32)
        d_out_2d = distance_transform_edt(~bm2, sampling=(1000.0, sp[1], sp[2])).astype(np.float32)
        sdt_2d_yx = d_out_2d - (d_in_2d - 0.5 * sp[1])

        sig_p1 = [0.40 / sp[0], 0.20 / sp[1], 0.55 / sp[2]]
        p1_tc6 = gaussian_filter(np.array(p['p1_tc6'], dtype=np.float32), sigma=sig_p1)
        p1_tb6 = gaussian_filter(np.array(p['p1_tb6'], dtype=np.float32), sigma=sig_p1)
        p1_tc7 = gaussian_filter(np.array(p['p1_tc7'], dtype=np.float32), sigma=sig_p1)

        p1_tc = 0.50 * p1_tc6 + 0.50 * p1_tc7
        p1_tb = p1_tb6
        s1 = p1_tc - 0.20 * p1_tb + 0.22 * a_new
        s1_bot = p1_tc / np.maximum(p1_tc + p1_tb, 1e-4)

        core_c = p1_tc >= 0.70
        zb_med_c = float(np.median(zb_raw[core_c])) if np.any(core_c) else 15.0
        zb_norm = zb_raw / max(zb_med_c, 3.0)

        sig_tan = [3.2 / sp[0], 1.2 / sp[1], 3.2 / sp[2]]
        w_ct = (np.clip(p1_tc - 0.35, 0.0, 1.0) ** 2) + 1e-5
        i_sheet = gaussian_filter(img_lat * w_ct, sigma=sig_tan) / gaussian_filter(w_ct, sigma=sig_tan)
        w_bn = ((dt2 <= -1.2) & (dt2 >= -4.5)).astype(np.float32) + 1e-5
        i_bone = gaussian_filter(img_lat * w_bn, sigma=sig_tan) / gaussian_filter(w_bn, sigma=sig_tan)
        denom_i = np.maximum(i_sheet - i_bone, 5.0)
        inorm_f0 = (img_f0 - i_bone) / denom_i
        inorm_lat = (img_lat - i_bone) / denom_i

        gz0, gy0, gx0 = np.gradient(inorm_f0, *sp)
        gmag0 = np.sqrt(gz0**2 + gy0**2 + gx0**2)
        gyy0 = np.gradient(gy0, sp[1], axis=1)
        gxx0 = np.gradient(gx0, sp[2], axis=2)

        dts = gaussian_filter(dt2, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
        nzt, nyt_g, nxt = np.gradient(dts, *sp)
        norm_n = np.sqrt(nzt**2 + nyt_g**2 + nxt**2) + 1e-6
        nzt_u = (nzt / norm_n).astype(np.float32)
        nyt_u = (nyt_g / norm_n).astype(np.float32)
        nxt_u = (nxt / norm_n).astype(np.float32)
        gn0 = gz0 * nzt_u + gy0 * nyt_u + gx0 * nxt_u

        nz, ny, nx = lbl.shape
        yy = np.arange(ny, dtype=np.float32)[None, :, None]

        w_c = np.maximum(p1_tc**2, 1e-5)
        w_c_sum = np.sum(w_c, axis=1, keepdims=True)
        y_mean_col = np.sum(w_c * yy, axis=1, keepdims=True) / w_c_sum
        y_std_col = np.sqrt(
            np.maximum(np.sum(w_c * ((yy - y_mean_col) ** 2), axis=1, keepdims=True) / w_c_sum, 0.25)
        )

        thick_soft_mm = np.sum(p1_tc, axis=1, keepdims=True) * sp[1]
        p1_col_max = np.max(p1_tc, axis=1, keepdims=True)
        p1_row_max = np.max(p1_tc, axis=2, keepdims=True)
        thick_x_mm = np.sum(p1_tc * (np.abs(nyt_u) < 0.65), axis=2, keepdims=True) * sp[2]

        i_cart_col = np.sum(w_c * img_lat, axis=1, keepdims=True) / w_c_sum
        w_bcol = ((dt2 <= -0.4) & (dt2 >= -3.5)).astype(np.float32) + 1e-5
        i_bone_col = np.sum(w_bcol * img_lat, axis=1, keepdims=True) / np.sum(w_bcol, axis=1, keepdims=True)
        z_col_cart = (img_lat - i_cart_col) / np.maximum(i_cart_col - i_bone_col, 3.0)
        z_col_bone = (img_lat - i_bone_col) / np.maximum(i_cart_col - i_bone_col, 3.0)
        z_col_f0 = (img_f0 - i_bone_col) / np.maximum(i_cart_col - i_bone_col, 3.0)

        dy_mean = (yy - y_mean_col) * sp[1]
        zy_col = (yy - y_mean_col) / y_std_col
        rel_y_soft = dy_mean / np.maximum(thick_soft_mm, 0.8)

        in0_ym3 = shift_ax(img_f0, 3, 1); in0_ym2 = shift_ax(img_f0, 2, 1)
        in0_ym1_raw = shift_ax(img_f0, 1, 1); in0_yp1_raw = shift_ax(img_f0, -1, 1)
        in0_yp2 = shift_ax(img_f0, -2, 1); in0_yp3 = shift_ax(img_f0, -3, 1)
        i_loc_hi = np.maximum(in0_ym2, in0_ym3)
        i_loc_lo = np.minimum(in0_yp2, in0_yp3)
        span_loc = np.maximum(i_loc_hi - i_loc_lo, 5.0)
        p_step_0 = np.clip((img_f0 - i_loc_lo) / span_loc, -0.5, 1.5)
        p_step_ym1 = np.clip((in0_ym1_raw - i_loc_lo) / span_loc, -0.5, 1.5)
        p_step_yp1 = np.clip((in0_yp1_raw - i_loc_lo) / span_loc, -0.5, 1.5)

        p1_ym1 = shift_ax(p1_tc, 1, 1); p1_yp1 = shift_ax(p1_tc, -1, 1)
        p1_ym2 = shift_ax(p1_tc, 2, 1); p1_yp2 = shift_ax(p1_tc, -2, 1)
        tb1_ym1 = shift_ax(p1_tb, 1, 1); tb1_yp1 = shift_ax(p1_tb, -1, 1)
        s1_ym1 = shift_ax(s1, 1, 1); s1_yp1 = shift_ax(s1, -1, 1)
        s1_ym2 = shift_ax(s1, 2, 1); s1_yp2 = shift_ax(s1, -2, 1)
        zbn_ym1 = shift_ax(zb_norm, 1, 1); zbn_yp1 = shift_ax(zb_norm, -1, 1)
        in0_ym1 = shift_ax(inorm_f0, 1, 1); in0_yp1 = shift_ax(inorm_f0, -1, 1)

        s1_xm1 = shift_ax(s1, 1, 2); s1_xp1 = shift_ax(s1, -1, 2)
        p1_xm1 = shift_ax(p1_tc, 1, 2); p1_xp1 = shift_ax(p1_tc, -1, 2)
        in0_xm1 = shift_ax(inorm_f0, 1, 2); in0_xp1 = shift_ax(inorm_f0, -1, 2)

        s1_zm1 = shift_ax(s1, 1, 0); s1_zp1 = shift_ax(s1, -1, 0)
        p1_zm1 = shift_ax(p1_tc, 1, 0); p1_zp1 = shift_ax(p1_tc, -1, 0)
        in0_zm1 = shift_ax(inorm_f0, 1, 0); in0_zp1 = shift_ax(inorm_f0, -1, 0)

        s1_sm1 = gaussian_filter(s1, sigma=[0.8 / sp[0], 0.5 / sp[1], 0.8 / sp[2]])
        s1_sm2 = gaussian_filter(s1, sigma=[1.8 / sp[0], 1.0 / sp[1], 1.8 / sp[2]])
        p1_lat = gaussian_filter(p1_tc, sigma=[1.5 / sp[0], 0.2 / sp[1], 1.5 / sp[2]])
        s1bot_lat = gaussian_filter(s1_bot, sigma=[2.0 / sp[0], 0.4 / sp[1], 2.0 / sp[2]])

        m = cand
        zs_m, ys_m, xs_m = np.where(m)
        nzm, nym, nxm = nzt_u[m], nyt_u[m], nxt_u[m]

        ray_steps = [-1.2, -0.8, -0.4, 0.4, 0.8, 1.2, 1.6]
        nr_s1 = np.empty((len(zs_m), len(ray_steps)), dtype=np.float32)
        nr_in0 = np.empty((len(zs_m), len(ray_steps)), dtype=np.float32)
        nr_zbn = np.empty((len(zs_m), len(ray_steps)), dtype=np.float32)
        for k, d_r in enumerate(ray_steps):
            rz = np.clip(np.round(zs_m + d_r * nzm / sp[0]).astype(int), 0, nz - 1)
            ry = np.clip(np.round(ys_m + d_r * nym / sp[1]).astype(int), 0, ny - 1)
            rx = np.clip(np.round(xs_m + d_r * nxm / sp[2]).astype(int), 0, nx - 1)
            nr_s1[:, k] = s1[rz, ry, rx]
            nr_in0[:, k] = inorm_f0[rz, ry, rx]
            nr_zbn[:, k] = zb_norm[rz, ry, rx]

        def bcast(arr):
            return np.broadcast_to(arr, lbl.shape)[m]

        X_pure = np.hstack([
            np.column_stack([
                p1_tc[m], p1_tb[m], p1_tc6[m], p1_tc7[m], s1[m], s1_bot[m],
                s1[m] - s1_sm1[m], s1[m] - s1_sm2[m], s1_bot[m] - s1bot_lat[m],
                s1_sm1[m], s1_sm2[m], p1_lat[m],
                dt2[m], dt0[m], df2[m], dt2[m] - dt0[m], df2[m] + dt2[m],
                sdt_3d_true[m], sdt_2d_yx[m], d_in_3d[m], d_in_2d[m], sdt_2d_yx[m] - dt2[m],
                nyt[m], a_new[m], edt_ens[m],
                zb_norm[m], z_col_cart[m], z_col_bone[m], z_col_f0[m],
                bcast(thick_soft_mm), bcast(p1_col_max),
                bcast(thick_soft_mm) / np.maximum(2.0 * edt_ens[m], 0.8),
                bcast(p1_row_max), bcast(thick_x_mm),
                dy_mean[m], zy_col[m], rel_y_soft[m],
                p1_ym1[m], p1_yp1[m], p1_ym2[m], p1_yp2[m],
                tb1_ym1[m], tb1_yp1[m],
                s1_ym1[m], s1_yp1[m], s1_ym2[m], s1_yp2[m],
                zbn_ym1[m], zbn_yp1[m], zb_norm[m] - zbn_ym1[m], zb_norm[m] - zbn_yp1[m],
                p_step_0[m], p_step_ym1[m], p_step_yp1[m], p_step_ym1[m] - p_step_yp1[m],
                inorm_f0[m], inorm_lat[m], inorm_f0[m] - inorm_lat[m],
                gy0[m], gx0[m], gz0[m], gmag0[m], gn0[m], gyy0[m], gxx0[m],
                nzt_u[m], nyt_u[m], nxt_u[m],
                s1_xm1[m], s1_xp1[m], p1_xm1[m], p1_xp1[m],
                in0_xm1[m] - inorm_f0[m], in0_xp1[m] - inorm_f0[m],
                s1_zm1[m], s1_zp1[m], p1_zm1[m], p1_zp1[m],
                in0_zm1[m] - inorm_f0[m], in0_zp1[m] - inorm_f0[m],
                in0_ym1[m] - inorm_f0[m], in0_yp1[m] - inorm_f0[m],
            ]),
            nr_s1, nr_in0, nr_zbn,
        ]).astype(np.float32)

        y = (lbl[m] == 4).astype(np.uint8)
        return X_pure, y


def compute_case_smoothed_components(p, s3_glob, s3_ctx, s3_cand, f3_g, f3_c, p3_pure_1d):
    lbl_sh = p['st'].shape
    cand = s3_cand
    st = np.array(p['st'], dtype=np.float32)
    pt = np.array(p['pt'], dtype=np.float32)
    pt6 = np.array(p['pt6'], dtype=np.float32)
    pt7 = np.array(p['p2c_tc7'], dtype=np.float32)
    ptb = np.array(p['ptb'], dtype=np.float32)
    pf = np.array(p['pf'], dtype=np.float32)
    a_new = np.array(p['a_new_tc'], dtype=np.float32)
    a_old = np.array(p['a_old_tc'], dtype=np.float32)
    dt2 = np.array(p['dt2'], dtype=np.float32)
    zb = np.array(p['zb'], dtype=np.float32)
    img = np.array(p['img_lat'], dtype=np.float32)
    part = np.array(p['st_part']) > np.array(p['sf_part'])

    p3c_1d = 0.65 * s3_glob + 0.35 * s3_ctx
    p3_c = np.zeros(lbl_sh, dtype=np.float32)
    p3_c[cand] = p3c_1d + 0.12
    p3_c_s = gaussian_filter(p3_c, sigma=_SIG_T_CALIB)[cand]

    p3f_1d = 0.65 * f3_g + 0.35 * f3_c
    p3_f = np.zeros(lbl_sh, dtype=np.float32)
    p3_f[cand] = p3f_1d + 0.12
    p3_f_s = gaussian_filter(p3_f, sigma=_SIG_T_CALIB)[cand]

    p3_p_c = np.zeros(lbl_sh, dtype=np.float32)
    p3_p_c[cand] = p3_pure_1d - (float(np.mean(p3_pure_1d)) - float(np.mean(p3c_1d))) + 0.12
    p3_p_c_s = gaussian_filter(p3_p_c, sigma=_SIG_T_CALIB)[cand]

    p3_p_r = np.zeros(lbl_sh, dtype=np.float32)
    p3_p_r[cand] = p3_pure_1d + 0.12
    p3_p_r_s = gaussian_filter(p3_p_r, sigma=_SIG_T_CALIB)[cand]

    del_t = a_new - a_old
    base6_s = gaussian_filter(
        0.5 * pt6 + 0.5 * pt + 0.35 * del_t + 0.28 * np.minimum(pf, pt),
        sigma=_SIG_T_CALIB,
    )[cand]
    pt7_s = gaussian_filter(pt7, sigma=_SIG_T_CALIB)[cand]
    ptb_s = gaussian_filter(ptb, sigma=_SIG_T_CALIB)[cand]
    a_new_s = gaussian_filter(a_new, sigma=_SIG_T_CALIB)[cand]

    pt_ym1 = np.pad(pt, ((0, 0), (1, 0), (0, 0)), mode='edge')[:, :-1, :]
    pt_ym2 = np.pad(pt, ((0, 0), (2, 0), (0, 0)), mode='edge')[:, :-2, :]
    m_bot = (dt2 >= -0.45) & (dt2 <= 0.55) & (np.maximum(pt_ym1, pt_ym2) >= 0.50)
    pr_st = (st >= 0.51) & part
    core80 = (st >= 0.80) & part
    ic80 = float(np.median(img[core80])) if np.any(core80) else 100.0
    m_db = (dt2 <= -2.0) & (dt2 >= -4.5)
    ib_deep = float(np.median(img[m_db])) if np.any(m_db) else 50.0
    inorm_d = (img - ib_deep) / max(ic80 - ib_deep, 10.0)
    wb = (np.max(pt, axis=1, keepdims=True) >= 0.65) & (
        np.max(pf, axis=1, keepdims=True) >= 0.40
    )
    wb_pr = pr_st & wb & (~m_bot)
    in_wb = float(np.mean(inorm_d[wb_pr])) if np.any(wb_pr) else 0.8
    rim_dark = np.where(
        (~wb) & (~m_bot), np.clip(inorm_d - (in_wb - 0.18), -0.6, 0.3), 0.0
    ).astype(np.float32)[cand]

    dt2_c = dt2[cand]
    zb_c = zb[cand]
    img_c = img[cand]
    st_c = st[cand]
    part_c = part[cand]
    core_c = (st_c >= 0.78) & part_c
    ic_c = float(np.median(img_c[core_c])) if np.any(core_c) else 100.0
    ib_c = float(np.median(img_c[dt2_c <= -0.6])) if np.any(dt2_c <= -0.6) else 50.0
    inorm_c = (img_c - ib_c) / max(ic_c - ib_c, 10.0)
    zb_med = float(np.median(zb_c[core_c])) if np.any(core_c) else 15.0
    zbn_c = zb_c / max(zb_med, 3.0)
    w_sub = np.clip(1.0 - np.abs(dt2_c) / 0.55, 0.0, 1.0) * np.clip(
        a_new_s / 0.35, 0.0, 1.0
    )
    sub_in = (w_sub * np.clip(inorm_c - 0.55, -0.6, 0.6)).astype(np.float32)
    sub_zb = (w_sub * np.clip(zbn_c - 0.55, -0.6, 0.6)).astype(np.float32)

    return {
        'part': part_c,
        'p3c': p3_c_s,
        'p3f_raw': p3_f_s,
        'p3p': p3_p_c_s,
        'p3p_raw': p3_p_r_s,
        'base6': base6_s,
        'pt7': pt7_s,
        'ptb': ptb_s,
        'a_new': a_new_s,
        'rim': rim_dark,
        'dt2': dt2_c,
        'sub_in': sub_in,
        'sub_zb': sub_zb,
        'inorm_c': inorm_c,
        'zbn_c': zbn_c,
    }


def extract_clean_calib_feats_single(
    p, s3_glob, s3_ctx, s3_cand, f3_g, f3_c, p3_pure_1d, matched_p1p2=None, cc=None
):
    if cc is None:
        cc = compute_case_smoothed_components(
            p, s3_glob, s3_ctx, s3_cand, f3_g, f3_c, p3_pure_1d
        )
    c = s3_cand
    sp = p['sp']
    dt2_full = np.array(p['dt2'], dtype=np.float32)
    a_new_full = np.array(p['a_new_tc'], dtype=np.float32)
    a_old_full = np.array(p['a_old_tc'], dtype=np.float32)
    img_lat_full = np.array(p['img_lat'], dtype=np.float32)
    pt_full = np.array(p['pt'], dtype=np.float32)
    part_full = np.array(p['st_part']) > np.array(p['sf_part'])

    m_deep_b = (dt2_full >= -4.0) & (dt2_full <= -1.8) & (a_new_full >= 0.15)
    m_core_c = (dt2_full >= 0.5) & (dt2_full <= 1.8) & (pt_full >= 0.70) & part_full
    ib_f = float(np.median(img_lat_full[m_deep_b])) if np.any(m_deep_b) else 50.0
    ic_f = float(np.median(img_lat_full[m_core_c])) if np.any(m_core_c) else 200.0
    inorm_full = (img_lat_full - ib_f) / max(ic_f - ib_f, 10.0)
    m_in1 = (dt2_full > -0.60) & (dt2_full <= 0.0) & (a_new_full >= 0.25)
    m_out1 = (dt2_full > 0.0) & (dt2_full <= 0.55) & (a_new_full >= 0.25) & part_full
    gy = -np.gradient(inorm_full, sp[1], axis=1)
    m_grad = (
        (dt2_full >= -0.9) & (dt2_full <= 0.9) & (a_new_full >= 0.30) & (gy > 0)
    )
    dt2_at_grad = (
        float(
            np.sum(dt2_full[m_grad] * (gy[m_grad] ** 2))
            / (np.sum(gy[m_grad] ** 2) + 1e-5)
        )
        if np.any(m_grad)
        else 0.0
    )

    dt2 = cc['dt2']
    dt0 = np.array(p['dt0'][c], dtype=np.float32)
    edt = np.array(p['edt_ens'][c], dtype=np.float32)
    zb = np.array(p['zb'][c], dtype=np.float32)
    st = np.array(p['st'][c], dtype=np.float32)
    pt = pt_full[c]
    ptb = np.array(p['ptb'][c], dtype=np.float32)
    pt7 = np.array(p['pt7'][c], dtype=np.float32)
    src = matched_p1p2 if matched_p1p2 is not None else p
    p1_6 = np.array(src['p1_tc6'][c], dtype=np.float32)
    p1b_6 = np.array(src['p1_tb6'][c], dtype=np.float32)
    p2h_6 = np.array(src['p2h_tc6'][c], dtype=np.float32)
    p2hb_6 = np.array(src['p2h_tb6'][c], dtype=np.float32)
    p2r_7 = np.array(src['p2r_tc7'][c], dtype=np.float32)
    an = a_new_full[c]
    img = img_lat_full[c]
    img0 = np.array(p['img_f0'][c], dtype=np.float32)
    part = cc['part']

    sub = (ptb >= 0.12) & (pt >= 0.15) & part
    sub_s = (dt2 >= -0.45) & (dt2 <= 0.45) & (an >= 0.25) & part
    bnd = (st >= 0.35) & (st <= 0.65) & part
    b_sh = (st >= 0.40) & (st <= 0.60) & part
    sup = (dt2 >= 0.8) & (st >= 0.35) & (st <= 0.65) & part
    core = (st >= 0.75) & part

    s_bot = pt / np.maximum(pt + ptb, 1e-4)
    ic = float(np.median(img[core])) if np.any(core) else 100.0
    ib = float(np.median(img[dt2 <= -0.5])) if np.any(dt2 <= -0.5) else 50.0
    contrast = (ic - ib) / (abs(ib) + 50.0)

    s = 0.32 * cc['p3c'] + 0.08 * cc['p3p'] + 0.60 * cc['base6'] + 0.05 * cc['rim']
    pr = part & (s >= 0.51)
    l_th = pr & (s < 0.57)
    pr_sup = pr & (dt2 > 0.55)
    part_sup = part & (dt2 > 0.55)
    part_sub = part & (dt2 <= 0.45)
    zb_med = float(np.median(zb[core])) if np.any(core) else 15.0
    zbn = zb / max(zb_med, 3.0)
    n_pr = max(float(np.sum(pr)), 1.0)
    p3c_raw = 0.65 * s3_glob + 0.35 * s3_ctx
    p3f_sm = cc['p3f_raw'] - 0.12
    p3f_sm_c = p3f_sm - (float(np.mean(p3f_sm)) - float(np.mean(p3c_raw)))

    core78 = (st >= 0.78) & part
    ic0 = float(np.median(img0[core78])) if np.any(core78) else 100.0
    ib0 = float(np.median(img0[dt2 <= -0.6])) if np.any(dt2 <= -0.6) else 50.0
    inorm0 = (img0 - ib0) / max(ic0 - ib0, 10.0)
    inorm_c = cc['inorm_c']
    zbn_c = cc['zbn_c']
    an_sm = cc['a_new']
    sub_sm = (dt2 >= -0.45) & (dt2 <= 0.45) & (an_sm >= 0.25) & part
    sub_lo = (dt2 >= -0.50) & (dt2 <= 0.00) & (an_sm >= 0.25) & part
    sub_hi = (dt2 > 0.00) & (dt2 <= 0.50) & (an_sm >= 0.25) & part
    sc_base = (
        0.12 * cc['p3c']
        + 0.20 * cc['p3f_raw']
        + 0.08 * cc['p3p_raw']
        + 0.52 * cc['base6']
        + 0.08 * cc['pt7']
        + 0.05 * cc['rim']
        - 0.015 * np.clip(dt2 - 1.0, 0.0, 2.5)
        + 0.05 * cc['sub_in']
        - 0.06 * cc['sub_zb']
    )
    pr51 = part & (sc_base >= 0.51)
    p3f = cc['p3f_raw']
    p3c = cc['p3c']

    def _m(arr, mask, default=0.0):
        return float(np.mean(arr[mask])) if np.any(mask) else float(default)

    return np.array(
        [
            _m(p2h_6 - p1_6, sub),  # 0
            _m(p2hb_6 - p1b_6, sub),  # 1
            _m(p2h_6 - p1_6, bnd),  # 2
            _m(p2h_6 - p1_6, sup),  # 3
            float(np.sum((p1_6 >= 0.50) & part) / max(np.sum((p2h_6 >= 0.50) & part), 1.0)),  # 4
            float(np.sum((p2r_7 >= 0.50) & part) / max(np.sum((p2h_6 >= 0.50) & part), 1.0)),  # 5
            _m(p2r_7 - p2h_6, bnd),  # 6
            _m(p2r_7 - p2h_6, sub),  # 7
            _m(s_bot, sub_s, 0.5),  # 8
            _m(ptb, sub_s, 0.1),  # 9
            _m(dt2, b_sh, 0.5),  # 10
            float(contrast),  # 11
            float(p['dth_t']),  # 12
            _m(dt2 - dt0, sub_s),  # 13
            _m(dt2 <= 0.45, pr, 0.2),  # 14
            _m(dt2 <= 0.60, pr, 0.3),  # 15
            _m(dt2 / np.maximum(2.0 * edt, 0.8), pr, 0.5),  # 16
            _m(dt2 / np.maximum(2.0 * edt, 0.8), l_th, 0.5),  # 17
            _m(zbn, part_sub & pr, 0.5),  # 18
            float(np.log10(max(float(p['p99']), 10.0))),  # 19
            _m(inorm_full, m_in1, 0.3),  # 20
            _m(inorm_full, m_out1, 0.6),  # 21
            dt2_at_grad,  # 22
            _m(a_new_full - a_old_full, m_out1),  # 23
            (float(np.sum(pr_sup)) - float(np.sum(p3c_raw[part_sup]))) / n_pr,  # 24
            (float(np.sum(pr_sup)) - float(np.sum(p3f_sm_c[part_sup]))) / n_pr,  # 25
            _m(s - pt7, pr_sup) - _m(s - pt7, pr),  # 26
            float(np.sum(pr)) / max(float(np.sum(an[part])), 1.0),  # 27
            _m(p2h_6 - p1_6, sub) - _m(p2h_6 - p1_6, core),  # 28
            _m(p2hb_6 - p1b_6, sub) - _m(p2hb_6 - p1b_6, core),  # 29
            _m(p2h_6 - p1_6, bnd) - _m(p2h_6 - p1_6, core),  # 30
            _m(p3f - p3c, b_sh),  # 31
            0.0,  # 32
            _m(p3f - p3c, sub_sm),  # 33
            0.0,  # 34
            0.0,  # 35
            _m(inorm_c, sub_lo) - _m(inorm_c, sub_hi) if (np.any(sub_lo) and np.any(sub_hi)) else 0.0,  # 36
            _m(inorm0, sub_lo) - _m(inorm0, sub_hi) if (np.any(sub_lo) and np.any(sub_hi)) else 0.0,  # 37
            _m(zbn_c, sub_lo) - _m(zbn_c, sub_hi) if (np.any(sub_lo) and np.any(sub_hi)) else 0.0,  # 38
            _m(inorm_c, sub_sm, 0.5),  # 39
            _m(zbn_c, sub_sm, 0.5),  # 40
            _m(sc_base, b_sh, 0.5),  # 41
            _m(sc_base, sub_sm, 0.5),  # 42
            _m(sc_base, sup, 0.5),  # 43
            float(np.sum(pr51) / max(float(np.sum(an_sm[part])), 1.0)),  # 44
        ],
        dtype=np.float64,
    )


def extract_clean_calib_feats(
    prep_list, s3_dict, f3_g_list, f3_c_list, pure_list, matched_list=None, C_list=None
):
    return np.array(
        [
            extract_clean_calib_feats_single(
                prep_list[i],
                s3_dict['glob_declamp'][i],
                s3_dict['ctx_declamp'][i],
                s3_dict['cand'][i],
                f3_g_list[i],
                f3_c_list[i],
                pure_list[i],
                None if matched_list is None else matched_list[i],
                None if C_list is None else C_list[i],
            )
            for i in range(len(prep_list))
        ],
        dtype=np.float64,
    )


def predict_tibia_roi_from_components(cc, cand, th_val, bp_val):
    sc_base = (
        0.12 * cc['p3c']
        + 0.20 * cc['p3f_raw']
        + 0.08 * cc['p3p_raw']
        + 0.52 * cc['base6']
        + 0.08 * cc['pt7']
        + 0.05 * cc['rim']
        - 0.015 * np.clip(cc['dt2'] - 1.0, 0.0, 2.5)
        + 0.05 * cc['sub_in']
        - 0.06 * cc['sub_zb']
    )
    sc = sc_base - float(bp_val) * (cc['ptb'] - _PTB_0_CALIB)
    pr_1d = cc['part'] & (sc >= float(th_val))
    pr_3d = np.zeros(cand.shape, dtype=bool)
    pr_3d[cand] = pr_1d
    return fast_keep_top_k(pr_3d, k=4, min_size=50)


def _p1_folds_screened(geo, clfs, is_v7=False, return_folds=True):
    sh = geo['shell']
    if not np.any(sh):
        z4 = np.zeros((*sh.shape, 4), dtype=np.float32)
        z4[..., 0] = 1.0
        return z4, ([z4.copy() for _ in clfs] if return_folds else [])
    ext_fn = extract_features_tibia_v7 if is_v7 else extract_features_v6
    X = ext_fn(geo, sh, rng_jit=None)
    p0 = clfs[0].predict_proba(X).astype(np.float32)
    act = (p0[:, 2] >= 0.025) | (p0[:, 3] >= 0.025) | ((p0[:, 1] >= 0.025) & (p0[:, 1] <= 0.96))
    p_folds = [p0]
    if np.any(act):
        X_act = X[act]
        for c in clfs[1:]:
            pf = p0.copy()
            pf[act] = c.predict_proba(X_act).astype(np.float32)
            p_folds.append(pf)
        del X_act
    else:
        for _ in clfs[1:]:
            p_folds.append(p0.copy())
    del X
    p_avg = (sum(p_folds) / float(len(clfs))).astype(np.float32)
    pm_avg = np.zeros((*sh.shape, 4), dtype=np.float32)
    pm_avg[..., 0] = 1.0
    pm_avg[sh] = p_avg
    pm_f_list = []
    if return_folds:
        for pf in p_folds:
            v = np.zeros((*sh.shape, 4), dtype=np.float32)
            v[..., 0] = 1.0
            v[sh] = pf
            pm_f_list.append(v)
    return pm_avg, pm_f_list


def _to_strided_boundary_safe(pm, sh, xp=0):
    out = pm.copy()
    n_x = sh.shape[2]
    if xp == 0:
        even_lim = n_x - 1
        sh_even = sh[:, :, 0:even_lim:2]
        sh_odd = sh[:, :, 1::2]
        copy_m = sh_even & sh_odd
        out[:, :, 1::2][copy_m] = pm[:, :, 0:even_lim:2][copy_m]
    else:
        odd_lim = n_x - 1 if (n_x % 2 == 0) else n_x - 2
        if odd_lim >= 1:
            sh_odd = sh[:, :, 1:odd_lim:2]
            sh_next = sh[:, :, 2 : odd_lim + 1 : 2]
            copy_m = sh_odd & sh_next
            out[:, :, 2 : odd_lim + 1 : 2][copy_m] = pm[:, :, 1:odd_lim:2][copy_m]
    out[~sh] = [1.0, 0.0, 0.0, 0.0]
    return out


def _p2_screened(geo, pm1_s, pm1_o, clfs_hgb, clf_et, roi_slice=None):
    sh = geo['shell']
    act = sh & ((pm1_s[..., 2] >= 0.012) | (pm1_s[..., 3] >= 0.012) | (pm1_o[..., 2] >= 0.02))
    if roi_slice is not None:
        roi_m = np.zeros_like(act, dtype=bool)
        roi_m[roi_slice] = True
        act = act & roi_m
    if not np.any(act):
        return pm1_s.copy(), pm1_s.copy(), pm1_s.copy()
    X2 = extract_features_v6(geo, act, pm_self=pm1_s, pm_other=pm1_o, rng_jit=None)
    p0 = clfs_hgb[0].predict_proba(X2).astype(np.float32)
    bnd = (p0[:, 2] >= 0.015) & (p0[:, 2] <= 0.985)
    p_hgb = p0.copy()
    if len(clfs_hgb) > 1 and np.any(bnd):
        X2_b = X2[bnd]
        p_hgb[bnd] = (p0[bnd] + sum(c.predict_proba(X2_b) for c in clfs_hgb[1:])) / float(len(clfs_hgb))
    p_et = p_hgb.copy()
    if clf_et is not None and np.any(bnd):
        with joblib.parallel_backend('threading'):
            p_et[bnd] = clf_et.predict_proba(X2[bnd]).astype(np.float32)
    del X2
    pm2_hgb = pm1_s.copy()
    pm2_hgb[act] = p_hgb
    pm2_et = pm1_s.copy()
    pm2_et[act] = p_et
    w_et = 0.22
    pm2 = pm1_s.copy()
    pm2[act] = (1.0 - w_et) * p_hgb + w_et * p_et
    return pm2, pm2_hgb, pm2_et


def predict_p2_hybrid(geo, pm1_self, pm1_other, clfs, is_v7=False, only_rim=False):
    sh = geo['shell']
    p1c = pm1_self[..., 2]
    m_base = sh & ((p1c >= 0.14) | (np.roll(p1c, 1, axis=1) >= 0.24) | (np.roll(p1c, -1, axis=1) >= 0.24))
    m_even = np.zeros_like(m_base)
    m_even[:, :, ::2] = m_base[:, :, ::2]
    m_bnd = m_base & ((p1c >= 0.22) & (p1c <= 0.78))
    m_hyb = m_even | m_bnd

    if not np.any(m_hyb):
        if not is_v7 or only_rim:
            return pm1_self.copy()
        return pm1_self.copy(), pm1_self.copy(), pm1_self.copy()

    if not is_v7:
        X2 = extract_features_v6(geo, m_hyb, pm_self=pm1_self, pm_other=pm1_other)
        p2_sub = (sum(c.predict_proba(X2) for c in clfs) / float(len(clfs))).astype(np.float32)
        del X2
        pm2 = pm1_self.copy()
        pm2[m_hyb] = p2_sub
        odd_fill = m_base & (~m_hyb)
        zs, ys, xs = np.where(odd_fill)
        xm1 = np.maximum(xs - 1, 0)
        xp1 = np.minimum(xs + 1, pm2.shape[2] - 1)
        pm2[zs, ys, xs] = 0.5 * (pm2[zs, ys, xm1] + pm2[zs, ys, xp1])
        return pm2
    else:
        X2 = extract_features_tibia_v7(geo, m_hyb, pm_self=pm1_self, pm_other=pm1_other)
        h1, h2, h3, hm, hl, hr = clfs
        p_r = hr.predict_proba(X2).astype(np.float32)
        odd_fill = m_base & (~m_hyb)
        zs, ys, xs = np.where(odd_fill)
        xm1 = np.maximum(xs - 1, 0)
        xp1 = np.minimum(xs + 1, pm1_self.shape[2] - 1)
        if only_rim:
            del X2
            vol = pm1_self.copy()
            vol[m_hyb] = p_r
            vol[zs, ys, xs] = 0.5 * (vol[zs, ys, xm1] + vol[zs, ys, xp1])
            return vol
        p_g = (h1.predict_proba(X2) + h2.predict_proba(X2) + h3.predict_proba(X2)) / 3.0
        p_m = hm.predict_proba(X2)
        p_l = hl.predict_proba(X2)
        del X2
        inds_sh = np.argwhere(m_hyb)
        med_low_z = geo['ml_info'][0]
        z_med_split = geo['ml_info'][2][4]
        sp = geo['sp']
        w_hz = np.clip(0.5 + ((inds_sh[:, 0] * sp[0]) - z_med_split) / 4.0, 0.0, 1.0)[:, None].astype(np.float32)
        w_lat_sh = w_hz if med_low_z else (1.0 - w_hz)
        p_comp = (1.0 - w_lat_sh) * p_m + w_lat_sh * p_l
        p_all = (0.55 * p_g + 0.25 * p_comp + 0.20 * p_r).astype(np.float32)

        outs = []
        for arr_sub in (p_all, p_comp.astype(np.float32), p_r):
            vol = pm1_self.copy()
            vol[m_hyb] = arr_sub
            vol[zs, ys, xs] = 0.5 * (vol[zs, ys, xm1] + vol[zs, ys, xp1])
            outs.append(vol)
        return outs[0], outs[1], outs[2]



def _vol_fingerprint_v6(img):
    flat = img.ravel()
    step = max(1, len(flat) // 64)
    return (img.shape, float(flat[::step].sum()), float(flat[:100].mean()))


def _normalize_bone_masks_dict(image_shape, bone_masks):
    if isinstance(bone_masks, dict):
        return {
            'femur': (bone_masks['femur'] > 0).astype(np.uint8),
            'tibia': (bone_masks['tibia'] > 0).astype(np.uint8),
        }
    bm_arr = np.asarray(bone_masks).reshape(image_shape)
    return {
        'femur': (bm_arr == 1).astype(np.uint8),
        'tibia': ((bm_arr == 2) | (bm_arr == 3)).astype(np.uint8),
    }


def predict_cartilage_v6_p1(image, bone_masks, sp, p1_models, atlas_priors, model_dir=None):
    from .bone_rf import get_or_compute_bone_v2_aux

    bm_v2 = _normalize_bone_masks_dict(image.shape, bone_masks)
    if not np.any(bm_v2['femur']) or not np.any(bm_v2['tibia']):
        y_pred = np.zeros(image.shape, dtype=np.uint8)
        pv = np.zeros((*image.shape, 3), dtype=np.float32)
        pv[..., 0] = 1.0
        return y_pred, pv

    fp = _vol_fingerprint_v6(image)
    bm_fp = (_vol_fingerprint_v6(bm_v2['femur']), _vol_fingerprint_v6(bm_v2['tibia']))
    bone_aux = get_or_compute_bone_v2_aux(image, sp, bm_v2=bm_v2, model_dir=model_dir)
    bm_v1 = bone_aux.get('bm_v1', bm_v2) if isinstance(bone_aux, dict) else bm_v2
    bm_raw = bone_aux.get('bm_raw', bm_v2) if isinstance(bone_aux, dict) else bm_v2

    df2 = fast_signed_distance_transform(bm_v2['femur'] > 0, sp)
    dt2 = fast_signed_distance_transform(bm_v2['tibia'] > 0, sp)
    df1 = df2 if (bm_v1['femur'] is bm_v2['femur']) else fast_signed_distance_transform(bm_v1['femur'] > 0, sp)
    dt1 = dt2 if (bm_v1['tibia'] is bm_v2['tibia']) else fast_signed_distance_transform(bm_v1['tibia'] > 0, sp)
    df0 = df2 if (bm_raw['femur'] is bm_v2['femur']) else fast_signed_distance_transform(bm_raw['femur'] > 0, sp)
    dt0 = dt2 if (bm_raw['tibia'] is bm_v2['tibia']) else fast_signed_distance_transform(bm_raw['tibia'] > 0, sp)
    dts_v6 = (df2, dt2, df1, dt1, df0, dt0)

    f_st2, _, f_cp2 = compute_articular_landmarks_and_stats(bm_v2['femur'], df2, dt2, 45.0, sp)
    t_st2, _, t_cp2 = compute_articular_landmarks_and_stats(bm_v2['tibia'], dt2, df2, 25.0, sp)
    ml2 = compute_ml_frames(bm_v2, df2, dt2, sp)
    lm_v6 = (f_st2, f_cp2, t_st2, t_cp2, ml2)

    geo_t6 = prepare_geometry_v6(
        image, sp, bm_v2, bm_v1, bm_raw, 'tibia', atlas_priors, dts_pre=dts_v6, landmarks_pre=lm_v6
    )
    geo_f6 = prepare_geometry_v6(
        image, sp, bm_v2, bm_v1, bm_raw, 'femur', atlas_priors, dts_pre=dts_v6, landmarks_pre=lm_v6
    )
    pm1_t6, pm1_t6_f = _p1_folds_screened(geo_t6, p1_models['tibia'], is_v7=False)
    pm1_f6, pm1_f6_f = _p1_folds_screened(geo_f6, p1_models['femur'], is_v7=False)

    while len(_CART_P1_SESSION_CACHE) >= 8:
        _CART_P1_SESSION_CACHE.pop(next(iter(_CART_P1_SESSION_CACHE)), None)
    _CART_P1_SESSION_CACHE[fp] = {
        'bm_fp': bm_fp,
        'dts': dts_v6,
        'geo_t6': geo_t6,
        'geo_f6': geo_f6,
        'pm1_t6': pm1_t6,
        'pm1_f6': pm1_f6,
        'pm1_t6_f': pm1_t6_f,
        'pm1_f6_f': pm1_f6_f,
        'bm_v2': bm_v2,
        'bm_v1': bm_v1,
        'bm_raw': bm_raw,
    }

    pv = np.zeros((*image.shape, 3), dtype=np.float32)
    pv[..., 1] = pm1_f6[..., 2]
    pv[..., 2] = pm1_t6[..., 2]
    s_fg = pv[..., 1] + pv[..., 2]
    over = s_fg > 1.0
    if np.any(over):
        pv[over, 1] /= s_fg[over]
        pv[over, 2] /= s_fg[over]
    pv[..., 0] = np.clip(1.0 - pv[..., 1] - pv[..., 2], 0.0, 1.0)

    p_f = gaussian_filter(pv[..., 1], sigma=[0.35 / sp[0], 0.15 / sp[1], 0.25 / sp[2]])
    p_t = gaussian_filter(pv[..., 2], sigma=[0.45 / sp[0], 0.18 / sp[1], 0.35 / sp[2]])
    pred_fc = fast_keep_top_k((p_f >= 0.48) & (p_f >= p_t), k=1, min_size=300)
    pred_tc = fast_keep_top_k((p_t >= 0.46) & (p_t > p_f), k=2, min_size=150)
    y_pred = np.zeros(image.shape, dtype=np.uint8)
    y_pred[pred_fc] = 2
    y_pred[pred_tc] = 4
    return y_pred, pv


def _p2_v7_screened(geo, pm1_self, pm1_other, clfs_hgb, clf_et):
    sh = geo['shell']
    act = sh & ((pm1_self[..., 2] >= 0.012) | (pm1_self[..., 3] >= 0.012) | (pm1_other[..., 2] >= 0.02))
    if not np.any(act):
        c = pm1_self.copy()
        return c, c.copy(), c.copy(), c.copy()
    X2 = extract_features_tibia_v7(geo, act, pm_self=pm1_self, pm_other=pm1_other, rng_jit=None)
    h1, h2, h3, hm, hl, hr = clfs_hgb
    p0 = h1.predict_proba(X2).astype(np.float32)
    bnd = (p0[:, 2] >= 0.015) & (p0[:, 2] <= 0.985)
    p_g = p0.copy()
    p_m = p0.copy()
    p_l = p0.copy()
    p_r = p0.copy()
    p_et = p0.copy()
    if np.any(bnd):
        Xb = X2[bnd]
        p_g[bnd] = ((p0[bnd] + h2.predict_proba(Xb) + h3.predict_proba(Xb)) / 3.0).astype(np.float32)
        p_m[bnd] = hm.predict_proba(Xb).astype(np.float32)
        p_l[bnd] = hl.predict_proba(Xb).astype(np.float32)
        p_r[bnd] = hr.predict_proba(Xb).astype(np.float32)
        if clf_et is not None:
            with joblib.parallel_backend('threading'):
                p_et[bnd] = clf_et.predict_proba(Xb).astype(np.float32)
        del Xb
    del X2

    inds_act = np.argwhere(act)
    med_low_z = geo['ml_info'][0]
    z_med_split = geo['ml_info'][2][4]
    sp = geo['sp']
    w_hz = np.clip(0.5 + ((inds_act[:, 0] * sp[0]) - z_med_split) / 4.0, 0.0, 1.0)[:, None].astype(np.float32)
    w_lat = w_hz if med_low_z else (1.0 - w_hz)
    p_comp = ((1.0 - w_lat) * p_m + w_lat * p_l).astype(np.float32)
    p_hgb_all = (0.55 * p_g + 0.25 * p_comp + 0.20 * p_r).astype(np.float32)

    pm2_hgb = pm1_self.copy(); pm2_hgb[act] = p_hgb_all
    pm2_comp = pm1_self.copy(); pm2_comp[act] = p_comp
    pm2_rim = pm1_self.copy(); pm2_rim[act] = p_r
    pm2_et = pm1_self.copy(); pm2_et[act] = p_et
    return pm2_hgb, pm2_comp, pm2_rim, pm2_et


def predict_cartilage_v6_p2(image, bone_masks, sp, meta, prob_map=None, model_dir=None):
    from .bone_rf import get_or_compute_bone_v2_aux

    bm_v2 = _normalize_bone_masks_dict(image.shape, bone_masks)
    if not np.any(bm_v2['femur']) or not np.any(bm_v2['tibia']):
        y_pred = np.zeros(image.shape, dtype=np.uint8)
        pv = np.zeros((*image.shape, 3), dtype=np.float32)
        pv[..., 0] = 1.0
        return y_pred, pv

    fp = _vol_fingerprint_v6(image)
    bm_fp = (_vol_fingerprint_v6(bm_v2['femur']), _vol_fingerprint_v6(bm_v2['tibia']))
    cached_p1 = _CART_P1_SESSION_CACHE.get(fp, None)
    if cached_p1 is not None and cached_p1.get('bm_fp') == bm_fp:
        bm_v1 = cached_p1.get('bm_v1', bm_v2)
        bm_raw = cached_p1.get('bm_raw', bm_v2)
    else:
        cached_p1 = None
        bone_aux = get_or_compute_bone_v2_aux(image, sp, bm_v2=bm_v2, model_dir=model_dir)
        bm_v1 = bone_aux.get('bm_v1', bm_v2) if isinstance(bone_aux, dict) else bm_v2
        bm_raw = bone_aux.get('bm_raw', bm_v2) if isinstance(bone_aux, dict) else bm_v2

    m_v6 = meta['m_v6']
    ap_v6 = m_v6['atlas_priors']

    if cached_p1 is not None:
        df, dt, df1, dt1_full, df0, dt0_full = cached_p1['dts']
        geo_t6 = cached_p1['geo_t6']
        geo_f6 = cached_p1['geo_f6']
        pm1_t6 = cached_p1['pm1_t6']
        pm1_f6 = cached_p1['pm1_f6']
        pm1_t6_f = cached_p1['pm1_t6_f']
        pm1_f6_f = cached_p1['pm1_f6_f']
    else:
        df = fast_signed_distance_transform(bm_v2['femur'] > 0, sp)
        dt = fast_signed_distance_transform(bm_v2['tibia'] > 0, sp)
        df1 = df if (bm_v1['femur'] is bm_v2['femur']) else fast_signed_distance_transform(bm_v1['femur'] > 0, sp)
        dt1_full = dt if (bm_v1['tibia'] is bm_v2['tibia']) else fast_signed_distance_transform(bm_v1['tibia'] > 0, sp)
        df0 = df if (bm_raw['femur'] is bm_v2['femur']) else fast_signed_distance_transform(bm_raw['femur'] > 0, sp)
        dt0_full = dt if (bm_raw['tibia'] is bm_v2['tibia']) else fast_signed_distance_transform(bm_raw['tibia'] > 0, sp)
        dts_v6 = (df, dt, df1, dt1_full, df0, dt0_full)
        f_st2, _, f_cp2 = compute_articular_landmarks_and_stats(bm_v2['femur'], df, dt, 45.0, sp)
        t_st2, _, t_cp2 = compute_articular_landmarks_and_stats(bm_v2['tibia'], dt, df, 25.0, sp)
        ml2 = compute_ml_frames(bm_v2, df, dt, sp)
        lm_v6 = (f_st2, f_cp2, t_st2, t_cp2, ml2)
        geo_t6 = prepare_geometry_v6(
            image, sp, bm_v2, bm_v1, bm_raw, 'tibia', ap_v6, dts_pre=dts_v6, landmarks_pre=lm_v6
        )
        geo_f6 = prepare_geometry_v6(
            image, sp, bm_v2, bm_v1, bm_raw, 'femur', ap_v6, dts_pre=dts_v6, landmarks_pre=lm_v6
        )
        pm1_t6, pm1_t6_f = _p1_folds_screened(geo_t6, m_v6['p1_folds']['tibia'], is_v7=False)
        pm1_f6, pm1_f6_f = _p1_folds_screened(geo_f6, m_v6['p1_folds']['femur'], is_v7=False)
        while len(_CART_P1_SESSION_CACHE) >= 8:
            _CART_P1_SESSION_CACHE.pop(next(iter(_CART_P1_SESSION_CACHE)), None)
        _CART_P1_SESSION_CACHE[fp] = {
            'bm_fp': bm_fp,
            'dts': dts_v6,
            'geo_t6': geo_t6,
            'geo_f6': geo_f6,
            'pm1_t6': pm1_t6,
            'pm1_f6': pm1_f6,
            'pm1_t6_f': pm1_t6_f,
            'pm1_f6_f': pm1_f6_f,
            'bm_v2': bm_v2,
            'bm_v1': bm_v1,
            'bm_raw': bm_raw,
        }

    pm2_t6, pm2_t6_hgb, pm2_t6_et = _p2_screened(
        geo_t6, pm1_t6, pm1_f6, m_v6['p2_tibia_hgb'], m_v6['p2_tibia_et']
    )
    pm2_f6, pm2_f6_hgb, pm2_f6_et = _p2_screened(
        geo_f6, pm1_f6, pm1_t6, m_v6['p2_femur_hgb'], m_v6['p2_femur_et']
    )

    union_sh = geo_t6['shell'] | geo_f6['shell']
    inds_u = np.argwhere(union_sh)
    uz0, uy0, ux0 = np.maximum(inds_u.min(axis=0) - 4, 0)
    uz1, uy1, ux1 = np.minimum(inds_u.max(axis=0) + 5, image.shape)
    sl = (slice(uz0, uz1), slice(uy0, uy1), slice(ux0, ux1))

    w_et = 0.22
    pm_f = (
        0.10 * pm1_f6[sl].astype(np.float32)
        + (0.90 * (1.0 - w_et)) * pm2_f6_hgb[sl].astype(np.float32)
        + (0.90 * w_et) * pm2_f6_et[sl].astype(np.float32)
    )
    pm_t = (
        0.10 * pm1_t6[sl].astype(np.float32)
        + (0.90 * (1.0 - w_et)) * pm2_t6_hgb[sl].astype(np.float32)
        + (0.90 * w_et) * pm2_t6_et[sl].astype(np.float32)
    )

    f_stats = geo_t6['f_stats']
    t_comp = geo_t6['t_comp']
    ml_info = geo_t6['ml_info']

    df_c = df[sl]
    dt_c = dt[sl]
    dfs_c = gaussian_filter(df_c, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
    dts_c = gaussian_filter(dt_c, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
    zz, yy, xx = np.meshgrid(
        np.arange(uz0, uz1), np.arange(uy0, uy1), np.arange(ux0, ux1), indexing='ij'
    )
    coords_c = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(np.float32) * np.array(
        sp, dtype=np.float32
    )
    bins_c, _ = compute_atlas_bins_v6(coords_c, dfs_c.ravel(), dts_c.ravel(), f_stats, t_comp, ml_info)
    sh_c = df_c.shape
    a_old_fc = ap_v6['pf_old_fc'][bins_c['f_old']].reshape(sh_c)
    a_new_fc = (
        0.45 * ap_v6['pf_fml_fc'][bins_c['f_ml']]
        + 0.35 * ap_v6['pf_sep_fc'][bins_c['f_sep']]
        + 0.20 * ap_v6['pf_old_fc'][bins_c['f_old']]
    ).reshape(sh_c)
    a_old_tc = ap_v6['pt_old_tc'][bins_c['t_old']].reshape(sh_c)
    a_new_tc = (
        0.32 * ap_v6['pt_ml_tc'][bins_c['t_ml']]
        + 0.30 * ap_v6['pt_sep_tc'][bins_c['t_sep']]
        + 0.20 * ap_v6['pt_mly_tc'][bins_c['t_mly']]
        + 0.18 * ap_v6['pt_wide_tc'][bins_c['t_wide']]
    ).reshape(sh_c)

    b_pen = 0.22
    w_del_f = 0.25
    th_f = 0.50
    sig_f = [0.25 / sp[0], 0.10 / sp[1], 0.18 / sp[2]]
    sig_t = [0.35 / sp[0], 0.14 / sp[1], 0.25 / sp[2]]
    pf_v6 = gaussian_filter(pm_f[..., 2], sigma=sig_f)
    pfb_v6 = gaussian_filter(pm_f[..., 1], sigma=sig_f)
    pt_v6 = gaussian_filter(pm_t[..., 2], sigma=sig_t)
    ptb_v6 = gaussian_filter(pm_t[..., 1], sigma=sig_t)

    del_f = a_new_fc - a_old_fc
    del_t = a_new_tc - a_old_tc
    sf_part_v6 = pf_v6 - b_pen * pfb_v6 + w_del_f * del_f
    st_part_v6 = pt_v6 - b_pen * ptb_v6 + 0.35 * del_t
    sf_v6 = sf_part_v6 + 0.28 * np.minimum(pf_v6, pt_v6)

    pf_50 = float(
        np.sum((pf_v6 - b_pen * pfb_v6) >= th_f + 0.05 * np.clip(df_c - 0.8, 0.0, 2.5))
    )
    af_sum = max(float(np.sum(a_new_fc)), 1e-5)
    pf_sum = max(float(np.sum(pf_v6[pf_v6 >= 0.05])), 1e-5)
    ind_f = 0.5 * ((pf_50 / pf_sum - 0.815) / 0.035) + 0.5 * ((pf_50 / af_sum - 0.98) / 0.10)
    dth_f = np.clip(-0.04 * 0.60 * ind_f, -0.09, 0.09)
    pred_fc_roi = fast_keep_top_k(
        (sf_v6 >= th_f + dth_f) & (sf_part_v6 >= st_part_v6), k=1, min_size=300
    )

    m_v5 = meta.get('m_v5')
    m_v7 = meta.get('m_v7')
    s3_clean = meta.get('s3_clean')
    f3_models = meta.get('f3_models')
    pure_models = meta.get('pure_models')
    calib_model = meta.get('calib_model')

    sub_bbox = None
    pred_tc_roi = None

    if (
        m_v5 is not None
        and m_v7 is not None
        and s3_clean is not None
        and f3_models is not None
        and pure_models is not None
        and calib_model is not None
    ):
        shell_t = (dt >= -0.95) & (dt <= 4.6) & (df >= -1.0) & (df <= 25.0)
        inds_t = np.argwhere(shell_t)
        if len(inds_t) > 0:
            margin = [int(np.ceil(6.0 / s)) for s in sp]
            z0 = max(uz0, int(inds_t[:, 0].min()) - margin[0])
            z1 = min(uz1, int(inds_t[:, 0].max()) + 5)
            y0 = max(uy0, int(inds_t[:, 1].min()) - margin[1])
            y1 = min(uy1, int(inds_t[:, 1].max()) + 5)
            x0 = max(ux0, int(inds_t[:, 2].min()) - margin[2])
            x1 = min(ux1, int(inds_t[:, 2].max()) + 5)
        else:
            z0, z1, y0, y1, x0, x1 = uz0, uz1, uy0, uy1, ux0, ux1
        sub_bbox = (z0 - uz0, z1 - uz0, y0 - uy0, y1 - uy0, x0 - ux0, x1 - ux0)
        sl_sub = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        sub_sl = (slice(z0 - uz0, z1 - uz0), slice(y0 - uy0, y1 - uy0), slice(x0 - ux0, x1 - ux0))

        ap_v5 = m_v5['atlas_priors']
        dts_v5 = (df1, dt1_full, df, dt, df0, dt0_full)
        f_st1, _, f_cp1 = compute_articular_landmarks_and_stats(bm_v1['femur'], df1, dt1_full, 45.0, sp)
        t_st1, _, t_cp1 = compute_articular_landmarks_and_stats(bm_v1['tibia'], dt1_full, df1, 25.0, sp)
        ml1 = compute_ml_frames(bm_v1, df1, dt1_full, sp)
        lm_v5 = (f_st1, f_cp1, t_st1, t_cp1, ml1)
        geo_t5 = prepare_geometry_v6(
            image, sp, bm_v1, bm_v2, bm_raw, 'tibia', ap_v5, dts_pre=dts_v5, landmarks_pre=lm_v5
        )
        geo_f5 = prepare_geometry_v6(
            image, sp, bm_v1, bm_v2, bm_raw, 'femur', ap_v5, dts_pre=dts_v5, landmarks_pre=lm_v5
        )
        pm1_t5, _ = _p1_folds_screened(geo_t5, m_v5['p1_folds']['tibia'], is_v7=False, return_folds=False)
        pm1_f5, _ = _p1_folds_screened(geo_f5, m_v5['p1_folds']['femur'], is_v7=False, return_folds=False)
        _, pm2_t5_hgb, pm2_t5_et = _p2_screened(
            geo_t5, pm1_t5, pm1_f5, m_v5['p2_tibia_hgb'], m_v5['p2_tibia_et']
        )
        _, pm2_f5_hgb, pm2_f5_et = _p2_screened(
            geo_f5, pm1_f5, pm1_t5, m_v5['p2_femur_hgb'], m_v5['p2_femur_et'], roi_slice=sl_sub
        )

        geo_t7 = add_v7_fields(dict(geo_t6))
        pm1_t7, pm1_t7_f = _p1_folds_screened(geo_t7, m_v7['p1_folds_v7'], is_v7=True)
        pm1_t67 = 0.45 * pm1_t6 + 0.55 * pm1_t7
        pm2_t7_hgb, pm2_t7_comp, pm2_t7_rim, pm2_t7_et = _p2_v7_screened(
            geo_t7, pm1_t67, pm1_f6, m_v7['p2_tibia_hgb'], m_v7['p2_tibia_et']
        )

        df2_sub = df[sl_sub]
        dt2_sub = dt[sl_sub]
        dt1_sub = dt1_full[sl_sub]
        dt0_sub = dt0_full[sl_sub]

        dts_sub = gaussian_filter(dt2_sub, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
        _, nyt_sub, _ = np.gradient(dts_sub, *sp)

        edt_ml_v = ap_v6['edt_ml_2d'][bins_c['t_ml'][0], bins_c['t_ml'][1]]
        edt_sep_v = ap_v6['edt_sep_2d'][bins_c['t_sep'][0], bins_c['t_sep'][1], bins_c['t_sep'][2]]
        edt_wide_v = ap_v6['edt_wide_2d'][bins_c['t_wide'][0], bins_c['t_wide'][1]]
        edt_ens_full = (0.40 * edt_ml_v + 0.35 * edt_sep_v + 0.25 * edt_wide_v).reshape(sh_c)
        edt_ens_sub = edt_ens_full[sub_sl]
        a_old_tc_sub = a_old_tc[sub_sl]
        a_new_tc_sub = a_new_tc[sub_sl]

        img_sub = image[sl_sub].astype(np.float32)
        img_f0_sub = gaussian_filter(img_sub, sigma=[0.35 / s for s in sp])
        img_lat_sub = gaussian_filter(img_sub, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
        sig_bn_sub = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
        w_bn_sub = ((dt2_sub <= -1.4) & (dt2_sub >= -5.5)).astype(np.float32) + 1e-5
        w_bn_s = gaussian_filter(w_bn_sub, sigma=sig_bn_sub)
        i_bn_mean = gaussian_filter(img_lat_sub * w_bn_sub, sigma=sig_bn_sub) / w_bn_s
        i_bn_var = gaussian_filter(((img_lat_sub - i_bn_mean) ** 2) * w_bn_sub, sigma=sig_bn_sub) / w_bn_s
        i_bn_std = np.maximum(np.sqrt(np.maximum(i_bn_var, 0.0)), 3.0)
        zb_sub = ((img_lat_sub - i_bn_mean) / i_bn_std).astype(np.float32)

        w_et_t = 0.18
        pm_t6_s = (
            0.10 * pm1_t6[sl_sub]
            + (0.90 * (1.0 - w_et_t)) * pm2_t6_hgb[sl_sub]
            + (0.90 * w_et_t) * pm2_t6_et[sl_sub]
        )
        pm_f6_s = (
            0.10 * pm1_f6[sl_sub]
            + (0.90 * (1.0 - w_et_t)) * pm2_f6_hgb[sl_sub]
            + (0.90 * w_et_t) * pm2_f6_et[sl_sub]
        )
        pm_t5_s = (
            0.10 * pm1_t5[sl_sub]
            + (0.90 * (1.0 - w_et_t)) * pm2_t5_hgb[sl_sub]
            + (0.90 * w_et_t) * pm2_t5_et[sl_sub]
        )
        pm_f5_s = (
            0.10 * pm1_f5[sl_sub]
            + (0.90 * (1.0 - w_et_t)) * pm2_f5_hgb[sl_sub]
            + (0.90 * w_et_t) * pm2_f5_et[sl_sub]
        )
        pm7_s = (
            0.10 * pm1_t7[sl_sub]
            + (0.90 * (1.0 - w_et_t)) * pm2_t7_hgb[sl_sub]
            + (0.90 * w_et_t) * pm2_t7_et[sl_sub]
        )

        pm_t_prep = 0.70 * pm_t6_s + 0.30 * pm_t5_s
        pm_f_prep = 0.70 * pm_f6_s + 0.30 * pm_f5_s

        pt_bg = gaussian_filter(pm_t_prep[..., 0], sigma=sig_t)
        ptb = gaussian_filter(pm_t_prep[..., 1], sigma=sig_t)
        pt = gaussian_filter(pm_t_prep[..., 2], sigma=sig_t)
        pt_ot = gaussian_filter(pm_t_prep[..., 3], sigma=sig_t)

        pf_bg = gaussian_filter(pm_f_prep[..., 0], sigma=sig_f)
        pfb = gaussian_filter(pm_f_prep[..., 1], sigma=sig_f)
        pf = gaussian_filter(pm_f_prep[..., 2], sigma=sig_f)
        pf_ot = gaussian_filter(pm_f_prep[..., 3], sigma=sig_f)

        pt6_s = gaussian_filter(pm_t6_s[..., 2], sigma=sig_t)
        ptb6_s = gaussian_filter(pm_t6_s[..., 1], sigma=sig_t)
        pt5_s = gaussian_filter(pm_t5_s[..., 2], sigma=sig_t)
        ptb5_s = gaussian_filter(pm_t5_s[..., 1], sigma=sig_t)
        pt7_s = gaussian_filter(pm7_s[..., 2], sigma=sig_t)
        ptb7_s = gaussian_filter(pm7_s[..., 1], sigma=sig_t)

        b_pen_t = 0.20
        del_t_sub = a_new_tc_sub - a_old_tc_sub
        sf_part = pf - b_pen_t * pfb
        st_part = pt - b_pen_t * ptb + 0.25 * del_t_sub
        sf = sf_part + 0.28 * np.minimum(pf, pt)
        st = st_part + 0.28 * np.minimum(pf, pt)

        pt_52 = float(
            np.sum((pt - b_pen_t * ptb) >= 0.50 + 0.05 * np.clip(dt2_sub - 0.8, 0.0, 2.5))
        )
        at_sum = max(float(np.sum(a_new_tc_sub)), 1e-5)
        pt_sum = max(float(np.sum(pt[pt >= 0.05])), 1e-5)
        ind_t = 0.5 * ((pt_52 / pt_sum - 0.765) / 0.040) + 0.5 * ((pt_52 / at_sum - 0.96) / 0.12)
        dth_t = float(np.clip(-0.04 * 0.45 * ind_t, -0.09, 0.09))
        pred_tc_init = fast_keep_top_k(
            (st >= 0.50 + dth_t) & (st_part > sf_part), k=2, min_size=150
        )

        p = {
            'fname': 'infer',
            'sp': sp,
            'p99': float(np.percentile(image, 99)),
            'fov_y': float(image.shape[1] * sp[1]),
            'fov_z': float(image.shape[0] * sp[0]),
            'orig_shape': image.shape,
            'u_bbox': (uz0, uz1, uy0, uy1, ux0, ux1),
            'sub_bbox': sub_bbox,
            'lbl_c': np.zeros(dt2_sub.shape, dtype=np.uint8),
            'img_f0': img_f0_sub.astype(np.float16),
            'img_lat': img_lat_sub.astype(np.float16),
            'zb': zb_sub.astype(np.float16),
            'df2': df2_sub.astype(np.float16),
            'dt2': dt2_sub.astype(np.float16),
            'dt1': dt1_sub.astype(np.float16),
            'dt0': dt0_sub.astype(np.float16),
            'nyt': nyt_sub.astype(np.float16),
            'a_old_tc': a_old_tc_sub.astype(np.float16),
            'a_new_tc': a_new_tc_sub.astype(np.float16),
            'edt_ens': edt_ens_sub.astype(np.float16),
            'pt_bg': pt_bg.astype(np.float16),
            'ptb': ptb.astype(np.float16),
            'pt': pt.astype(np.float16),
            'pt_ot': pt_ot.astype(np.float16),
            'pf_bg': pf_bg.astype(np.float16),
            'pfb': pfb.astype(np.float16),
            'pf': pf.astype(np.float16),
            'pf_ot': pf_ot.astype(np.float16),
            'pt6': pt6_s.astype(np.float16),
            'ptb6': ptb6_s.astype(np.float16),
            'pt5': pt5_s.astype(np.float16),
            'ptb5': ptb5_s.astype(np.float16),
            'pt7': pt7_s.astype(np.float16),
            'ptb7': ptb7_s.astype(np.float16),
            'p1_tc6': pm1_t6[sl_sub][..., 2].astype(np.float16),
            'p1_tb6': pm1_t6[sl_sub][..., 1].astype(np.float16),
            'p2h_tc6': pm2_t6_hgb[sl_sub][..., 2].astype(np.float16),
            'p2h_tb6': pm2_t6_hgb[sl_sub][..., 1].astype(np.float16),
            'p1_tc7': pm1_t7[sl_sub][..., 2].astype(np.float16),
            'p2h_tc7': pm2_t7_hgb[sl_sub][..., 2].astype(np.float16),
            'p2h_tb7': pm2_t7_hgb[sl_sub][..., 1].astype(np.float16),
            'p2c_tc7': pm2_t7_comp[sl_sub][..., 2].astype(np.float16),
            'p2r_tc7': pm2_t7_rim[sl_sub][..., 2].astype(np.float16),
            'st': st.astype(np.float16),
            'sf': sf.astype(np.float16),
            'st_part': st_part.astype(np.float16),
            'sf_part': sf_part.astype(np.float16),
            'dth_t': dth_t,
            'pred_tc': pred_tc_init,
        }

        X_loc, y_dummy, is_mbot, s3_cand, case_ctx = extract_clean_s3_case(p)
        if np.sum(s3_cand) >= 20:
            n_cand = len(y_dummy)
            Xi_d = X_loc[:, 5:]
            ctx_mat = np.broadcast_to(case_ctx[None, :], (n_cand, len(case_ctx)))
            Xi_ctx = np.hstack([Xi_d, ctx_mat])
            s3_folds = s3_clean['s3_models_folds']
            s3_glob = (
                sum(fd['d1'].predict_proba(Xi_d)[:, 1].astype(np.float32) for fd in s3_folds)
                / float(len(s3_folds))
            ).astype(np.float32)
            s3_ctx = (
                sum(fd['cd1'].predict_proba(Xi_ctx)[:, 1].astype(np.float32) for fd in s3_folds)
                / float(len(s3_folds))
            ).astype(np.float32)
            del Xi_d, Xi_ctx

            # Matched OOF Pass-1 & Pass-2 probabilities in bsl parity
            u_mask = (
                (bm_v2['femur'] > 0)
                | (bm_v2['tibia'] > 0)
                | (bm_v1['femur'] > 0)
                | (bm_v1['tibia'] > 0)
                | (bm_raw['tibia'] > 0)
            )
            uz_b, _, ux_b = np.where(u_mask)
            bsl_x0 = max(0, int(ux_b.min()) - 22)
            xp_6 = bsl_x0 % 2

            pm1_t6_str = [_to_strided_boundary_safe(pf, geo_t6['shell'], xp=xp_6) for pf in pm1_t6_f]
            pm1_f6_str = [_to_strided_boundary_safe(pf, geo_f6['shell'], xp=xp_6) for pf in pm1_f6_f]
            pm1_t7_str = [_to_strided_boundary_safe(pf, geo_t7['shell'], xp=xp_6) for pf in pm1_t7_f]

            pm1_t6_m = sum(pm1_t6_str) / 4.0
            pm2_t6_m = (
                sum(
                    predict_p2_hybrid(
                        geo_t6,
                        pm1_t6_str[f],
                        pm1_f6_str[f],
                        m_v6['p2_tibia_oof_folds'][f],
                        is_v7=False,
                    )
                    for f in range(4)
                )
                / 4.0
            )
            pm2r_t7_m = (
                sum(
                    predict_p2_hybrid(
                        geo_t7,
                        0.45 * pm1_t6_str[f] + 0.55 * pm1_t7_str[f],
                        pm1_f6_str[f],
                        m_v7['p2_tibia_oof_folds'][f],
                        is_v7=True,
                        only_rim=True,
                    )
                    for f in range(4)
                )
                / 4.0
            )
            del pm1_t6_str, pm1_f6_str, pm1_t7_str

            matched_p1p2 = {
                'p1_tc6': pm1_t6_m[sl_sub][..., 2].astype(np.float16),
                'p1_tb6': pm1_t6_m[sl_sub][..., 1].astype(np.float16),
                'p2h_tc6': pm2_t6_m[sl_sub][..., 2].astype(np.float16),
                'p2h_tb6': pm2_t6_m[sl_sub][..., 1].astype(np.float16),
                'p2r_tc7': pm2r_t7_m[sl_sub][..., 2].astype(np.float16),
            }

            f3_g, f3_c, _ = predict_full_oof_s3_case(
                p, f3_models, m_override=matched_p1p2, pre_clean=(X_loc, y_dummy, is_mbot, s3_cand, case_ctx)
            )
            del X_loc

            Xp_m, _ = extract_pure_oof_case({**p, **matched_p1p2}, s3_cand)
            pure_list = pure_models['m_glob'] if isinstance(pure_models, dict) else pure_models
            pure_1d = (
                sum(m.predict_proba(Xp_m)[:, 1].astype(np.float32) for m in pure_list)
                / float(len(pure_list))
            ).astype(np.float32)
            del Xp_m

            cc = compute_case_smoothed_components(p, s3_glob, s3_ctx, s3_cand, f3_g, f3_c, pure_1d)
            f_row = extract_clean_calib_feats_single(
                p, s3_glob, s3_ctx, s3_cand, f3_g, f3_c, pure_1d, matched_p1p2, cc
            )[None, :]
            th_cols = calib_model['th_cols']
            bp_cols = calib_model['bp_cols']
            sc_th = calib_model['sc_th']
            sc_bp = calib_model['sc_bp']
            m_th = calib_model['m_th']
            m_bp = calib_model['m_bp']
            th_base = float(calib_model.get('th_base', 0.506))
            s_th = float(calib_model.get('s_th', 1.65))
            s_bp = float(calib_model.get('s_bp', 1.60))
            th_val = float(th_base + s_th * m_th.predict(sc_th.transform(f_row[:, th_cols]))[0])
            bp_val = float(s_bp * m_bp.predict(sc_bp.transform(f_row[:, bp_cols]))[0])
            pred_tc_roi = predict_tibia_roi_from_components(cc, s3_cand, th_val, bp_val)
        else:
            pred_tc_roi = p['pred_tc']

    if pred_tc_roi is None:
        st_v6 = st_part_v6 + 0.28 * np.minimum(pf_v6, pt_v6)
        pt_52 = float(
            np.sum((pt_v6 - 0.20 * ptb_v6) >= 0.50 + 0.05 * np.clip(dt_c - 0.8, 0.0, 2.5))
        )
        at_sum = max(float(np.sum(a_new_tc)), 1e-5)
        pt_sum = max(float(np.sum(pt_v6[pt_v6 >= 0.05])), 1e-5)
        ind_t = 0.5 * ((pt_52 / pt_sum - 0.765) / 0.040) + 0.5 * ((pt_52 / at_sum - 0.96) / 0.12)
        dth_t = float(np.clip(-0.04 * 0.45 * ind_t, -0.09, 0.09))
        pred_tc_roi = fast_keep_top_k(
            (st_v6 >= 0.50 + dth_t) & (st_part_v6 > sf_part_v6), k=2, min_size=150
        )

    y_pred = np.zeros(image.shape, dtype=np.uint8)
    roi_view = y_pred[sl]
    roi_view[pred_fc_roi] = 2
    if sub_bbox is not None:
        sz0, sz1, sy0, sy1, sx0, sx1 = sub_bbox
        roi_view[sz0:sz1, sy0:sy1, sx0:sx1][pred_tc_roi] = 4
    else:
        roi_view[pred_tc_roi] = 4

    pv = np.zeros((*image.shape, 3), dtype=np.float32)
    pv[sl][..., 1] = np.clip(pm_f[..., 2], 0.0, 1.0)
    pv[sl][..., 2] = np.clip(pm_t[..., 2], 0.0, 1.0)
    pv[y_pred == 2, 1] = np.maximum(pv[y_pred == 2, 1], 0.75)
    pv[y_pred == 4, 2] = np.maximum(pv[y_pred == 4, 2], 0.75)
    s_fg = pv[..., 1] + pv[..., 2]
    over = s_fg > 1.0
    if np.any(over):
        pv[over, 1] /= s_fg[over]
        pv[over, 2] /= s_fg[over]
    pv[..., 0] = np.clip(1.0 - pv[..., 1] - pv[..., 2], 0.0, 1.0)
    return y_pred, pv


