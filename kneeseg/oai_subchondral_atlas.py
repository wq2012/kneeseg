"""3-Bone & 3-Cartilage Subchondral Surface Refiner and Articular Atlas for OAI and Unified SKI10+OAI."""

import os
import joblib
import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
    maximum_filter1d,
    minimum_filter1d,
    zoom,
)

from .features import fast_signed_distance_transform

_OAI_BONE_AUX_CACHE = {}
_OAI_CART_P1_CACHE = {}
_OAI_P1_MODEL_CACHE = {}
_OAI_P2_MODEL_CACHE = {}

SH_OLD = (26, 26, 16, 18)
SH_ML = (34, 26, 16, 18)
SH_MLY = (34, 18, 26, 16)
SH_SEP = (2, 24, 24, 16, 18)
SH_WIDE = (32, 28, 16, 18)
SH_FO = (22, 26, 26, 16)
SH_FML = (34, 26, 26, 16)
SH_FSEP = (2, 22, 26, 26, 16)
SH_PO = (20, 24, 16, 18)
SH_PML = (28, 24, 16, 18)
SH_PSEP = (2, 18, 24, 16, 18)

_D_GRID = np.array([-3.6, -2.8, -2.0, -1.4, -0.9, -0.45, 0.0, 0.45, 0.9, 1.4, 2.0, 2.7, 3.5, 4.5, 5.6], dtype=np.float32)
_H_GRID = np.array([-1.0, -0.4, 0.0, 0.5, 1.0, 1.6, 2.2, 3.0, 4.0, 5.2], dtype=np.float32)
_DY_STEPS = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=int)
_DX_STEPS = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=int)
_R_STEPS = np.array([-6.0, -3.5, -1.8, 1.8, 3.5, 6.0], dtype=np.float32)


def _vol_fingerprint_oai(img):
    flat = np.asarray(img).ravel()
    step = max(1, len(flat) // 64)
    return (img.shape, float(flat[::step].sum()), float(flat[:min(100, len(flat))].mean()))


def detect_oai_layout(arr):
    """Detect whether a volume is C-order OAI ('oai_c'), F-order OAI ('oai_f'), or native."""
    if arr is None or arr.shape[:3] != (112, 140, 140):
        return 'native'
    sub = np.asarray(arr[:32, :64, :64], dtype=np.float32)
    if sub.ndim > 3:
        sub = sub[..., 0]
    mad0 = float(np.mean(np.abs(np.diff(sub, axis=0))))
    mad2 = float(np.mean(np.abs(np.diff(sub, axis=2))))
    if mad0 + mad2 < 1e-5:
        full = np.asarray(arr[..., 0] if arr.ndim > 3 else arr, dtype=np.float32)
        mad0 = float(np.mean(np.abs(np.diff(full, axis=0))))
        mad2 = float(np.mean(np.abs(np.diff(full, axis=2))))
    if mad0 / max(mad2, 1e-5) > 2.0:
        return 'oai_c'
    return 'oai_f'


def to_canonical_oai(arr, layout=None):
    """Transform an OAI volume or mask (3D or 4D) into canonical (Z, Y, X) orientation."""
    if arr is None:
        return None
    if layout is None:
        layout = detect_oai_layout(arr)
    if layout == 'native':
        return arr
    if arr.ndim == 4:
        chans = [to_canonical_oai(arr[..., c], layout=layout) for c in range(arr.shape[3])]
        return np.stack(chans, axis=-1)
    if layout == 'oai_c':
        arr_f = arr.ravel().reshape((112, 140, 140), order='F')
    else:
        arr_f = arr
    return np.ascontiguousarray(np.transpose(arr_f, (0, 2, 1))[:, ::-1, :])


def from_canonical_oai(arr_can, layout):
    """Transform a canonical (Z, Y, X) volume/mask/prob_map back to caller's layout."""
    if arr_can is None or layout == 'native':
        return arr_can
    if arr_can.ndim == 4:
        chans = [from_canonical_oai(arr_can[..., c], layout=layout) for c in range(arr_can.shape[3])]
        return np.stack(chans, axis=-1)
    arr_f = np.transpose(arr_can[:, ::-1, :], (0, 2, 1))
    if layout == 'oai_c':
        return np.ascontiguousarray(arr_f.ravel(order='F').reshape((112, 140, 140)))
    return np.ascontiguousarray(arr_f)


def _postprocess_cc(mask):
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
    best = int(np.argmax(counts))
    if counts[best] == 0:
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    out[z0:z1, y0:y1, x0:x1] = (labeled == best)
    return out


def fast_keep_top_k(mask, k=1, min_size=50):
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
        top_k = [int(np.argmax(counts))]
        if counts[top_k[0]] == 0:
            return np.zeros_like(mask, dtype=bool)
    else:
        top_k = valid[np.argsort(counts[valid])[::-1][:k]]
    out = np.zeros_like(mask, dtype=bool)
    out[z0:z1, y0:y1, x0:x1] = np.isin(labeled, top_k)
    return out


def postprocess_3bone_mask(mask, sp, c_j=None, span_z=65.0, bone_name='femur'):
    cc = _postprocess_cc(mask)
    if not np.any(cc):
        return cc
    inds = np.argwhere(cc)
    z0, y0, x0 = np.maximum(inds.min(axis=0) - 1, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + 2, cc.shape)
    crop = cc[z0:z1, y0:y1, x0:x1].copy()
    if bone_name in ('femur', 'tibia') and c_j is not None:
        is_femur = (bone_name == 'femur')
        for y_loc in range(crop.shape[1]):
            y_abs = y0 + y_loc
            jy = (y_abs * sp[1] - c_j[1]) / span_z
            if (is_femur and jy <= -0.16) or ((not is_femur) and jy >= 0.14):
                crop[:, y_loc, :] = binary_fill_holes(crop[:, y_loc, :])
    for z_loc in range(crop.shape[0]):
        crop[z_loc] = binary_fill_holes(crop[z_loc])
    if bone_name == 'patella':
        for y_loc in range(crop.shape[1]):
            crop[:, y_loc, :] = binary_fill_holes(crop[:, y_loc, :])
    crop = binary_fill_holes(crop)
    out = np.zeros_like(cc, dtype=bool)
    out[z0:z1, y0:y1, x0:x1] = crop
    return out


def _compute_shifted_diffs_3d(img_norm, z_idx, y_idx, x_idx, offsets, sp):
    Z, Y, X = img_norm.shape
    base = img_norm[z_idx, y_idx, x_idx]
    out = np.empty((len(z_idx), len(offsets)), dtype=np.float32)
    for k, (dz, dy, dx) in enumerate(offsets):
        sz = np.clip(z_idx + int(round(dz / sp[0])), 0, Z - 1)
        sy = np.clip(y_idx + int(round(dy / sp[1])), 0, Y - 1)
        sx = np.clip(x_idx + int(round(dx / sp[2])), 0, X - 1)
        out[:, k] = base - img_norm[sz, sy, sx]
    return out


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


def get_fov_bins(shape):
    Z, Y, X = shape
    bz1 = np.clip((np.arange(Z) / Z * 20).astype(int), 0, 19)[:, None, None]
    by1 = np.clip((np.arange(Y) / Y * 32).astype(int), 0, 31)[None, :, None]
    bx1 = np.clip((np.arange(X) / X * 32).astype(int), 0, 31)[None, None, :]
    return (bz1 * 1024 + by1 * 32 + bx1).ravel()


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


def get_joint_bins_3bone(shape, sp_s, c_j, span_z):
    Z, Y, X = shape
    flat1 = get_fov_bins(shape)
    jz = (np.arange(Z) * sp_s[0] - c_j[0]) / span_z
    jy = (np.arange(Y) * sp_s[1] - c_j[1]) / span_z
    jx = (np.arange(X) * sp_s[2] - c_j[2]) / span_z
    bz2 = np.clip(((jz + 0.70) / 1.40 * 24).astype(int), 0, 23)[:, None, None]
    by2 = np.clip(((jy + 1.20) / 2.40 * 40).astype(int), 0, 39)[None, :, None]
    bx2 = np.clip(((jx + 0.85) / 1.70 * 32).astype(int), 0, 31)[None, None, :]
    flat2 = (bz2 * 1280 + by2 * 32 + bx2).ravel()
    bzs = np.clip((np.abs(jz) / 0.70 * 16).astype(int), 0, 15)[:, None, None]
    flats = (bzs * 1280 + by2 * 32 + bx2).ravel()
    return flat1, flat2, flats


def extract_base_strided_features(img_s, sp_s, mask_s, atl_fov):
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
    pf_fov, pt_fov, pp_fov = atl_fov[0][f1], atl_fov[1][f1], atl_fov[2][f1]

    cont_cols = []
    g1_m = g1[z_idx, y_idx, x_idx]
    zl1_m = z_loc1_vol[z_idx, y_idx, x_idx]
    for v_vol, v_m in [(g1, g1_m), (z_loc1_vol, zl1_m)]:
        for d_mm in [18.0, 45.0]:
            mn_x, mp_x, mxn_x, mxp_x = [a[z_idx, y_idx, x_idx] for a in dir_min_max_1d(v_vol, 2, d_mm, sp_s[2])]
            mn_z, mp_z, mxn_z, mxp_z = [a[z_idx, y_idx, x_idx] for a in dir_min_max_1d(v_vol, 0, d_mm, sp_s[0])]
            mn_y, mp_y, mxn_y, mxp_y = [a[z_idx, y_idx, x_idx] for a in dir_min_max_1d(v_vol, 1, d_mm, sp_s[1])]
            encl_x = v_m - np.maximum(mn_x, mp_x)
            encl_z = v_m - np.maximum(mn_z, mp_z)
            encl_xz = v_m - 0.5 * (np.maximum(mn_x, mp_x) + np.maximum(mn_z, mp_z))
            cont_cols.extend([mn_x, mp_x, mxn_x, mxp_x, mn_z, mp_z, mxn_z, mxp_z, mn_y, mp_y, encl_x, encl_z, encl_xz])

    dx_steps = [4.0, 8.0, 14.0, 22.0, 32.0, 44.0]
    dz_steps = [4.0, 9.0, 16.0, 25.0, 36.0]
    dy_steps = [5.0, 12.0, 22.0, 35.0]
    rx_neg = np.column_stack([g2[z_idx, y_idx, np.clip(x_idx - int(round(d / sp_s[2])), 0, X - 1)] for d in dx_steps])
    rx_pos = np.column_stack([g2[z_idx, y_idx, np.clip(x_idx + int(round(d / sp_s[2])), 0, X - 1)] for d in dx_steps])
    rz_neg = np.column_stack([g2[np.clip(z_idx - int(round(d / sp_s[0])), 0, Z - 1), y_idx, x_idx] for d in dz_steps])
    rz_pos = np.column_stack([g2[np.clip(z_idx + int(round(d / sp_s[0])), 0, Z - 1), y_idx, x_idx] for d in dz_steps])
    ry_neg = np.column_stack([g2[z_idx, np.clip(y_idx - int(round(d / sp_s[1])), 0, Y - 1), x_idx] for d in dy_steps])
    ry_pos = np.column_stack([g2[z_idx, np.clip(y_idx + int(round(d / sp_s[1])), 0, Y - 1), x_idx] for d in dy_steps])

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
        pf_fov, pt_fov, pp_fov,
        np.log((pf_fov + 1e-4) / (pt_fov + 1e-4)),
        np.log((pp_fov + 1e-4) / (pf_fov + 1e-4)),
    ] + cont_cols

    rng = np.random.RandomState(42)
    offsets = np.vstack([rng.uniform(-14.0, 14.0, size=(10, 3)), rng.uniform(-36.0, 36.0, size=(10, 3))])
    rsid = _compute_shifted_diffs_3d(img_norm, z_idx, y_idx, x_idx, offsets, sp_s)

    return np.hstack([
        np.column_stack(cols),
        rx_neg, rx_pos, rz_neg, rz_pos, ry_neg, ry_pos,
        rsid,
    ]).astype(np.float32)


def augment_tierA_3bone_features(X_base, img_s_shape, sp_s, df_s, dt_s, dp_s, c_j, span_z, mask_s, atl_3b):
    inds = np.argwhere(mask_s)
    z_idx, y_idx, x_idx = inds[:, 0], inds[:, 1], inds[:, 2]
    jz = (z_idx * sp_s[0] - c_j[0]) / span_z
    jy = (y_idx * sp_s[1] - c_j[1]) / span_z
    jx = (x_idx * sp_s[2] - c_j[2]) / span_z
    bz2 = np.clip(((jz + 0.70) / 1.40 * 24).astype(int), 0, 23)
    by2 = np.clip(((jy + 1.20) / 2.40 * 40).astype(int), 0, 39)
    bx2 = np.clip(((jx + 0.85) / 1.70 * 32).astype(int), 0, 31)
    f2 = bz2 * 1280 + by2 * 32 + bx2
    bzs = np.clip((np.abs(jz) / 0.70 * 16).astype(int), 0, 15)
    fs = bzs * 1280 + by2 * 32 + bx2

    pf_fov, pt_fov, pp_fov = X_base[:, 23], X_base[:, 24], X_base[:, 25]
    pf_j, pt_j, pp_j = atl_3b[3][f2], atl_3b[4][f2], atl_3b[5][f2]
    pf_s, pt_s, pp_s = atl_3b[6][fs], atl_3b[7][fs], atl_3b[8][fs]
    p_fb = 0.45 * pf_j + 0.35 * pf_s + 0.20 * pf_fov
    p_tb = 0.45 * pt_j + 0.35 * pt_s + 0.20 * pt_fov
    p_pb = 0.50 * pp_j + 0.30 * pp_s + 0.20 * pp_fov
    log_ft = np.log((p_fb + 1e-4) / (p_tb + 1e-4))
    log_pf = np.log((p_pb + 1e-4) / (p_fb + 1e-4))

    df_feat = np.clip(df_s[z_idx, y_idx, x_idx], -8.0, 14.0)
    dt_feat = np.clip(dt_s[z_idx, y_idx, x_idx], -8.0, 14.0)
    dp_feat = np.clip(dp_s[z_idx, y_idx, x_idx], -8.0, 14.0)

    extra = np.column_stack([
        jz, jy, jx, np.abs(jz), np.sqrt(jy**2 + jx**2),
        pf_j, pt_j, pp_j, pf_s, pt_s, pp_s, p_fb, p_tb, p_pb, log_ft, log_pf,
        df_feat, dt_feat, dp_feat, df_feat - dt_feat, dp_feat - df_feat,
    ]).astype(np.float32)
    return np.hstack([X_base, extra])

def get_distal_femur_frame(fb, dt, sp):
    sp_arr = np.array(sp, dtype=np.float32)
    art = (fb > 0) & (dt <= 45.0)
    if np.sum(art) < 100:
        art = fb > 0
    if np.sum(art) < 10:
        Z, Y, X = fb.shape
        f_min = np.array([0.2 * Z * sp[0], 0.2 * Y * sp[1], 0.3 * X * sp[2]], dtype=np.float32)
        f_max = np.array([0.8 * Z * sp[0], 0.6 * Y * sp[1], 0.8 * X * sp[2]], dtype=np.float32)
        return f_min, f_max, np.maximum(f_max - f_min, 10.0)
    pts = np.argwhere(art).astype(np.float32) * sp_arr
    f_min = np.percentile(pts, 5, axis=0).astype(np.float32)
    f_max = np.percentile(pts, 95, axis=0).astype(np.float32)
    f_span = np.maximum(f_max - f_min, 10.0).astype(np.float32)
    return f_min, f_max, f_span


def get_patella_fem_bins(coords_mm, df_m, f_min, f_span):
    rel = (coords_mm - f_min) / f_span
    bz = np.clip(((rel[:, 0] + 0.25) / 1.30 * 24).astype(int), 0, 23)
    by = np.clip(((rel[:, 1] + 0.85) / 2.20 * 32).astype(int), 0, 31)
    bx = np.clip(((rel[:, 2] + 0.85) / 1.00 * 24).astype(int), 0, 23)
    bdf = np.clip((df_m / 34.0 * 20).astype(int), 0, 19)
    return (bz, by, bx), (bz, by, bdf), rel


def extract_patella_specialist_features(img, sp, fb, tb, pb_init, pat_atl, df=None, dt=None, dp_0=None, mask=None):
    if df is None:
        df = fast_signed_distance_transform(fb > 0, sp)
    if dt is None:
        dt = fast_signed_distance_transform(tb > 0, sp)
    if dp_0 is None:
        dp_0 = fast_signed_distance_transform(pb_init > 0, sp) if np.any(pb_init > 0) else np.full_like(df, 20.0)

    f_min, _, f_span = get_distal_femur_frame(fb, dt, sp)
    if mask is None:
        cand = (df >= -0.5) & (df <= 35.0)
        inds_c = np.argwhere(cand)
        if len(inds_c) == 0:
            return np.zeros(img.shape, dtype=bool), np.zeros((0, 95), dtype=np.float32)
        coords_c = inds_c.astype(np.float32) * np.array(sp, dtype=np.float32)
        rel_c = (coords_c - f_min) / f_span
        in_box = (
            (rel_c[:, 0] >= -0.25) & (rel_c[:, 0] <= 1.05)
            & (rel_c[:, 1] >= -0.85) & (rel_c[:, 1] <= 1.35)
            & (rel_c[:, 2] >= -0.85) & (rel_c[:, 2] <= 0.15)
        )
        dp_c = dp_0[inds_c[:, 0], inds_c[:, 1], inds_c[:, 2]]
        keep = in_box | (dp_c <= 10.0)
        mask = np.zeros(img.shape, dtype=bool)
        kept = inds_c[keep]
        mask[kept[:, 0], kept[:, 1], kept[:, 2]] = True

    inds = np.argwhere(mask)
    if len(inds) == 0:
        return mask, np.zeros((0, 95), dtype=np.float32)

    z_idx, y_idx, x_idx = inds[:, 0], inds[:, 1], inds[:, 2]
    coords_mm = inds.astype(np.float32) * np.array(sp, dtype=np.float32)
    df_m = df[z_idx, y_idx, x_idx]
    dt_m = dt[z_idx, y_idx, x_idx]
    dp_m = dp_0[z_idx, y_idx, x_idx]

    b1, b2, rel_f = get_patella_fem_bins(coords_mm, np.maximum(df_m, 0.0), f_min, f_span)
    p_atl1 = pat_atl[0][b1]
    p_atl2 = pat_atl[1][b2]
    p_atl_ens = 0.55 * p_atl1 + 0.45 * p_atl2

    margin = [int(np.ceil(26.0 / s)) for s in sp]
    z0, y0, x0 = np.maximum(inds.min(axis=0) - margin, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + margin + 1, img.shape)
    zc, yc, xc = z_idx - z0, y_idx - y0, x_idx - x0
    cz_max, cy_max, cx_max = z1 - z0, y1 - y0, x1 - x0

    p5, p50, p95 = np.percentile(img, [5, 50, 95])
    scale = max(float(p95 - p5), 1.0)
    img_c = ((img[z0:z1, y0:y1, x0:x1].astype(np.float32) - p50) / scale).astype(np.float32)
    df_c = df[z0:z1, y0:y1, x0:x1]

    dfs_c = gaussian_filter(df_c, sigma=[2.0 / s for s in sp])
    nz_f, ny_f, nx_f = np.gradient(dfs_c, *sp)
    n_norm = np.sqrt(nz_f**2 + ny_f**2 + nx_f**2) + 1e-6
    nz_m = (nz_f[zc, yc, xc] / n_norm[zc, yc, xc]).astype(np.float32)
    ny_m = (ny_f[zc, yc, xc] / n_norm[zc, yc, xc]).astype(np.float32)
    nx_m = (nx_f[zc, yc, xc] / n_norm[zc, yc, xc]).astype(np.float32)

    g05 = gaussian_filter(img_c, sigma=[0.5 / s for s in sp])
    g1 = gaussian_filter(img_c, sigma=[1.0 / s for s in sp])
    g2 = gaussian_filter(img_c, sigma=[2.0 / s for s in sp])
    g4 = gaussian_filter(img_c, sigma=[4.0 / s for s in sp])
    g8 = gaussian_filter(img_c, sigma=[8.0 / s for s in sp])
    g16 = gaussian_filter(img_c, sigma=[16.0 / s for s in sp])
    var16 = gaussian_filter((img_c - g16) ** 2, sigma=[16.0 / s for s in sp])
    std16 = np.sqrt(np.maximum(var16, 0.0) + 0.015)
    z_loc0 = (g05 - g16) / std16
    z_loc1 = (g1 - g16) / std16

    gz1, gy1, gx1 = np.gradient(g1, *sp)
    gmag1 = np.sqrt(gz1**2 + gy1**2 + gx1**2)
    gz2, gy2, gx2 = np.gradient(g2, *sp)
    gmag2 = np.sqrt(gz2**2 + gy2**2 + gx2**2)

    encl_cols = []
    v0_m = g05[zc, yc, xc]
    v1_m = g1[zc, yc, xc]
    zl1_m = z_loc1[zc, yc, xc]
    for v_vol, v_m in [(g1, v1_m), (z_loc1, zl1_m)]:
        for d_mm in [12.0, 24.0]:
            mn_x, mp_x, mxn_x, mxp_x = [a[zc, yc, xc] for a in dir_min_max_1d(v_vol, 2, d_mm, sp[2])]
            mn_y, mp_y, mxn_y, mxp_y = [a[zc, yc, xc] for a in dir_min_max_1d(v_vol, 1, d_mm, sp[1])]
            mn_z, mp_z, mxn_z, mxp_z = [a[zc, yc, xc] for a in dir_min_max_1d(v_vol, 0, d_mm, sp[0])]
            encl_cols.extend([
                mn_x, mp_x, mxn_x, mxp_x,
                mn_y, mp_y, mxn_y, mxp_y,
                mn_z, mp_z, mxn_z, mxp_z,
                v_m - np.maximum(mn_x, mp_x),
                v_m - np.maximum(mn_z, mp_z),
                mxp_x - v_m,
            ])

    x_steps = [-8.0, -5.0, -2.5, 2.0, 4.0, 6.0, 8.5, 11.5, 15.0, 19.0]
    rx_v = np.column_stack([
        g1[zc, yc, np.clip(xc + int(round(d / sp[2])), 0, cx_max - 1)] for d in x_steps
    ])
    N = len(zc)
    fracs = [0.20, 0.40, 0.60, 0.78, 0.92, 1.05, 1.25]
    r_to_fem = np.empty((N, len(fracs)), dtype=np.float32)
    d_clamp = np.clip(df_m, 1.0, 32.0)
    for k, fr in enumerate(fracs):
        step_mm = fr * d_clamp
        sz = np.clip(np.round(zc - step_mm * nz_m / sp[0]).astype(int), 0, cz_max - 1)
        sy = np.clip(np.round(yc - step_mm * ny_m / sp[1]).astype(int), 0, cy_max - 1)
        sx = np.clip(np.round(xc - step_mm * nx_m / sp[2]).astype(int), 0, cx_max - 1)
        r_to_fem[:, k] = g05[sz, sy, sx]

    max_cart_between = np.max(r_to_fem[:, :5], axis=1)
    fem_interior_val = r_to_fem[:, 6]
    sandwich_contrast = max_cart_between - 0.5 * (v0_m + fem_interior_val)
    dp_feat = np.clip(dp_m, -7.0, 14.0)

    X_pat = np.hstack([
        np.column_stack([
            v0_m, v1_m, g2[zc, yc, xc], g4[zc, yc, xc], g8[zc, yc, xc], g16[zc, yc, xc],
            v0_m - g2[zc, yc, xc], v1_m - g4[zc, yc, xc], g2[zc, yc, xc] - g8[zc, yc, xc],
            z_loc0[zc, yc, xc], zl1_m,
            gmag1[zc, yc, xc], gmag2[zc, yc, xc], gx1[zc, yc, xc], gy1[zc, yc, xc],
            rel_f[:, 0], rel_f[:, 1], rel_f[:, 2], np.abs(rel_f[:, 0] - 0.45),
            df_m, dt_m, dp_feat, df_m - dp_feat,
            p_atl1, p_atl2, p_atl_ens,
            nz_m, ny_m, nx_m,
            max_cart_between, max_cart_between - v0_m, sandwich_contrast,
        ]),
        np.column_stack(encl_cols),
        rx_v,
        r_to_fem,
    ]).astype(np.float32)
    return mask, X_pat


def detect_medial_is_low_z_3bone(bm, sp):
    fb = bm['femur'] > 0
    tb = bm['tibia'] > 0
    if np.sum(fb) < 20 or np.sum(tb) < 20:
        return False
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
    tb_dx = ((np.percentile(xt1, 95) - np.percentile(xt1, 5)) - (np.percentile(xt2, 95) - np.percentile(xt2, 5))) * sp[2]
    return bool((f_valg + 0.5 * tb_dx) >= 0.0)


def get_robust_bone_frame_3bone(bm_self, d_other, max_other, sp, med_low_z=False):
    sp_arr = np.array(sp, dtype=np.float32)
    art = (bm_self > 0) & (d_other <= max_other)
    if np.sum(art) < 80:
        art = bm_self > 0
    if np.sum(art) < 10:
        Z, Y, X = bm_self.shape
        q50 = np.array([0.5 * Z * sp[0], 0.5 * Y * sp[1], 0.5 * X * sp[2]], dtype=np.float32)
        return q50, np.array([40.0, 40.0, 40.0], dtype=np.float32), med_low_z
    pts = np.argwhere(art).astype(np.float32) * sp_arr
    q10 = np.percentile(pts, 10, axis=0)
    q50 = np.percentile(pts, 50, axis=0)
    q90 = np.percentile(pts, 90, axis=0)
    scale = np.maximum(q90 - q10, 5.0)
    return q50.astype(np.float32), scale.astype(np.float32), med_low_z


def coords_to_bone_atlas_bins_3bone(coords_mm, q50, scale, med_low_z=False):
    rel = (coords_mm - q50) / scale
    uz_sym = np.abs(rel[:, 0])
    uz_ml = rel[:, 0] if med_low_z else -rel[:, 0]
    uy = rel[:, 1]
    ux = rel[:, 2]
    bz_sym = np.clip((uz_sym / 1.10 * 24).astype(int), 0, 23)
    bz_ml = np.clip(((uz_ml + 1.10) / 2.20 * 28).astype(int), 0, 27)
    by = np.clip(((uy + 1.20) / 2.40 * 32).astype(int), 0, 31)
    bx = np.clip(((ux + 1.10) / 2.20 * 24).astype(int), 0, 23)
    return (bz_sym, by, bx), (bz_ml, by, bx)


def _get_bone_d_self_other(bm_pr, bone_name, sp, dts_pre=None):
    if dts_pre is not None:
        df, dt, dp = dts_pre
    else:
        df = fast_signed_distance_transform(bm_pr['femur'] > 0, sp)
        dt = fast_signed_distance_transform(bm_pr['tibia'] > 0, sp)
        dp = (
            fast_signed_distance_transform(bm_pr['patella'] > 0, sp)
            if ('patella' in bm_pr and np.any(bm_pr['patella'] > 0))
            else np.full_like(df, 50.0)
        )
    if bone_name == 'femur':
        return df, np.minimum(dt, dp), dt, dp, 45.0
    elif bone_name == 'tibia':
        return dt, df, dt, dp, 25.0
    else:
        return dp, df, dt, dp, 25.0


def extract_ray_feats_3bone_v2(img, sp, bm_pr, bone_name, bone_atlas_p, dts_pre=None, return_bins=False):
    from .rf_seg import _compute_articular_landmarks_and_stats
    d_self, d_other, dt_raw, dp_raw, max_other = _get_bone_d_self_other(bm_pr, bone_name, sp, dts_pre)

    d_norm_ref = gaussian_filter(d_self, sigma=[2.2 / s for s in sp])
    nz, ny, nx = np.gradient(d_norm_ref, *sp)
    norm = np.sqrt(nz**2 + ny**2 + nx**2) + 1e-6
    nz, ny, nx = (nz / norm).astype(np.float32), (ny / norm).astype(np.float32), (nx / norm).astype(np.float32)

    shell = np.abs(d_self) <= 3.0
    if not np.any(shell):
        empty_x1 = np.zeros((0, 108), dtype=np.float32)
        empty_x2 = np.zeros((0, 158), dtype=np.float32)
        if return_bins:
            return shell, d_self, d_other, empty_x1, empty_x2, None, None
        return shell, d_self, d_other, empty_x1, empty_x2
    inds = np.argwhere(shell)
    z_idx, y_idx, x_idx = inds[:, 0], inds[:, 1], inds[:, 2]
    nz_m, ny_m, nx_m = nz[z_idx, y_idx, x_idx], ny[z_idx, y_idx, x_idx], nx[z_idx, y_idx, x_idx]

    img_f32 = img.astype(np.float32)
    img_f = gaussian_filter(img_f32, sigma=[0.4 / s for s in sp])
    img_lat = gaussian_filter(img_f32, sigma=[1.2 / sp[0], 0.45 / sp[1], 1.2 / sp[2]])
    img_s1 = gaussian_filter(img_f32, sigma=[1.0 / s for s in sp])
    img_s2 = gaussian_filter(img_f32, sigma=[2.0 / s for s in sp])

    z0, y0, x0 = np.maximum(inds.min(axis=0) - 6, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + 7, img.shape)
    ds_c = d_self[z0:z1, y0:y1, x0:x1]
    im_c = img_lat[z0:z1, y0:y1, x0:x1]
    sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
    w_bn = ((ds_c <= -1.2) & (ds_c >= -5.5)).astype(np.float32) + 1e-5
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

    med_low_z = detect_medial_is_low_z_3bone(bm_pr, sp)
    q50, scale, _ = get_robust_bone_frame_3bone(bm_pr[bone_name], d_other, max_other, sp, med_low_z)
    bins_s, bins_m = coords_to_bone_atlas_bins_3bone(coords_mm, q50, scale, med_low_z)
    p_atl_s = bone_atlas_p[0][bins_s]
    p_atl_m = bone_atlas_p[1][bins_m]
    rel_f = (coords_mm - q50) / scale
    uz_ml = rel_f[:, 0] if med_low_z else -rel_f[:, 0]

    d_self_m = d_self[z_idx, y_idx, x_idx]
    d_other_m = d_other[z_idx, y_idx, x_idx]
    dt_m = dt_raw[z_idx, y_idx, x_idx]
    dp_m = dp_raw[z_idx, y_idx, x_idx]

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
            p_atl_m, uz_ml, d_self_m, d_other_m, dt_m, dp_m,
            idx_peak_rise, val_peak_rise, idx_cort_min, val_cort_min,
            z_bone_crop[zc, yc, xc],
        ]),
        ray_zb[:, 3:16], vy_lat, vyg_lat, vy_zb,
    ]).astype(np.float32)
    if return_bins:
        return shell, d_self, d_other, X_v1, X_v2, bins_s, bins_m
    return shell, d_self, d_other, X_v1, X_v2


def predict_p0_case(img_s, sp_s, full_shape, sp_full, p0_models_list, atl_fov):
    f1 = get_fov_bins(img_s.shape)
    p_fov_sum = (atl_fov[0][f1] + atl_fov[1][f1] + atl_fov[2][f1]).reshape(img_s.shape)
    cand0 = p_fov_sum >= 0.002
    if not np.any(cand0):
        cand0 = np.ones(img_s.shape, dtype=bool)
    X0 = extract_base_strided_features(img_s, sp_s, cand0, atl_fov)
    max_n_jobs = min(8, os.cpu_count() or 4)
    for m in p0_models_list:
        if hasattr(m, 'n_jobs') and getattr(m, 'n_jobs', 1) not in range(1, max_n_jobs + 1):
            m.n_jobs = max_n_jobs
    with joblib.parallel_backend('threading'):
        p0 = sum(m.predict_proba(X0) for m in p0_models_list) / float(len(p0_models_list))
    v0 = np.zeros((*img_s.shape, 4), dtype=np.float32)
    v0[..., 0] = 1.0
    v0[cand0] = p0
    for c in range(4):
        v0[..., c] = gaussian_filter(v0[..., c], sigma=[1.0 / s for s in sp_s])

    fb_s = _postprocess_cc((v0[..., 1] >= 0.42) & (v0[..., 1] >= v0[..., 2]) & (v0[..., 1] >= v0[..., 3]))
    tb_s = _postprocess_cc((v0[..., 2] >= 0.42) & (v0[..., 2] > v0[..., 1]) & (v0[..., 2] >= v0[..., 3]))
    pb_s = _postprocess_cc((v0[..., 3] >= 0.35) & (v0[..., 3] > v0[..., 1]) & (v0[..., 3] > v0[..., 2]))
    if not np.any(pb_s):
        pb_s = _postprocess_cc(v0[..., 3] >= 0.22)
    df_s, dt_s, c_j, span_z = compute_joint_frame_strided(fb_s, tb_s, sp_s)
    dp_s = fast_signed_distance_transform(pb_s, sp_s) if np.any(pb_s) else np.full_like(df_s, 25.0)

    zf = [full_shape[d] / img_s.shape[d] for d in range(3)]
    pf_up = zoom(v0[..., 1], zf, order=1)
    pt_up = zoom(v0[..., 2], zf, order=1)
    pp_up = zoom(v0[..., 3], zf, order=1)
    bm_0 = {
        'femur': postprocess_3bone_mask((pf_up >= 0.45) & (pf_up >= pt_up) & (pf_up >= pp_up), sp_full, c_j, span_z, 'femur').astype(np.uint8),
        'tibia': postprocess_3bone_mask((pt_up >= 0.45) & (pt_up > pf_up) & (pt_up >= pp_up), sp_full, c_j, span_z, 'tibia').astype(np.uint8),
        'patella': postprocess_3bone_mask((pp_up >= 0.38) & (pp_up > pf_up) & (pp_up > pt_up), sp_full, c_j, span_z, 'patella').astype(np.uint8),
    }
    return bm_0, df_s, dt_s, dp_s, c_j, span_z


def predict_tierA_3bone_case(img, sp, img_s, sp_s, df_s, dt_s, dp_s, c_j, span_z, tierA_models_list, atl_3b, pat_models_list=None, pat_atl=None):
    _, f2, fs = get_joint_bins_3bone(img_s.shape, sp_s, c_j, span_z)
    p_atl = np.maximum(
        atl_3b[3][f2] + atl_3b[4][f2] + atl_3b[5][f2],
        atl_3b[6][fs] + atl_3b[7][fs] + atl_3b[8][fs],
    ).reshape(img_s.shape)
    cand = (df_s <= 10.0) | (dt_s <= 10.0) | (dp_s <= 10.0) | (p_atl >= 0.015)
    if not np.any(cand):
        cand = np.ones(img_s.shape, dtype=bool)
    X_base = extract_base_strided_features(img_s, sp_s, cand, atl_3b[:3])
    X_c = augment_tierA_3bone_features(X_base, img_s.shape, sp_s, df_s, dt_s, dp_s, c_j, span_z, cand, atl_3b)
    max_n_jobs = min(8, os.cpu_count() or 4)
    for m in tierA_models_list:
        if hasattr(m, 'n_jobs') and getattr(m, 'n_jobs', 1) not in range(1, max_n_jobs + 1):
            m.n_jobs = max_n_jobs
    with joblib.parallel_backend('threading'):
        p_pred = sum(m.predict_proba(X_c) for m in tierA_models_list) / float(len(tierA_models_list))

    pf_raw_s = 1.0 / (1.0 + np.exp(np.clip(df_s[cand] / 1.5, -15.0, 15.0)))
    pt_raw_s = 1.0 / (1.0 + np.exp(np.clip(dt_s[cand] / 1.5, -15.0, 15.0)))
    pp_raw_s = 1.0 / (1.0 + np.exp(np.clip(dp_s[cand] / 1.5, -15.0, 15.0)))
    p0_raw_s = np.clip(1.0 - pf_raw_s - pt_raw_s - pp_raw_s, 0.0, 1.0)
    pr_raw_s = np.column_stack([p0_raw_s, pf_raw_s, pt_raw_s, pp_raw_s])

    disagree = np.max(np.abs(p_pred - pr_raw_s), axis=1, keepdims=True)
    w_raw = 0.12 * np.clip(1.0 - disagree / 0.45, 0.0, 1.0)
    pb_s = (1.0 - w_raw) * p_pred + w_raw * pr_raw_s

    vol_s = np.zeros((*img_s.shape, 4), dtype=np.float32)
    vol_s[..., 0] = 1.0
    vol_s[cand] = pb_s
    for c in range(4):
        vol_s[..., c] = gaussian_filter(vol_s[..., c], sigma=[1.0 / s for s in sp_s])

    zf = [img.shape[d] / img_s.shape[d] for d in range(3)]
    pf_up = zoom(vol_s[..., 1], zf, order=1).astype(np.float32)
    pt_up = zoom(vol_s[..., 2], zf, order=1).astype(np.float32)
    pp_up = zoom(vol_s[..., 3], zf, order=1).astype(np.float32)

    fb_mask = postprocess_3bone_mask((pf_up >= 0.47) & (pf_up >= pt_up) & (pf_up >= pp_up), sp, c_j, span_z, 'femur').astype(np.uint8)
    tb_mask = postprocess_3bone_mask((pt_up >= 0.47) & (pt_up > pf_up) & (pt_up >= pp_up), sp, c_j, span_z, 'tibia').astype(np.uint8)
    pb_mask = postprocess_3bone_mask((pp_up >= 0.44) & (pp_up > pf_up) & (pp_up > pt_up), sp, c_j, span_z, 'patella').astype(np.uint8)
    if not np.any(pb_mask):
        pb_mask = postprocess_3bone_mask((pp_up >= 0.28) & (pp_up > pf_up), sp, c_j, span_z, 'patella').astype(np.uint8)

    if pat_models_list is not None and pat_atl is not None:
        df_full = fast_signed_distance_transform(fb_mask > 0, sp)
        dt_full = fast_signed_distance_transform(tb_mask > 0, sp)
        dp_init = fast_signed_distance_transform(pb_mask > 0, sp) if np.any(pb_mask > 0) else np.full_like(df_full, 20.0)
        cand_pat, X_pat = extract_patella_specialist_features(
            img, sp, fb_mask, tb_mask, pb_mask, pat_atl, df=df_full, dt=dt_full, dp_0=dp_init, mask=None
        )
        if np.any(cand_pat):
            for m in pat_models_list:
                if hasattr(m, 'n_jobs') and getattr(m, 'n_jobs', 1) not in range(1, max_n_jobs + 1):
                    m.n_jobs = max_n_jobs
            with joblib.parallel_backend('threading'):
                p_pat_spec = sum(m.predict_proba(X_pat)[:, 1] for m in pat_models_list) / float(len(pat_models_list))
            vol_pat = 0.20 * pp_up
            vol_pat[cand_pat] = 0.78 * p_pat_spec + 0.22 * pp_up[cand_pat]
            vol_pat = gaussian_filter(vol_pat, sigma=[0.85 / s for s in sp])
            pb_spec = postprocess_3bone_mask((vol_pat >= 0.46) & (fb_mask == 0) & (tb_mask == 0), sp, c_j, span_z, 'patella').astype(np.uint8)
            if np.sum(pb_spec) >= 300:
                pb_mask = pb_spec
                pp_up = vol_pat.astype(np.float32)

    return {'femur': fb_mask, 'tibia': tb_mask, 'patella': pb_mask}, {'femur': pf_up, 'tibia': pt_up, 'patella': pp_up, 'c_j': c_j, 'span_z': span_z}


def refine_one_bone_3bone_v2(
    img, sp, bm_a, bone_name, m_d, dts_pre=None, return_v1=False,
    joint_atl=None, joint_clf=None, alpha_jb=0.0, return_prob=False,
):
    if np.sum(bm_a['femur'] > 0) < 20 or np.sum(bm_a['tibia'] > 0) < 20 or np.sum(bm_a.get(bone_name, 0) > 0) < 20:
        m0 = (bm_a.get(bone_name, np.zeros_like(img, dtype=np.uint8)) > 0).astype(np.uint8)
        p0 = m0.astype(np.float32)
        if return_v1:
            return (m0, m0, p0) if return_prob else (m0, m0)
        return (m0, p0) if return_prob else m0
    shell, d_self, d_other, X1, X2, bins_s, _ = extract_ray_feats_3bone_v2(
        img, sp, bm_a, bone_name, m_d['p_full'], dts_pre=dts_pre, return_bins=True
    )
    if not np.any(shell):
        m0 = (bm_a[bone_name] > 0).astype(np.uint8)
        p0 = m0.astype(np.float32)
        if return_v1:
            return (m0, m0, p0) if return_prob else (m0, m0)
        return (m0, p0) if return_prob else m0

    max_et_jobs = min(max(os.cpu_count() or 4, 1), 16)
    for et_key in ('et_v1', 'et_whole', 'et_sub'):
        if et_key in m_d and getattr(m_d[et_key], 'n_jobs', 1) not in range(1, max_et_jobs + 1):
            m_d[et_key].n_jobs = max_et_jobs

    with joblib.parallel_backend('threading'):
        p1 = m_d['et_v1'].predict_proba(X1)[:, 1].astype(np.float32)
        p_whole = (
            0.40 * m_d['hgb_whole_1'].predict_proba(X2)[:, 1]
            + 0.35 * m_d['hgb_whole_2'].predict_proba(X2)[:, 1]
            + 0.25 * m_d['et_whole'].predict_proba(X2)[:, 1]
        ).astype(np.float32)
        p_sub = (
            0.40 * m_d['hgb_sub_1'].predict_proba(X2)[:, 1]
            + 0.35 * m_d['hgb_sub_2'].predict_proba(X2)[:, 1]
            + 0.25 * m_d['et_sub'].predict_proba(X2)[:, 1]
        ).astype(np.float32)

    max_other = 45.0 if bone_name == 'femur' else 25.0
    d_o_sh = d_other[shell]
    w_art = np.clip((max_other + 4.0 - d_o_sh) / 8.0, 0.0, 1.0) * (d_o_sh >= -3.0)
    p2_art = 0.30 * p1 + 0.50 * p_sub + 0.20 * p_whole
    p2_non = 0.30 * p1 + 0.70 * p_whole
    p_bl = w_art * p2_art + (1.0 - w_art) * p2_non

    if joint_clf is not None and alpha_jb > 0.0:
        X1_j = X1.copy()
        if joint_atl is not None and bins_s is not None:
            X1_j[:, 6] = joint_atl[0][bins_s]
        p_jb = joint_clf.predict_proba(X1_j)[:, 1].astype(np.float32)
        if bone_name == 'femur':
            dp_m = X2[:, X1.shape[1] + 5]
            w_jb = float(alpha_jb) * np.clip((dp_m - 22.0) / 4.0, 0.0, 1.0)
        else:
            w_jb = float(alpha_jb)
        p_bl = (1.0 - w_jb) * p_bl + w_jb * p_jb

    p_orig = 1.0 / (1.0 + np.exp(np.clip(d_self / 0.45, -20.0, 20.0)))
    sig_s = [0.75 / sp[0], 0.55 / sp[1], 0.65 / sp[2]]
    volb = p_orig.copy()
    volb[shell] = p_bl
    volb_s = gaussian_filter(volb, sigma=sig_s).astype(np.float32)
    m_v2 = postprocess_3bone_mask(volb_s >= 0.50, sp, bone_name=bone_name).astype(np.uint8)
    if not return_v1:
        return (m_v2, volb_s) if return_prob else m_v2
    volb1 = p_orig.copy()
    volb1[shell] = p1
    m_v1 = postprocess_3bone_mask(gaussian_filter(volb1, sigma=sig_s) >= 0.50, sp, bone_name=bone_name).astype(np.uint8)
    return (m_v2, m_v1, volb_s) if return_prob else (m_v2, m_v1)


def _pack_3bone_pred_prob(bm_dict, probs_dict, img_shape):
    pred_mask = np.zeros(img_shape, dtype=np.uint8)
    pred_mask[bm_dict['femur'] > 0] = 1
    pred_mask[bm_dict['tibia'] > 0] = 2
    pred_mask[bm_dict['patella'] > 0] = 3

    prob_vol = np.zeros((*img_shape, 4), dtype=np.float32)
    prob_vol[..., 1] = np.clip(probs_dict['femur'], 0.0, 1.0)
    prob_vol[..., 2] = np.clip(probs_dict['tibia'], 0.0, 1.0)
    prob_vol[..., 3] = np.clip(probs_dict['patella'], 0.0, 1.0)
    prob_vol[pred_mask == 1, 1] = np.maximum(prob_vol[pred_mask == 1, 1], 0.75)
    prob_vol[pred_mask == 2, 2] = np.maximum(prob_vol[pred_mask == 2, 2], 0.75)
    prob_vol[pred_mask == 3, 3] = np.maximum(prob_vol[pred_mask == 3, 3], 0.75)
    prob_vol[pred_mask != 1, 1] = np.minimum(prob_vol[pred_mask != 1, 1], 0.40)
    prob_vol[pred_mask != 2, 2] = np.minimum(prob_vol[pred_mask != 2, 2], 0.40)
    prob_vol[pred_mask != 3, 3] = np.minimum(prob_vol[pred_mask != 3, 3], 0.35)
    s_fg = prob_vol[..., 1] + prob_vol[..., 2] + prob_vol[..., 3]
    over = s_fg > 1.0
    if np.any(over):
        prob_vol[over, 1] /= s_fg[over]
        prob_vol[over, 2] /= s_fg[over]
        prob_vol[over, 3] /= s_fg[over]
    prob_vol[..., 0] = np.clip(1.0 - prob_vol[..., 1] - prob_vol[..., 2] - prob_vol[..., 3], 0.0, 1.0)
    return pred_mask, prob_vol


def predict_oai_bone_p1(img, spacing, model_dict):
    """Run Pass-1 (Pass-0 + Tier-A + Patella Specialist) 3-bone segmentation on an OAI volume."""
    layout = detect_oai_layout(img)
    img_c = to_canonical_oai(img, layout=layout).astype(np.float32)
    sp_c = (1.0, 1.0, 1.0) if layout != 'native' else tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))
    stride = model_dict.get('stride', (1, 2, 2))
    img_s = img_c[::stride[0], ::stride[1], ::stride[2]].astype(np.float32)
    sp_s = (sp_c[0] * stride[0], sp_c[1] * stride[1], sp_c[2] * stride[2])

    _, df_s, dt_s, dp_s, c_j, span_z = predict_p0_case(
        img_s, sp_s, img_c.shape, sp_c, model_dict['p0_models'], model_dict['atl_0_full'][:3]
    )
    atl_a = model_dict.get('atl_A_full') or model_dict['atl_a_full']
    bm_a, probs_a = predict_tierA_3bone_case(
        img_c, sp_c, img_s, sp_s, df_s, dt_s, dp_s, c_j, span_z,
        model_dict['tierA_models'], atl_a,
        pat_models_list=model_dict.get('pat_models'), pat_atl=model_dict.get('pat_atl_full')
    )
    pred_c, prob_c = _pack_3bone_pred_prob(bm_a, probs_a, img_c.shape)
    prob_out = from_canonical_oai(prob_c, layout)
    fp_c = _vol_fingerprint_oai(img_c)
    while len(_OAI_BONE_AUX_CACHE) >= 16:
        _OAI_BONE_AUX_CACHE.pop(next(iter(_OAI_BONE_AUX_CACHE)), None)
    _OAI_BONE_AUX_CACHE[fp_c] = {
        'bm_raw': bm_a,
        'prob_p1_fp': _vol_fingerprint_oai(prob_out),
    }
    return from_canonical_oai(pred_c, layout), prob_out


def predict_oai_bone_p2(
    img, prob_map, spacing, ref_v2_dict, p1_dict=None,
    joint_bone_atlases=None, joint_bone_models=None,
):
    """Run Pass-2 Tier-B Subchondral Surface Refiner for all 3 bones on an OAI volume."""
    layout = detect_oai_layout(img)
    img_c = to_canonical_oai(img, layout=layout).astype(np.float32)
    sp_c = (1.0, 1.0, 1.0) if layout != 'native' else tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))
    fp_c = _vol_fingerprint_oai(img_c)

    cached_aux = _OAI_BONE_AUX_CACHE.get(fp_c)
    has_valid_prob_map = (
        prob_map is not None
        and prob_map.ndim == 4
        and prob_map.shape[3] >= 4
        and np.any(prob_map[..., 1:4] > 0.2)
    )
    prob_matches_cache = (
        isinstance(cached_aux, dict)
        and 'bm_raw' in cached_aux
        and (prob_map is None or _vol_fingerprint_oai(prob_map) == cached_aux.get('prob_p1_fp'))
    )

    bm_a = None
    if prob_matches_cache:
        bm_a = cached_aux['bm_raw']
    elif has_valid_prob_map:
        pm_c = to_canonical_oai(prob_map, layout=layout)
        pf, pt, pp = pm_c[..., 1], pm_c[..., 2], pm_c[..., 3]
        bm_a = {
            'femur': postprocess_3bone_mask((pf >= 0.45) & (pf >= pt) & (pf >= pp), sp_c, bone_name='femur').astype(np.uint8),
            'tibia': postprocess_3bone_mask((pt >= 0.45) & (pt > pf) & (pt >= pp), sp_c, bone_name='tibia').astype(np.uint8),
            'patella': postprocess_3bone_mask((pp >= 0.38) & (pp > pf) & (pp > pt), sp_c, bone_name='patella').astype(np.uint8),
        }
    elif isinstance(cached_aux, dict) and 'bm_raw' in cached_aux:
        bm_a = cached_aux['bm_raw']
    elif p1_dict is not None:
        predict_oai_bone_p1(img, spacing, p1_dict)
        bm_a = _OAI_BONE_AUX_CACHE[fp_c]['bm_raw']
    else:
        raise ValueError("predict_oai_bone_p2 requires Pass-1 probability map or p1_dict.")

    dts_a = (
        fast_signed_distance_transform(bm_a['femur'] > 0, sp_c),
        fast_signed_distance_transform(bm_a['tibia'] > 0, sp_c),
        fast_signed_distance_transform(bm_a['patella'] > 0, sp_c) if np.any(bm_a['patella'] > 0) else np.full(img_c.shape, 50.0, dtype=np.float32),
    )
    bm_v2 = {}
    bm_v1 = {}
    probs_v2 = {}
    jb_alphas = {'femur': 0.06, 'tibia': 0.15, 'patella': 0.0}
    for b_name in ('femur', 'tibia', 'patella'):
        j_atl = joint_bone_atlases.get(b_name) if isinstance(joint_bone_atlases, dict) else None
        j_clf = joint_bone_models.get(b_name) if isinstance(joint_bone_models, dict) else None
        m2, m1, p2 = refine_one_bone_3bone_v2(
            img_c, sp_c, bm_a, b_name, ref_v2_dict[b_name], dts_pre=dts_a, return_v1=True,
            joint_atl=j_atl, joint_clf=j_clf, alpha_jb=jb_alphas[b_name], return_prob=True,
        )
        bm_v2[b_name] = m2
        bm_v1[b_name] = m1
        probs_v2[b_name] = p2

    prob_p1_fp = (
        cached_aux.get('prob_p1_fp')
        if isinstance(cached_aux, dict) and prob_matches_cache
        else (_vol_fingerprint_oai(prob_map) if has_valid_prob_map else None)
    )
    bm_fp = tuple(_vol_fingerprint_oai(bm_v2[k]) for k in ('femur', 'tibia', 'patella'))
    while len(_OAI_BONE_AUX_CACHE) >= 16:
        _OAI_BONE_AUX_CACHE.pop(next(iter(_OAI_BONE_AUX_CACHE)), None)
    _OAI_BONE_AUX_CACHE[fp_c] = {
        'bm_fp': bm_fp,
        'bm_raw': bm_a,
        'bm_v1': bm_v1,
        'bm_v2': bm_v2,
        'prob_p1_fp': prob_p1_fp,
    }
    pred_c, prob_c = _pack_3bone_pred_prob(bm_v2, probs_v2, img_c.shape)
    return from_canonical_oai(pred_c, layout), from_canonical_oai(prob_c, layout)


def compute_ml_frames_3bone(bm, df, dt, sp):
    sp_arr = np.array(sp, dtype=np.float32)
    med_low_z = detect_medial_is_low_z_3bone(bm, sp)
    tb = bm['tibia'] > 0
    inds_tb = np.argwhere(tb & (dt >= -2.5) & (df <= 34.0))
    if len(inds_tb) < 20:
        inds_tb = np.argwhere(tb)
    if len(inds_tb) < 10:
        Z, Y, X = tb.shape
        c_mid = np.array([0.5 * Z * sp[0], 0.5 * Y * sp[1], 0.5 * X * sp[2]], dtype=np.float32)
        span = np.array([50.0, 20.0, 40.0], dtype=np.float32)
        return med_low_z, (c_mid - 20, c_mid + 20, c_mid, span), (c_mid, span, c_mid, span, float(c_mid[0]))
    z0, y0, x0 = np.maximum(inds_tb.min(axis=0) - 2, 0)
    z1, y1, x1 = np.minimum(inds_tb.max(axis=0) + 3, tb.shape)
    dts = gaussian_filter(dt[z0:z1, y0:y1, x0:x1], sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
    _, nyt, _ = np.gradient(dts, *sp)

    plat_sub = (
        tb[z0:z1, y0:y1, x0:x1]
        & (dt[z0:z1, y0:y1, x0:x1] >= -1.5)
        & (df[z0:z1, y0:y1, x0:x1] <= 26.0)
        & (nyt <= -0.35)
    )
    if np.sum(plat_sub) < 60:
        plat_sub = tb[z0:z1, y0:y1, x0:x1] & (dt[z0:z1, y0:y1, x0:x1] >= -2.0) & (df[z0:z1, y0:y1, x0:x1] <= 28.0)
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
    if len(pts_low) < 5:
        pts_low = pts
    if len(pts_high) < 5:
        pts_high = pts
    pts_M = pts_low if med_low_z else pts_high
    pts_L = pts_high if med_low_z else pts_low
    c_M, s_M = pts_M.mean(axis=0), pts_M.std(axis=0) + 1.2
    c_L, s_L = pts_L.mean(axis=0), pts_L.std(axis=0) + 1.2
    return med_low_z, (p_min, p_max, p_mid, p_span), (c_M, s_M, c_L, s_L, z_med_split)


def compute_atlas_bins_oai_v6(coords_mm, dfs_m, dts_m, dps_m, f_stats, t_comp, p_stats, ml_info):
    med_low_z, (p_min, p_max, p_mid, p_span), (c_M, s_M, c_L, s_L, z_med_split) = ml_info

    c1_t, s1_t, c2_t, s2_t, spine_t, z_mid_t = t_comp
    in_1t = coords_mm[:, 0] < z_mid_t
    uz_t = np.where(in_1t, (c1_t[0] - coords_mm[:, 0]) / s1_t[0], (coords_mm[:, 0] - c2_t[0]) / s2_t[0])
    uy_t = np.where(in_1t, (coords_mm[:, 1] - c1_t[1]) / s1_t[1], (coords_mm[:, 1] - c2_t[1]) / s2_t[1])
    ux_t = np.where(in_1t, (coords_mm[:, 2] - c1_t[2]) / s1_t[2], (coords_mm[:, 2] - c2_t[2]) / s2_t[2])
    bz_old = np.clip(((uz_t + 2.8) / 5.6 * 26).astype(int), 0, 25)
    bx_old = np.clip(((ux_t + 2.8) / 5.6 * 26).astype(int), 0, 25)
    bdt_16 = np.clip(((dts_m + 0.8) / 7.6 * 16).astype(int), 0, 15)
    bdf_18 = np.clip(((dfs_m + 0.8) / 27.0 * 18).astype(int), 0, 17)

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
    bdf_f = np.clip(((dfs_m + 0.8) / 7.6 * 16).astype(int), 0, 15)

    rel_fz_ml = rel_f[:, 0] if med_low_z else (1.0 - rel_f[:, 0])
    bz_fml = np.clip(((rel_fz_ml + 0.10) / 1.20 * 34).astype(int), 0, 33)
    is_lat_f = (rel_fz_ml >= 0.5).astype(int)
    uz_fsep = np.where(is_lat_f == 1, (rel_fz_ml - 0.5) * 2.0, (0.5 - rel_fz_ml) * 2.0)
    bz_fsep = np.clip((uz_fsep / 1.25 * 22).astype(int), 0, 21)

    pat_min, pat_max, _ = p_stats
    rel_pat = (coords_mm - pat_min) / (pat_max - pat_min + 1e-6)
    rel_pz_ml = rel_pat[:, 0] if med_low_z else (1.0 - rel_pat[:, 0])
    z_sym_pat = np.abs(rel_pz_ml - 0.5) * 2.0
    bz_po = np.clip((z_sym_pat / 1.30 * 20).astype(int), 0, 19)
    bz_pml = np.clip(((rel_pz_ml + 0.15) / 1.30 * 28).astype(int), 0, 27)
    by_pat = np.clip(((rel_pat[:, 1] + 0.15) / 1.30 * 24).astype(int), 0, 23)
    bdp_16 = np.clip(((dps_m + 0.8) / 8.2 * 16).astype(int), 0, 15)
    bdf_p18 = np.clip(((dfs_m + 0.8) / 22.0 * 18).astype(int), 0, 17)
    is_lat_pat = (rel_pz_ml >= 0.5).astype(int)
    uz_psep = np.where(is_lat_pat == 1, (rel_pz_ml - 0.5) * 2.0, (0.5 - rel_pz_ml) * 2.0)
    bz_psep = np.clip((uz_psep / 1.30 * 18).astype(int), 0, 17)

    bins = {
        't_old': (bz_old, bx_old, bdt_16, bdf_18),
        't_ml': (bz_ml, bx_ml, bdt_16, bdf_18),
        't_mly': (bz_ml, by_ml, bx_ml, bdt_16),
        't_sep': (is_lat, bz_p, bx_p, bdt_16, bdf_18),
        't_wide': (bz_wide, bx_wide, bdt_16, bdf_18),
        'f_old': (bz_fo, by_f, bx_f, bdf_f),
        'f_ml': (bz_fml, by_f, bx_f, bdf_f),
        'f_sep': (is_lat_f, bz_fsep, by_f, bx_f, bdf_f),
        'p_old': (bz_po, by_pat, bdp_16, bdf_p18),
        'p_ml': (bz_pml, by_pat, bdp_16, bdf_p18),
        'p_sep': (is_lat_pat, bz_psep, by_pat, bdp_16, bdf_p18),
    }
    coords = {
        'uz_t': uz_t, 'uy_t': uy_t, 'ux_t': ux_t,
        'u_ml': u_ml, 'u_ap': u_ap, 'u_y': u_y,
        'is_lat': is_lat.astype(np.float32),
        'uz_p': uz_p, 'uy_p': uy_p, 'ux_p': ux_p,
        'spine_z': spine_t[0],
        'rel_f': rel_f, 'rel_fz_ml': rel_fz_ml, 'z_sym_f': z_sym_f,
        'rel_pat': rel_pat, 'rel_pz_ml': rel_pz_ml, 'z_sym_pat': z_sym_pat,
        'is_lat_pat': is_lat_pat.astype(np.float32),
    }
    return bins, coords


def prepare_geometry_oai_v6(img, sp, bm_ref, bm_alt, bm_raw, bone_name, atlas_p, dts_pre=None, landmarks_pre=None):
    from .rf_seg import _compute_articular_landmarks_and_stats
    if dts_pre is not None:
        df, dt, dp, df_alt, dt_alt, dp_alt, df_raw, dt_raw, dp_raw = dts_pre
    else:
        df = fast_signed_distance_transform(bm_ref['femur'] > 0, sp)
        dt = fast_signed_distance_transform(bm_ref['tibia'] > 0, sp)
        dp = fast_signed_distance_transform(bm_ref['patella'] > 0, sp) if ('patella' in bm_ref and np.any(bm_ref['patella'] > 0)) else np.full_like(df, 50.0)
        df_alt = fast_signed_distance_transform(bm_alt['femur'] > 0, sp)
        dt_alt = fast_signed_distance_transform(bm_alt['tibia'] > 0, sp)
        dp_alt = fast_signed_distance_transform(bm_alt['patella'] > 0, sp) if ('patella' in bm_alt and np.any(bm_alt['patella'] > 0)) else np.full_like(df, 50.0)
        df_raw = fast_signed_distance_transform(bm_raw['femur'] > 0, sp)
        dt_raw = fast_signed_distance_transform(bm_raw['tibia'] > 0, sp)
        dp_raw = fast_signed_distance_transform(bm_raw['patella'] > 0, sp) if ('patella' in bm_raw and np.any(bm_raw['patella'] > 0)) else np.full_like(df, 50.0)

    if landmarks_pre is not None:
        f_stats, f_comp, t_stats, t_comp, p_stats, p_comp, ml_info = landmarks_pre
    else:
        f_stats, _, f_comp = _compute_articular_landmarks_and_stats(bm_ref['femur'], df, np.minimum(dt, dp), 45.0, sp)
        t_stats, _, t_comp = _compute_articular_landmarks_and_stats(bm_ref['tibia'], dt, df, 25.0, sp)
        p_stats, _, p_comp = _compute_articular_landmarks_and_stats(bm_ref.get('patella', bm_ref['tibia']), dp, df, 25.0, sp)
        ml_info = compute_ml_frames_3bone(bm_ref, df, dt, sp)

    if bone_name == 'tibia':
        shell = (dt >= -1.2) & (dt <= 6.4) & (df >= -1.2) & (df <= 27.0)
    elif bone_name == 'patella':
        shell = (dp >= -1.2) & (dp <= 7.6) & (df >= -1.2) & (df <= 22.0)
    else:
        raw_shell = (df >= -1.2) & (df <= 6.6) & (dt >= -1.2) & (dp >= -1.2) & ((dt <= 56.0) | (dp <= 26.0))
        raw_inds = np.argwhere(raw_shell)
        if len(raw_inds) > 0:
            coords_raw = raw_inds.astype(np.float32) * np.array(sp, dtype=np.float32)
            dfs_raw = df[raw_inds[:, 0], raw_inds[:, 1], raw_inds[:, 2]]
            dts_raw = dt[raw_inds[:, 0], raw_inds[:, 1], raw_inds[:, 2]]
            dps_raw = dp[raw_inds[:, 0], raw_inds[:, 1], raw_inds[:, 2]]
            bins_r, c_dict_r = compute_atlas_bins_oai_v6(coords_raw, dfs_raw, dts_raw, dps_raw, f_stats, t_comp, p_stats, ml_info)
            rel_f_r = c_dict_r['rel_f']
            in_box = (
                (rel_f_r[:, 0] >= -0.12) & (rel_f_r[:, 0] <= 1.12)
                & (rel_f_r[:, 1] >= -0.15) & (rel_f_r[:, 1] <= 1.08)
                & (rel_f_r[:, 2] >= -0.15) & (rel_f_r[:, 2] <= 1.12)
            )
            p_ens = (
                0.45 * atlas_p['pf_fml_fc'][bins_r['f_ml']]
                + 0.35 * atlas_p['pf_sep_fc'][bins_r['f_sep']]
                + 0.20 * atlas_p['pf_old_fc'][bins_r['f_old']]
            )
            p3d = atlas_p['p3d_fml_fc'][bins_r['f_ml'][0], bins_r['f_ml'][1], bins_r['f_ml'][2]]
            keep = in_box & ((p_ens > 0.001) | (p3d > 0.002) | (dts_raw <= 42.0) | (dps_raw <= 20.0))
            shell = np.zeros_like(raw_shell, dtype=bool)
            kept_inds = raw_inds[keep]
            shell[kept_inds[:, 0], kept_inds[:, 1], kept_inds[:, 2]] = True
        else:
            shell = raw_shell

    all_inds = np.argwhere(shell)
    if len(all_inds) == 0:
        z0, z1, y0, y1, x0, x1 = 0, img.shape[0], 0, img.shape[1], 0, img.shape[2]
    else:
        margin = [int(np.ceil(6.0 / s)) for s in sp]
        z0, y0, x0 = np.maximum(all_inds.min(axis=0) - margin, 0)
        z1, y1, x1 = np.minimum(all_inds.max(axis=0) + margin + 1, img.shape)

    df_crop = df[z0:z1, y0:y1, x0:x1]
    dt_crop = dt[z0:z1, y0:y1, x0:x1]
    dp_crop = dp[z0:z1, y0:y1, x0:x1]
    df_alt_crop = df_alt[z0:z1, y0:y1, x0:x1]
    dt_alt_crop = dt_alt[z0:z1, y0:y1, x0:x1]
    dp_alt_crop = dp_alt[z0:z1, y0:y1, x0:x1]
    df_raw_crop = df_raw[z0:z1, y0:y1, x0:x1]
    dt_raw_crop = dt_raw[z0:z1, y0:y1, x0:x1]
    dp_raw_crop = dp_raw[z0:z1, y0:y1, x0:x1]

    sig_s = [2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]]
    dfs_crop = gaussian_filter(df_crop, sigma=sig_s)
    dts_crop = gaussian_filter(dt_crop, sigma=sig_s)
    dps_crop = gaussian_filter(dp_crop, sigma=sig_s)

    if bone_name == 'femur':
        d_self_crop = df_crop
        d_other_crop = np.minimum(dt_crop, dp_crop)
        d_alt_self_crop = df_alt_crop
        d_raw_self_crop = df_raw_crop
        ds_self_crop = dfs_crop
        ds_other_crop = np.minimum(dts_crop, dps_crop)
    elif bone_name == 'tibia':
        d_self_crop = dt_crop
        d_other_crop = df_crop
        d_alt_self_crop = dt_alt_crop
        d_raw_self_crop = dt_raw_crop
        ds_self_crop = dts_crop
        ds_other_crop = dfs_crop
    else:
        d_self_crop = dp_crop
        d_other_crop = df_crop
        d_alt_self_crop = dp_alt_crop
        d_raw_self_crop = dp_raw_crop
        ds_self_crop = dps_crop
        ds_other_crop = dfs_crop

    nz, ny, nx = np.gradient(ds_self_crop, *sp)
    norm = np.sqrt(nz**2 + ny**2 + nx**2) + 1e-6
    nz, ny, nx = (nz / norm).astype(np.float32), (ny / norm).astype(np.float32), (nx / norm).astype(np.float32)

    nzo, nyo, nxo = np.gradient(ds_other_crop, *sp)
    normo = np.sqrt(nzo**2 + nyo**2 + nxo**2) + 1e-6
    nzo, nyo, nxo = (nzo / normo).astype(np.float32), (nyo / normo).astype(np.float32), (nxo / normo).astype(np.float32)

    img_f32 = img[z0:z1, y0:y1, x0:x1].astype(np.float32)
    m_low = (d_self_crop <= -1.2) & (d_self_crop >= -5.5)
    m_high = (d_self_crop >= 0.4) & (d_self_crop <= 2.5) & (d_other_crop <= 12.0)
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

    img_lat_raw = gaussian_filter(img_f32, sigma=[1.2 / sp[0], 0.35 / sp[1], 1.2 / sp[2]])
    sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
    w_bn = ((d_self_crop <= -1.2) & (d_self_crop >= -5.5)).astype(np.float32) + 1e-5
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
        'dp_crop': dp_crop,
        'dfs_crop': dfs_crop,
        'dts_crop': dts_crop,
        'dps_crop': dps_crop,
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
        'p_stats': p_stats,
        'p_comp': p_comp,
        'ml_info': ml_info,
        'atlas_p': atlas_p,
    }


def extract_features_oai_v6(geo, target_mask, pm_self=None, pm_other=None):
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

    df_m = geo['df_crop'][zc, yc, xc]
    dt_m = geo['dt_crop'][zc, yc, xc]
    dp_m = geo['dp_crop'][zc, yc, xc]
    dfs_m = geo['dfs_crop'][zc, yc, xc]
    dts_m = geo['dts_crop'][zc, yc, xc]
    dps_m = geo['dps_crop'][zc, yc, xc]

    shift_ref_alt = d_raw_m - d_alt_m
    shift_ref_st2 = d_raw_m - d_st2_m
    d_raw_feat = d_raw_m
    d_alt_feat = d_alt_m
    d_s_feat = d_s_m
    d_s_anchor = d_s_m

    ap = geo['atlas_p']
    dfs_atl = d_s_anchor if bone_name == 'femur' else dfs_m
    dts_atl = d_s_anchor if bone_name == 'tibia' else dts_m
    dps_atl = d_s_anchor if bone_name == 'patella' else dps_m
    bins, c_dict = compute_atlas_bins_oai_v6(
        coords_mm, dfs_atl, dts_atl, dps_atl, geo['f_stats'], geo['t_comp'], geo['p_stats'], geo['ml_info']
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
    pf_old_pc = ap['pf_old_pc'][bins['f_old']]
    pf_fml_fc = ap['pf_fml_fc'][bins['f_ml']]
    pf_fml_tc = ap['pf_fml_tc'][bins['f_ml']]
    pf_fml_pc = ap['pf_fml_pc'][bins['f_ml']]
    pf_sep_fc = ap['pf_sep_fc'][bins['f_sep']]
    pf_sep_tc = ap['pf_sep_tc'][bins['f_sep']]
    pf_sep_pc = ap['pf_sep_pc'][bins['f_sep']]
    pf_ens_fc = 0.45 * pf_fml_fc + 0.35 * pf_sep_fc + 0.20 * pf_old_fc
    pf_ens_tc = 0.50 * pf_fml_tc + 0.50 * pf_sep_tc
    pf_ens_pc = 0.50 * pf_fml_pc + 0.30 * pf_sep_pc + 0.20 * pf_old_pc

    bdf_m1, bdf_p1 = np.clip(bins['f_ml'][3] - 1, 0, 15), np.clip(bins['f_ml'][3] + 1, 0, 15)
    pf_fml_max3 = np.maximum(
        np.maximum(ap['pf_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2], bdf_m1], pf_fml_fc),
        ap['pf_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2], bdf_p1],
    )
    p3d_fml_v = ap['p3d_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2]]
    edf_fml_v = ap['edf_fml_3d'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2]]

    pp_old_pc = ap['pp_old_pc'][bins['p_old']]
    pp_old_fc = ap['pp_old_fc'][bins['p_old']]
    pp_ml_pc = ap['pp_ml_pc'][bins['p_ml']]
    pp_ml_fc = ap['pp_ml_fc'][bins['p_ml']]
    pp_sep_pc = ap['pp_sep_pc'][bins['p_sep']]
    pp_sep_fc = ap['pp_sep_fc'][bins['p_sep']]
    pp_ens_pc = 0.45 * pp_ml_pc + 0.35 * pp_sep_pc + 0.20 * pp_old_pc
    pp_ens_fc = 0.50 * pp_ml_fc + 0.50 * pp_sep_fc
    bdp_m1, bdp_p1 = np.clip(bins['p_ml'][2] - 1, 0, 15), np.clip(bins['p_ml'][2] + 1, 0, 15)
    pp_ml_max3 = np.maximum(
        np.maximum(ap['pp_ml_pc'][bins['p_ml'][0], bins['p_ml'][1], bdp_m1, bins['p_ml'][3]], pp_ml_pc),
        ap['pp_ml_pc'][bins['p_ml'][0], bins['p_ml'][1], bdp_p1, bins['p_ml'][3]],
    )
    p2d_pml_v = ap['p2d_pml_pc'][bins['p_ml'][0], bins['p_ml'][1]]
    edp_pml_v = ap['edp_pml_2d'][bins['p_ml'][0], bins['p_ml'][1]]

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
        if np.sum(m_p1_core) >= 40:
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
    _s_out = np.partition(rv_lat[:, 5:14], (6, 7), axis=1)
    r_out = 0.05 * _s_out[:, 6:7] + 0.95 * _s_out[:, 7:8]
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

    vx_lat = np.empty((N, len(_DX_STEPS)), dtype=np.float32)
    vx_zb = np.empty((N, len(_DX_STEPS)), dtype=np.float32)
    for k, dx_px in enumerate(_DX_STEPS):
        sx = np.clip(xc + dx_px, 0, cx_max - 1)
        vx_lat[:, k] = img_lat[zc, yc, sx]
        vx_zb[:, k] = z_bone[zc, yc, sx]
    vxn_lat = (vx_lat - r_in) / r_span
    vxg_lat = np.diff(vx_lat, axis=1)

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

    if bone_name == 'femur':
        comp = geo['f_comp']
        c_stats = geo['f_stats']
    elif bone_name == 'tibia':
        comp = geo['t_comp']
        c_stats = geo['t_stats']
    else:
        comp = geo['p_comp']
        c_stats = geo['p_stats']

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

    c_min, c_max, _ = c_stats
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
    rel_pat = c_dict['rel_pat']

    base_cols = np.column_stack([
        v0_m, vlat_m, vs1_m, vs2_m, vs4_m, vs8_m, z_m, absz_m, zb_m,
        v0_m - vlat_m, v0_m - vs1_m, vs1_m - vs2_m, vs2_m - vs4_m, v0_m - vs4_m, v0_m - vs8_m,
        gmag0[zc, yc, xc], gmag1[zc, yc, xc], gn0, gn1, gy0[zc, yc, xc], gy1[zc, yc, xc], gx0[zc, yc, xc], gx1[zc, yc, xc],
        d_raw_feat, d_alt_feat, 0.5 * (d_raw_feat + d_alt_feat), shift_ref_alt, shift_ref_st2, d_s_feat,
        do_raw_m, do_s_m, df_m, dt_m, dp_m, dfs_m, dts_m, dps_m,
        d_s_feat + do_s_m, d_s_feat - do_s_m,
        np.maximum(d_s_feat, 0.0) / (np.maximum(d_s_feat, 0.0) + np.maximum(do_s_m, 0.0) + 0.5),
        nz_m, ny_m, nx_m, nzo_m, nyo_m, nxo_m,
        nz_m * nzo_m + ny_m * nyo_m + nx_m * nxo_m,
        c_dict['is_lat'], c_dict['u_ml'], c_dict['u_ap'], c_dict['u_y'],
        uz_p, uy_p, ux_p, d_zx_p, c_dict['rel_fz_ml'],
        rel_pat[:, 0], rel_pat[:, 1], rel_pat[:, 2], c_dict['rel_pz_ml'], c_dict['z_sym_pat'],
        uz, uy, ux, np.abs(uz), d_zx, d_spine_z,
        rel_box[:, 0], rel_box[:, 1], rel_box[:, 2], z_sym,
        pt_ens_tc, pt_ens_fc, pt_old_tc, pt_old_fc, pt_ml_tc, pt_sep_tc, pt_mly_tc, pt_wide_tc, pt_ml_max3,
        pt_ens_tc - pt_ens_fc, pt_ens_tc - pt_old_tc, pt_wide_tc - pt_ml_tc,
        p2d_ens, p2d_wide_v, edt_ens, edt_wide_v, d_s_feat - 2.0 * edt_ens, d_s_feat / (2.0 * edt_ens + 0.5),
        pf_ens_fc, pf_ens_tc, pf_ens_pc, pf_old_fc, pf_old_tc, pf_fml_fc, pf_sep_fc, pf_fml_max3,
        pf_ens_fc - pf_ens_tc, pf_ens_fc - pf_ens_pc, pf_ens_fc - pf_old_fc,
        p3d_fml_v, edf_fml_v, d_s_feat - 2.0 * edf_fml_v, d_s_feat / (2.0 * edf_fml_v + 0.5),
        pp_ens_pc, pp_ens_fc, pp_old_pc, pp_old_fc, pp_ml_pc, pp_sep_pc, pp_ml_max3,
        pp_ens_pc - pp_ens_fc, pp_ens_pc - pp_old_pc,
        p2d_pml_v, edp_pml_v, d_s_feat - 2.0 * edp_pml_v, d_s_feat / (2.0 * edp_pml_v + 0.5),
        r_in[:, 0], r_out[:, 0], r_out[:, 0] - r_in[:, 0],
        idx_rise, max_rise, idx_drop, min_drop,
        idx_cort_min, val_cort_min, idx_rise_zb, max_rise_zb,
    ])

    blocks = [
        base_cols,
        rv0, rv_lat, rv1, rv_zb, rn0, rn_lat, rg0, rg_lat, rgn_lat, rlap_lat, rg_zb,
        vy_lat, vyn_lat, vyg_lat, vy_zb,
        vx_lat, vxn_lat, vxg_lat, vx_zb,
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
        in_loc_lat = gaussian_filter(in_loc_vol, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])

        p_ct_lat = gaussian_filter(p_ct_c, sigma=[1.4 / sp[0], 0.35 / sp[1], 1.4 / sp[2]])
        p_ct_lat_wide = gaussian_filter(p_ct_c, sigma=[2.8 / sp[0], 0.45 / sp[1], 2.8 / sp[2]])
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

        c_ant_vol = np.cumsum(p_ct_c * valid_c, axis=2) * sp[2]
        t_row_2d = c_ant_vol[:, :, -1:]
        c_post_vol = np.maximum(0.0, t_row_2d - c_ant_vol)
        frac_ant_vol = c_ant_vol / (t_row_2d + 0.25)

        t_sheet_s2 = gaussian_filter(t_col_2d, sigma=[2.2 / sp[0], 0.0, 2.2 / sp[2]])
        t_sheet_s4 = gaussian_filter(t_col_2d, sigma=[4.5 / sp[0], 0.0, 4.5 / sp[2]])
        pmax_col_2d = np.max(p_ct_c, axis=1, keepdims=True)
        pmax_sheet_s2 = gaussian_filter(pmax_col_2d, sigma=[2.2 / sp[0], 0.0, 2.2 / sp[2]])
        pmax_sheet_s4 = gaussian_filter(pmax_col_2d, sigma=[4.5 / sp[0], 0.0, 4.5 / sp[2]])

        p_ray_steps = np.array([-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.8, 2.6], dtype=np.float32)
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
        c_out_ray = np.sum(pr_ct[:, 5:10], axis=1) * 0.45
        t_ray_tot = c_in_ray + c_out_ray
        f_in_ray = c_in_ray / (t_ray_tot + 0.20)

        py_ct = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
        py_bn = np.empty((N, len(_DY_STEPS)), dtype=np.float32)
        for k, dy_px in enumerate(_DY_STEPS):
            sy = np.clip(yc + dy_px, 0, cy_max - 1)
            py_ct[:, k] = p_ct_lat[zc, sy, xc]
            py_bn[:, k] = p_bn_c[zc, sy, xc]

        px_ct = np.empty((N, len(_DX_STEPS)), dtype=np.float32)
        px_bn = np.empty((N, len(_DX_STEPS)), dtype=np.float32)
        for k, dx_px in enumerate(_DX_STEPS):
            sx = np.clip(xc + dx_px, 0, cx_max - 1)
            px_ct[:, k] = p_ct_lat[zc, yc, sx]
            px_bn[:, k] = p_bn_c[zc, yc, sx]

        ph_ct = np.empty((N, len(_H_GRID)), dtype=np.float32)
        ph_ot = np.empty((N, len(_H_GRID)), dtype=np.float32)
        for k, h_val in enumerate(_H_GRID):
            st = h_val - d_s_anchor
            sz = np.clip(np.round(zc + st * nz_m / sp[0]).astype(int), 0, cz_max - 1)
            sy = np.clip(np.round(yc + st * ny_m / sp[1]).astype(int), 0, cy_max - 1)
            sx = np.clip(np.round(xc + st * nx_m / sp[2]).astype(int), 0, cx_max - 1)
            ph_ct[:, k] = p_ct_lat[sz, sy, sx]
            ph_ot[:, k] = p_ot_s1[sz, sy, sx]
        col_psum = np.sum(ph_ct[:, 2:], axis=1) * 0.55
        col_pmax = np.max(ph_ct[:, 2:], axis=1)
        col_otsum = np.sum(ph_ot[:, 2:], axis=1) * 0.55

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
        c_ant_m = c_ant_vol[zc, yc, xc]
        c_post_m = c_post_vol[zc, yc, xc]
        f_ant_m = frac_ant_vol[zc, yc, xc]
        t_row_m = t_row_2d[zc, yc, 0]
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
            c_ant_m, c_post_m, f_ant_m, t_row_m,
            c_in_ray, c_out_ray, t_ray_tot, f_in_ray,
            t_col_m, t_sh2_m, t_sh4_m, t_col_m - t_sh2_m, c_bl_m - t_sh2_m,
            pm_col_m, pm_sh2_m, pm_sh4_m, pm_col_m - pm_sh2_m,
        ])
        blocks.extend([p_scalar_cols, pr_ct, pr_bn, pr_bg, pr_ot, pr_inloc, py_ct, py_bn, px_ct, px_bn, ph_ct, ph_ot, ptan_ct])

    return np.hstack(blocks).astype(np.float32, copy=False)


def combine_tp_for_femur(pm_t, pm_p, dt, dp):
    use_p = dp < dt
    out = np.where(use_p[..., None], pm_p, pm_t).astype(np.float32)
    out[..., 1] = np.maximum(pm_t[..., 1], pm_p[..., 1])
    out[..., 2] = np.maximum(pm_t[..., 2], pm_p[..., 2])
    out[..., 0] = np.clip(1.0 - out[..., 1] - out[..., 2] - out[..., 3], 0.0, 1.0)
    return out


def _p1_folds_screened_oai(geo, clfs):
    sh = geo['shell']
    if not np.any(sh):
        z4 = np.zeros((*sh.shape, 4), dtype=np.float32)
        z4[..., 0] = 1.0
        return z4
    X = extract_features_oai_v6(geo, sh)
    p0 = clfs[0].predict_proba(X).astype(np.float32)
    act = (p0[:, 2] >= 0.02) | (p0[:, 3] >= 0.02) | ((p0[:, 1] >= 0.02) & (p0[:, 1] <= 0.97))
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
    return pm_avg


def _p2_screened_oai(geo, pm1_s, pm1_o, clfs_hgb, clf_et):
    sh = geo['shell']
    act = sh & ((pm1_s[..., 2] >= 0.010) | (pm1_s[..., 3] >= 0.010) | (pm1_o[..., 2] >= 0.015))
    if not np.any(act):
        return pm1_s.copy()
    X2 = extract_features_oai_v6(geo, act, pm_self=pm1_s, pm_other=pm1_o)
    p0 = clfs_hgb[0].predict_proba(X2).astype(np.float32)
    bnd = (p0[:, 2] >= 0.012) & (p0[:, 2] <= 0.988)
    p_hgb = p0.copy()
    if len(clfs_hgb) > 1 and np.any(bnd):
        X2_b = X2[bnd]
        p_hgb[bnd] = (p0[bnd] + sum(c.predict_proba(X2_b) for c in clfs_hgb[1:])) / float(len(clfs_hgb))
    p_et = p_hgb.copy()
    if clf_et is not None and np.any(bnd):
        if getattr(clf_et, 'n_jobs', 1) not in range(1, 17):
            clf_et.n_jobs = min(max(os.cpu_count() or 8, 1), 16)
        with joblib.parallel_backend('threading'):
            p_et[bnd] = clf_et.predict_proba(X2[bnd]).astype(np.float32)
    del X2
    w_et = 0.22
    pm2 = pm1_s.copy()
    pm2[act] = (1.0 - w_et) * p_hgb + w_et * p_et
    return pm2


def _shift_ax(arr, shift, axis):
    if shift == 0:
        return arr
    pad_w = [(0, 0)] * arr.ndim
    if shift > 0:
        pad_w[axis] = (shift, 0)
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(None, -shift)
        return np.pad(arr, pad_w, mode='edge')[tuple(sl)]
    else:
        s = -shift
        pad_w[axis] = (0, s)
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(s, None)
        return np.pad(arr, pad_w, mode='edge')[tuple(sl)]


def extract_s3_compartment_features(
    bone_name, img_roi, sp, d_self_v2, d_self_v1, d_self_v0, d_other_v2,
    df2, dt2, dp2, pm1_s, pm2_s, pm_opp_ct, a_new, a_old, edt_exp
):
    sig_sm = [0.30 / sp[0], 0.15 / sp[1], 0.25 / sp[2]]
    p_bg = gaussian_filter(pm2_s[..., 0].astype(np.float32), sigma=sig_sm)
    p_bn = gaussian_filter(pm2_s[..., 1].astype(np.float32), sigma=sig_sm)
    p_ct = gaussian_filter(pm2_s[..., 2].astype(np.float32), sigma=sig_sm)
    p_ot = gaussian_filter(pm2_s[..., 3].astype(np.float32), sigma=sig_sm)

    p1_bn = gaussian_filter(pm1_s[..., 1].astype(np.float32), sigma=sig_sm)
    p1_ct = gaussian_filter(pm1_s[..., 2].astype(np.float32), sigma=sig_sm)
    p_opp = gaussian_filter(pm_opp_ct.astype(np.float32), sigma=sig_sm)

    del_a = a_new - a_old
    sc_part = p_ct - 0.20 * p_bn + 0.28 * del_a
    sc = sc_part + 0.25 * np.minimum(p_ct, p_opp)
    s_bot = p_ct / np.maximum(p_ct + p_bn, 1e-4)

    max_d = 7.2 if bone_name == 'patella' else 6.2
    cand = ((sc >= 0.12) | (p_ct >= 0.10) | (p1_ct >= 0.14)) & (d_self_v2 >= -1.2) & (d_self_v2 <= max_d)
    if not np.any(cand):
        return cand, sc, sc_part, np.zeros((0, 92), dtype=np.float32)

    bm2 = d_self_v2 <= 0.0
    d_in_3d = distance_transform_edt(bm2, sampling=sp).astype(np.float32)
    d_out_3d = distance_transform_edt(~bm2, sampling=sp).astype(np.float32)
    sdt_3d_true = d_out_3d - (d_in_3d - 0.5 * sp[1])

    img_f0 = gaussian_filter(img_roi.astype(np.float32), sigma=[0.35 / s for s in sp])
    img_lat = gaussian_filter(img_roi.astype(np.float32), sigma=[1.2 / sp[0], 0.35 / sp[1], 1.2 / sp[2]])
    sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
    w_bn = ((d_self_v2 <= -1.2) & (d_self_v2 >= -5.5)).astype(np.float32) + 1e-5
    w_bn_s = gaussian_filter(w_bn, sigma=sig_bn)
    i_bn_mean = gaussian_filter(img_lat * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_var = gaussian_filter(((img_lat - i_bn_mean) ** 2) * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_std = np.maximum(np.sqrt(np.maximum(i_bn_var, 0.0)), 3.0)
    zb_raw = ((img_lat - i_bn_mean) / i_bn_std).astype(np.float32)

    core_c = p_ct >= 0.65
    zb_med = float(np.median(zb_raw[core_c])) if np.any(core_c) else 10.0
    zb_norm = zb_raw / max(zb_med, 2.5)

    sig_tan = [3.2 / sp[0], 1.4 / sp[1], 3.2 / sp[2]]
    w_ct = (np.clip(p_ct - 0.30, 0.0, 1.0) ** 2) + 1e-5
    i_sheet = gaussian_filter(img_lat * w_ct, sigma=sig_tan) / gaussian_filter(w_ct, sigma=sig_tan)
    i_bone = gaussian_filter(img_lat * w_bn, sigma=sig_tan) / gaussian_filter(w_bn, sigma=sig_tan)
    denom_i = np.maximum(i_sheet - i_bone, 5.0)
    inorm_f0 = (img_f0 - i_bone) / denom_i
    inorm_lat = (img_lat - i_bone) / denom_i

    gz0, gy0, gx0 = np.gradient(inorm_f0, *sp)
    gmag0 = np.sqrt(gz0**2 + gy0**2 + gx0**2)
    ds_sm = gaussian_filter(d_self_v2, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
    nz_g, ny_g, nx_g = np.gradient(ds_sm, *sp)
    n_norm = np.sqrt(nz_g**2 + ny_g**2 + nx_g**2) + 1e-6
    nz_u, ny_u, nx_u = (nz_g / n_norm).astype(np.float32), (ny_g / n_norm).astype(np.float32), (nx_g / n_norm).astype(np.float32)
    gn0 = gz0 * nz_u + gy0 * ny_u + gx0 * nx_u

    thick_y_mm = np.sum(p_ct, axis=1, keepdims=True) * sp[1]
    thick_x_mm = np.sum(p_ct, axis=2, keepdims=True) * sp[2]
    p_col_max = np.max(p_ct, axis=1, keepdims=True)
    p_row_max = np.max(p_ct, axis=2, keepdims=True)

    sc_s1 = gaussian_filter(sc, sigma=[0.8 / sp[0], 0.5 / sp[1], 0.8 / sp[2]])
    sc_s2 = gaussian_filter(sc, sigma=[1.8 / sp[0], 1.0 / sp[1], 1.8 / sp[2]])
    p_lat = gaussian_filter(p_ct, sigma=[1.5 / sp[0], 0.3 / sp[1], 1.5 / sp[2]])
    sbot_lat = gaussian_filter(s_bot, sigma=[2.0 / sp[0], 0.5 / sp[1], 2.0 / sp[2]])

    m = cand
    zs_m, ys_m, xs_m = np.where(m)
    nz_v, ny_v, nx_v = img_roi.shape
    nzm, nym, nxm = nz_u[m], ny_u[m], nx_u[m]

    ray_steps = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.2]
    nr_sc = np.empty((len(zs_m), len(ray_steps)), dtype=np.float32)
    nr_in0 = np.empty((len(zs_m), len(ray_steps)), dtype=np.float32)
    nr_zbn = np.empty((len(zs_m), len(ray_steps)), dtype=np.float32)
    for k, d_r in enumerate(ray_steps):
        rz = np.clip(np.round(zs_m + d_r * nzm / sp[0]).astype(int), 0, nz_v - 1)
        ry = np.clip(np.round(ys_m + d_r * nym / sp[1]).astype(int), 0, ny_v - 1)
        rx = np.clip(np.round(xs_m + d_r * nxm / sp[2]).astype(int), 0, nx_v - 1)
        nr_sc[:, k] = sc[rz, ry, rx]
        nr_in0[:, k] = inorm_f0[rz, ry, rx]
        nr_zbn[:, k] = zb_norm[rz, ry, rx]

    def bcast(arr):
        return np.broadcast_to(arr, img_roi.shape)[m]

    X_s3 = np.hstack([
        np.column_stack([
            sc[m], sc_part[m], s_bot[m], sc_s1[m], sc_s2[m], p_lat[m], sbot_lat[m],
            sc[m] - sc_s1[m], sc[m] - sc_s2[m], s_bot[m] - sbot_lat[m],
            p_ct[m], p_bn[m], p_bg[m], p_ot[m], p_opp[m],
            p1_ct[m], p1_bn[m], p_ct[m] - p1_ct[m], p_bn[m] - p1_bn[m],
            p_ct[m] / np.maximum(p_ct[m] + p_bg[m], 1e-4),
            p_ct[m] - p_opp[m],
            d_self_v2[m], d_self_v1[m], d_self_v0[m],
            d_self_v2[m] - d_self_v1[m], d_self_v2[m] - d_self_v0[m],
            d_other_v2[m], df2[m], dt2[m], dp2[m],
            sdt_3d_true[m], d_in_3d[m],
            a_new[m], a_old[m], del_a[m], edt_exp[m],
            d_self_v2[m] - 2.0 * edt_exp[m], d_self_v2[m] / np.maximum(2.0 * edt_exp[m], 0.8),
            bcast(thick_y_mm), bcast(thick_x_mm), bcast(p_col_max), bcast(p_row_max),
            zb_norm[m], inorm_f0[m], inorm_lat[m], inorm_f0[m] - inorm_lat[m],
            gz0[m], gy0[m], gx0[m], gmag0[m], gn0[m],
            nzm, nym, nxm,
            _shift_ax(sc, 1, 1)[m], _shift_ax(sc, -1, 1)[m], _shift_ax(sc, 2, 1)[m], _shift_ax(sc, -2, 1)[m],
            _shift_ax(sc, 1, 2)[m], _shift_ax(sc, -1, 2)[m], _shift_ax(sc, 2, 2)[m], _shift_ax(sc, -2, 2)[m],
            _shift_ax(sc, 1, 0)[m], _shift_ax(sc, -1, 0)[m],
            _shift_ax(p_ct, 1, 1)[m], _shift_ax(p_ct, -1, 1)[m],
            _shift_ax(p_ct, 1, 2)[m], _shift_ax(p_ct, -1, 2)[m],
            _shift_ax(p_bn, 1, 1)[m], _shift_ax(p_bn, -1, 1)[m],
            _shift_ax(p_bn, 1, 2)[m], _shift_ax(p_bn, -1, 2)[m],
            _shift_ax(inorm_f0, 1, 1)[m] - inorm_f0[m], _shift_ax(inorm_f0, -1, 1)[m] - inorm_f0[m],
            _shift_ax(inorm_f0, 1, 2)[m] - inorm_f0[m], _shift_ax(inorm_f0, -1, 2)[m] - inorm_f0[m],
            _shift_ax(zb_norm, 1, 1)[m] - zb_norm[m], _shift_ax(zb_norm, -1, 1)[m] - zb_norm[m],
            _shift_ax(zb_norm, 1, 2)[m] - zb_norm[m], _shift_ax(zb_norm, -1, 2)[m] - zb_norm[m],
        ]),
        nr_sc, nr_in0, nr_zbn,
    ]).astype(np.float32)
    return cand, sc, sc_part, X_s3


def compute_roi_priors_oai(sl, sp, df, dt, dp, f_stats, t_comp, p_stats, ml_info, ap):
    uz0, uz1, uy0, uy1, ux0, ux1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, sl[2].start, sl[2].stop
    df_c, dt_c, dp_c = df[sl], dt[sl], dp[sl]
    sig_s = [2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]]
    dfs_c = gaussian_filter(df_c, sigma=sig_s)
    dts_c = gaussian_filter(dt_c, sigma=sig_s)
    dps_c = gaussian_filter(dp_c, sigma=sig_s)
    zz, yy, xx = np.meshgrid(
        np.arange(uz0, uz1), np.arange(uy0, uy1), np.arange(ux0, ux1), indexing='ij'
    )
    coords_c = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(np.float32) * np.array(sp, dtype=np.float32)
    bins_c, _ = compute_atlas_bins_oai_v6(
        coords_c, dfs_c.ravel(), dts_c.ravel(), dps_c.ravel(), f_stats, t_comp, p_stats, ml_info
    )
    sh_c = df_c.shape
    a_old_fc = ap['pf_old_fc'][bins_c['f_old']].reshape(sh_c)
    a_new_fc = (
        0.45 * ap['pf_fml_fc'][bins_c['f_ml']]
        + 0.35 * ap['pf_sep_fc'][bins_c['f_sep']]
        + 0.20 * ap['pf_old_fc'][bins_c['f_old']]
    ).reshape(sh_c)
    edf_exp = ap['edf_fml_3d'][bins_c['f_ml'][0], bins_c['f_ml'][1], bins_c['f_ml'][2]].reshape(sh_c)

    a_old_tc = ap['pt_old_tc'][bins_c['t_old']].reshape(sh_c)
    a_new_tc = (
        0.32 * ap['pt_ml_tc'][bins_c['t_ml']]
        + 0.30 * ap['pt_sep_tc'][bins_c['t_sep']]
        + 0.20 * ap['pt_mly_tc'][bins_c['t_mly']]
        + 0.18 * ap['pt_wide_tc'][bins_c['t_wide']]
    ).reshape(sh_c)
    edt_exp = (
        0.40 * ap['edt_ml_2d'][bins_c['t_ml'][0], bins_c['t_ml'][1]]
        + 0.35 * ap['edt_sep_2d'][bins_c['t_sep'][0], bins_c['t_sep'][1], bins_c['t_sep'][2]]
        + 0.25 * ap['edt_wide_2d'][bins_c['t_wide'][0], bins_c['t_wide'][1]]
    ).reshape(sh_c)

    a_old_pc = ap['pp_old_pc'][bins_c['p_old']].reshape(sh_c)
    a_new_pc = (
        0.45 * ap['pp_ml_pc'][bins_c['p_ml']]
        + 0.35 * ap['pp_sep_pc'][bins_c['p_sep']]
        + 0.20 * ap['pp_old_pc'][bins_c['p_old']]
    ).reshape(sh_c)
    edp_exp = ap['edp_pml_2d'][bins_c['p_ml'][0], bins_c['p_ml'][1]].reshape(sh_c)

    return {
        'a_old_fc': a_old_fc, 'a_new_fc': a_new_fc, 'edf_exp': edf_exp,
        'a_old_tc': a_old_tc, 'a_new_tc': a_new_tc, 'edt_exp': edt_exp,
        'a_old_pc': a_old_pc, 'a_new_pc': a_new_pc, 'edp_exp': edp_exp,
    }


def resolve_3cartilage_masks(
    img_shape, sl, sp, df_c, dt_c, dp_c,
    pm1_f_roi, pm1_t_roi, pm1_p_roi,
    pm2_f_roi, pm2_t_roi, pm2_p_roi,
    priors_roi, s3_scores=None, params=None
):
    if params is None:
        params = {}
    w_s3_f = float(params.get('w_s3_f', 0.42))
    w_s3_t = float(params.get('w_s3_t', 0.44))
    w_s3_p = float(params.get('w_s3_p', 0.44))
    th_f = float(params.get('th_f', 0.49))
    th_t = float(params.get('th_t', 0.48))
    th_p = float(params.get('th_p', 0.48))
    b_pen_f = float(params.get('b_pen_f', 0.20))
    b_pen_t = float(params.get('b_pen_t', 0.20))
    b_pen_p = float(params.get('b_pen_p', 0.20))

    sig_f = [0.25 / sp[0], 0.12 / sp[1], 0.20 / sp[2]]
    sig_t = [0.35 / sp[0], 0.15 / sp[1], 0.25 / sp[2]]
    sig_p = [0.35 / sp[0], 0.20 / sp[1], 0.20 / sp[2]]

    pf = gaussian_filter(pm2_f_roi[..., 2], sigma=sig_f)
    pfb = gaussian_filter(pm2_f_roi[..., 1], sigma=sig_f)
    pt = gaussian_filter(pm2_t_roi[..., 2], sigma=sig_t)
    ptb = gaussian_filter(pm2_t_roi[..., 1], sigma=sig_t)
    pp = gaussian_filter(pm2_p_roi[..., 2], sigma=sig_p)
    ppb = gaussian_filter(pm2_p_roi[..., 1], sigma=sig_p)

    del_f = priors_roi['a_new_fc'] - priors_roi['a_old_fc']
    del_t = priors_roi['a_new_tc'] - priors_roi['a_old_tc']
    del_p = priors_roi['a_new_pc'] - priors_roi['a_old_pc']

    sf_part = pf - b_pen_f * pfb + 0.25 * del_f
    st_part = pt - b_pen_t * ptb + 0.30 * del_t
    sp_part = pp - b_pen_p * ppb + 0.30 * del_p

    p_opp_f = np.maximum(pt, pp)
    sf_base = sf_part + 0.26 * np.minimum(pf, p_opp_f)
    st_base = st_part + 0.26 * np.minimum(pt, pf)
    sp_base = sp_part + 0.26 * np.minimum(pp, pf)

    if s3_scores is not None:
        if 'femur' in s3_scores and s3_scores['femur'] is not None:
            cand_f, p_s3_f = s3_scores['femur']
            v_s3_f = sf_base.copy()
            if np.any(cand_f) and len(p_s3_f) > 0:
                v_s3_f[cand_f] = (1.0 - w_s3_f) * sf_base[cand_f] + w_s3_f * (p_s3_f + 0.06)
            sf_fin = gaussian_filter(v_s3_f, sigma=sig_f)
        else:
            sf_fin = sf_base
        if 'tibia' in s3_scores and s3_scores['tibia'] is not None:
            cand_t, p_s3_t = s3_scores['tibia']
            v_s3_t = st_base.copy()
            if np.any(cand_t) and len(p_s3_t) > 0:
                v_s3_t[cand_t] = (1.0 - w_s3_t) * st_base[cand_t] + w_s3_t * (p_s3_t + 0.06)
            st_fin = gaussian_filter(v_s3_t, sigma=sig_t)
        else:
            st_fin = st_base
        if 'patella' in s3_scores and s3_scores['patella'] is not None:
            cand_p, p_s3_p = s3_scores['patella']
            v_s3_p = sp_base.copy()
            if np.any(cand_p) and len(p_s3_p) > 0:
                v_s3_p[cand_p] = (1.0 - w_s3_p) * sp_base[cand_p] + w_s3_p * (p_s3_p + 0.06)
            sp_fin = gaussian_filter(v_s3_p, sigma=sig_p)
        else:
            sp_fin = sp_base
    else:
        sf_fin, st_fin, sp_fin = sf_base, st_base, sp_base

    af_sum = max(float(np.sum(priors_roi['a_new_fc'])), 1e-5)
    pf_sum = max(float(np.sum(pf[pf >= 0.05])), 1e-5)
    pf_50 = float(np.sum((pf - b_pen_f * pfb) >= th_f))
    ind_f = 0.5 * ((pf_50 / pf_sum - 0.80) / 0.04) + 0.5 * ((pf_50 / af_sum - 0.98) / 0.12)
    dth_f = float(np.clip(-0.020 * ind_f, -0.06, 0.06))

    at_sum = max(float(np.sum(priors_roi['a_new_tc'])), 1e-5)
    pt_sum = max(float(np.sum(pt[pt >= 0.05])), 1e-5)
    pt_50 = float(np.sum((pt - b_pen_t * ptb) >= th_t))
    ind_t = 0.5 * ((pt_50 / pt_sum - 0.76) / 0.04) + 0.5 * ((pt_50 / at_sum - 0.96) / 0.12)
    dth_t = float(np.clip(-0.018 * ind_t, -0.06, 0.06))

    ap_sum = max(float(np.sum(priors_roi['a_new_pc'])), 1e-5)
    pp_sum = max(float(np.sum(pp[pp >= 0.05])), 1e-5)
    pp_50 = float(np.sum((pp - b_pen_p * ppb) >= th_p))
    ind_p = 0.5 * ((pp_50 / pp_sum - 0.76) / 0.04) + 0.5 * ((pp_50 / ap_sum - 0.96) / 0.14)
    dth_p = float(np.clip(-0.018 * ind_p, -0.06, 0.06))

    win_f = (sf_fin >= th_f + dth_f) & (sf_part >= st_part) & (sf_part >= sp_part)
    win_t = (st_fin >= th_t + dth_t) & (st_part > sf_part) & (st_part >= sp_part)
    win_p = (sp_fin >= th_p + dth_p) & (sp_part > sf_part) & (sp_part > st_part)

    pred_fc_roi = fast_keep_top_k(win_f, k=2, min_size=120)
    pred_tc_roi = fast_keep_top_k(win_t, k=2, min_size=60)
    pred_pc_roi = fast_keep_top_k(win_p, k=1, min_size=40)

    y_pred = np.zeros(img_shape, dtype=np.uint8)
    roi_view = y_pred[sl]
    roi_view[pred_fc_roi] = 2
    roi_view[pred_tc_roi] = 4
    roi_view[pred_pc_roi] = 6

    pv = np.zeros((*img_shape, 7), dtype=np.float32)
    pv[sl][..., 2] = np.clip(sf_fin, 0.0, 1.0)
    pv[sl][..., 4] = np.clip(st_fin, 0.0, 1.0)
    pv[sl][..., 6] = np.clip(sp_fin, 0.0, 1.0)
    pv[y_pred == 2, 2] = np.maximum(pv[y_pred == 2, 2], 0.75)
    pv[y_pred == 4, 4] = np.maximum(pv[y_pred == 4, 4], 0.75)
    pv[y_pred == 6, 6] = np.maximum(pv[y_pred == 6, 6], 0.75)
    s_fg = pv[..., 2] + pv[..., 4] + pv[..., 6]
    over = s_fg > 1.0
    if np.any(over):
        pv[over, 2] /= s_fg[over]
        pv[over, 4] /= s_fg[over]
        pv[over, 6] /= s_fg[over]
    pv[..., 0] = np.clip(1.0 - pv[..., 2] - pv[..., 4] - pv[..., 6], 0.0, 1.0)
    return y_pred, pv


def _blend_joint_atlas_priors(ap_base, ap_joint, alpha=0.015):
    if ap_joint is None or alpha <= 0.0:
        return ap_base
    w = float(alpha)
    out = dict(ap_base)
    shared_keys = (
        'pt_old_tc', 'pt_old_fc', 'pt_ml_tc', 'pt_ml_fc', 'pt_mly_tc',
        'pt_sep_tc', 'pt_sep_fc', 'pt_wide_tc',
        'p2d_ml_tc', 'edt_ml_2d', 'p2d_sep_tc', 'edt_sep_2d', 'p2d_wide_tc', 'edt_wide_2d',
        'pf_old_fc', 'pf_old_tc', 'pf_fml_fc', 'pf_fml_tc', 'pf_sep_fc', 'pf_sep_tc',
        'p3d_fml_fc', 'edf_fml_3d',
    )
    for k in shared_keys:
        if k in out and k in ap_joint and out[k].shape == ap_joint[k].shape:
            out[k] = ((1.0 - w) * out[k] + w * ap_joint[k]).astype(np.float32)
    return out


def _get_canonical_3bones_and_dts(img_c, sp_c, bone_masks, layout):
    from .rf_seg import _compute_articular_landmarks_and_stats
    fp_c = _vol_fingerprint_oai(img_c)
    bm_in = {
        k: to_canonical_oai((v > 0).astype(np.uint8), layout=layout)
        for k, v in bone_masks.items() if v is not None
    }
    if 'patella' not in bm_in:
        bm_in['patella'] = np.zeros_like(img_c, dtype=np.uint8)
    bm_fp = tuple(_vol_fingerprint_oai(bm_in[k]) for k in ('femur', 'tibia', 'patella'))

    bm_v2 = bm_in
    aux = _OAI_BONE_AUX_CACHE.get(fp_c)
    if (
        aux is not None
        and 'bm_v1' in aux
        and 'bm_raw' in aux
        and aux.get('bm_fp') == bm_fp
    ):
        bm_v1 = aux['bm_v1']
        bm_raw = aux['bm_raw']
    else:
        bm_v1 = bm_in
        bm_raw = bm_in

    df2 = fast_signed_distance_transform(bm_v2['femur'] > 0, sp_c)
    dt2 = fast_signed_distance_transform(bm_v2['tibia'] > 0, sp_c)
    dp2 = fast_signed_distance_transform(bm_v2['patella'] > 0, sp_c) if np.any(bm_v2['patella'] > 0) else np.full_like(df2, 50.0)
    if bm_v1 is bm_v2:
        df1, dt1, dp1 = df2, dt2, dp2
    else:
        df1 = fast_signed_distance_transform(bm_v1['femur'] > 0, sp_c)
        dt1 = fast_signed_distance_transform(bm_v1['tibia'] > 0, sp_c)
        dp1 = fast_signed_distance_transform(bm_v1['patella'] > 0, sp_c) if np.any(bm_v1['patella'] > 0) else np.full_like(df2, 50.0)
    if bm_raw is bm_v2:
        df0, dt0, dp0 = df2, dt2, dp2
    else:
        df0 = fast_signed_distance_transform(bm_raw['femur'] > 0, sp_c)
        dt0 = fast_signed_distance_transform(bm_raw['tibia'] > 0, sp_c)
        dp0 = fast_signed_distance_transform(bm_raw['patella'] > 0, sp_c) if np.any(bm_raw['patella'] > 0) else np.full_like(df2, 50.0)

    dts = (df2, dt2, dp2, df1, dt1, dp1, df0, dt0, dp0)
    f_st, _, f_cp = _compute_articular_landmarks_and_stats(bm_v2['femur'], df2, np.minimum(dt2, dp2), 45.0, sp_c)
    t_st, _, t_cp = _compute_articular_landmarks_and_stats(bm_v2['tibia'], dt2, df2, 25.0, sp_c)
    p_st, _, p_cp = _compute_articular_landmarks_and_stats(bm_v2['patella'], dp2, df2, 25.0, sp_c)
    ml_i = compute_ml_frames_3bone(bm_v2, df2, dt2, sp_c)
    lm = (f_st, f_cp, t_st, t_cp, p_st, p_cp, ml_i)
    return bm_v2, bm_v1, bm_raw, dts, lm, bm_fp


def predict_oai_cartilage_p1(img, bone_masks, spacing, model_dict, joint_atlas_priors=None, joint_alpha=0.015):
    """Run Pass-1 3-cartilage inference on an OAI volume."""
    layout = detect_oai_layout(img)
    img_c = to_canonical_oai(img, layout=layout).astype(np.float32)
    sp_c = (1.0, 1.0, 1.0) if layout != 'native' else tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))
    fp_c = _vol_fingerprint_oai(img_c)

    bm_v2, bm_v1, bm_raw, dts, lm, bm_fp = _get_canonical_3bones_and_dts(img_c, sp_c, bone_masks, layout)
    ap = model_dict['atlas_priors']
    ap_fp = float(ap['pf_fml_fc'].ravel()[0])
    p1_folds = model_dict['p1_folds']

    geos = {}
    pm1 = {}
    for c_name in ('femur', 'tibia', 'patella'):
        geos[c_name] = prepare_geometry_oai_v6(
            img_c, sp_c, bm_v2, bm_v1, bm_raw, c_name, ap, dts_pre=dts, landmarks_pre=lm
        )
        pm1[c_name] = _p1_folds_screened_oai(geos[c_name], p1_folds[c_name])

    while len(_OAI_CART_P1_CACHE) >= 8:
        _OAI_CART_P1_CACHE.pop(next(iter(_OAI_CART_P1_CACHE)), None)
    _OAI_CART_P1_CACHE[fp_c] = {
        'bm_fp': bm_fp,
        'ap_fp': ap_fp,
        'pm1': pm1,
        'geos': geos,
        'dts': dts,
        'lm': lm,
    }

    pv_c = np.zeros((*img_c.shape, 7), dtype=np.float32)
    pv_c[..., 1] = pm1['femur'][..., 1]
    pv_c[..., 2] = pm1['femur'][..., 2]
    pv_c[..., 3] = pm1['tibia'][..., 1]
    pv_c[..., 4] = pm1['tibia'][..., 2]
    pv_c[..., 5] = pm1['patella'][..., 1]
    pv_c[..., 6] = pm1['patella'][..., 2]
    if joint_atlas_priors is not None and joint_alpha > 0.0:
        u_sh = geos['femur']['shell'] | geos['tibia']['shell']
        inds_u = np.argwhere(u_sh)
        if len(inds_u) > 0:
            uz0, uy0, ux0 = np.maximum(inds_u.min(axis=0) - 2, 0)
            uz1, uy1, ux1 = np.minimum(inds_u.max(axis=0) + 3, img_c.shape)
            sl_p1 = (slice(uz0, uz1), slice(uy0, uy1), slice(ux0, ux1))
            ap_j = _blend_joint_atlas_priors(ap, joint_atlas_priors, alpha=1.0)
            pr_j = compute_roi_priors_oai(
                sl_p1, sp_c, dts[0], dts[1], dts[2], lm[0], lm[3], lm[4], lm[6], ap_j
            )
            w_j = float(joint_alpha)
            pv_c[sl_p1][..., 2] = (1.0 - w_j) * pv_c[sl_p1][..., 2] + w_j * pr_j['a_new_fc']
            pv_c[sl_p1][..., 4] = (1.0 - w_j) * pv_c[sl_p1][..., 4] + w_j * pr_j['a_new_tc']
    s_fg = np.sum(pv_c[..., 1:], axis=-1)
    over = s_fg > 1.0
    if np.any(over):
        for ch in range(1, 7):
            pv_c[over, ch] /= s_fg[over]
    pv_c[..., 0] = np.clip(1.0 - np.sum(pv_c[..., 1:], axis=-1), 0.0, 1.0)

    pred_c = np.zeros(img_c.shape, dtype=np.uint8)
    pf, pt, pp = pv_c[..., 2], pv_c[..., 4], pv_c[..., 6]
    pred_c[(pf >= 0.48) & (pf >= pt) & (pf >= pp)] = 2
    pred_c[(pt >= 0.48) & (pt > pf) & (pt >= pp)] = 4
    pred_c[(pp >= 0.48) & (pp > pf) & (pp > pt)] = 6
    return from_canonical_oai(pred_c, layout), from_canonical_oai(pv_c, layout)


def predict_oai_cartilage_p2(img, bone_masks, prob_map, spacing, model_dict, joint_atlas_priors=None, joint_alpha=0.015):
    """Run Pass-2 Auto-Context + Stage-3 Boundary & Subchondral Refiner 3-cartilage inference."""
    layout = detect_oai_layout(img)
    img_c = to_canonical_oai(img, layout=layout).astype(np.float32)
    sp_c = (1.0, 1.0, 1.0) if layout != 'native' else tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))
    fp_c = _vol_fingerprint_oai(img_c)

    m_v6 = model_dict['m_v6']
    ap = m_v6['atlas_priors']
    ap_fp = float(ap['pf_fml_fc'].ravel()[0])

    bm_in_fp = tuple(
        _vol_fingerprint_oai(
            to_canonical_oai((bone_masks[k] > 0).astype(np.uint8), layout=layout)
            if (k in bone_masks and bone_masks[k] is not None)
            else np.zeros_like(img_c, dtype=np.uint8)
        )
        for k in ('femur', 'tibia', 'patella')
    )
    cached = _OAI_CART_P1_CACHE.get(fp_c)
    if (
        cached is not None
        and 'pm1' in cached
        and 'geos' in cached
        and cached.get('bm_fp') == bm_in_fp
        and cached.get('ap_fp') == ap_fp
    ):
        pm1 = cached['pm1']
        geos = cached['geos']
        dts = cached['dts']
        lm = cached['lm']
    else:
        bm_v2, bm_v1, bm_raw, dts, lm, bm_fp = _get_canonical_3bones_and_dts(img_c, sp_c, bone_masks, layout)
        geos = {}
        pm1 = {}
        for c_name in ('femur', 'tibia', 'patella'):
            geos[c_name] = prepare_geometry_oai_v6(
                img_c, sp_c, bm_v2, bm_v1, bm_raw, c_name, ap, dts_pre=dts, landmarks_pre=lm
            )
            pm1[c_name] = _p1_folds_screened_oai(geos[c_name], m_v6['p1_folds'][c_name])
        while len(_OAI_CART_P1_CACHE) >= 8:
            _OAI_CART_P1_CACHE.pop(next(iter(_OAI_CART_P1_CACHE)), None)
        _OAI_CART_P1_CACHE[fp_c] = {'bm_fp': bm_fp, 'ap_fp': ap_fp, 'pm1': pm1, 'geos': geos, 'dts': dts, 'lm': lm}

    pm1_opp_f = combine_tp_for_femur(pm1['tibia'], pm1['patella'], dts[1], dts[2])
    pm2 = {
        'femur': _p2_screened_oai(geos['femur'], pm1['femur'], pm1_opp_f, m_v6['p2_femur_hgb'], m_v6['p2_femur_et']),
        'tibia': _p2_screened_oai(geos['tibia'], pm1['tibia'], pm1['femur'], m_v6['p2_tibia_hgb'], m_v6['p2_tibia_et']),
        'patella': _p2_screened_oai(geos['patella'], pm1['patella'], pm1['femur'], m_v6['p2_patella_hgb'], m_v6['p2_patella_et']),
    }

    u_sh = geos['femur']['shell'] | geos['tibia']['shell'] | geos['patella']['shell']
    inds_u = np.argwhere(u_sh)
    if len(inds_u) == 0:
        uz0, uy0, ux0, uz1, uy1, ux1 = 0, 0, 0, img_c.shape[0], img_c.shape[1], img_c.shape[2]
    else:
        uz0, uy0, ux0 = np.maximum(inds_u.min(axis=0) - 4, 0)
        uz1, uy1, ux1 = np.minimum(inds_u.max(axis=0) + 5, img_c.shape)
    sl = (slice(uz0, uz1), slice(uy0, uy1), slice(ux0, ux1))

    priors_roi = compute_roi_priors_oai(
        sl, sp_c, dts[0], dts[1], dts[2], lm[0], lm[3], lm[4], lm[6], ap
    )

    s3_models = model_dict.get('s3_models', {})
    s3_scores = {}
    df2, dt2, dp2, df1, dt1, dp1, df0, dt0, dp0 = [d[sl] for d in dts]
    for c_name in ('femur', 'tibia', 'patella'):
        if c_name not in s3_models or not s3_models[c_name]:
            continue
        if c_name == 'femur':
            ds2, ds1, ds0, do2 = df2, df1, df0, np.minimum(dt2, dp2)
            pm_opp_ct = np.maximum(pm2['tibia'][sl][..., 2], pm2['patella'][sl][..., 2])
            a_new, a_old, edt_e = priors_roi['a_new_fc'], priors_roi['a_old_fc'], priors_roi['edf_exp']
        elif c_name == 'tibia':
            ds2, ds1, ds0, do2 = dt2, dt1, dt0, df2
            pm_opp_ct = pm2['femur'][sl][..., 2]
            a_new, a_old, edt_e = priors_roi['a_new_tc'], priors_roi['a_old_tc'], priors_roi['edt_exp']
        else:
            ds2, ds1, ds0, do2 = dp2, dp1, dp0, df2
            pm_opp_ct = pm2['femur'][sl][..., 2]
            a_new, a_old, edt_e = priors_roi['a_new_pc'], priors_roi['a_old_pc'], priors_roi['edp_exp']

        cand, _, _, X_s3 = extract_s3_compartment_features(
            c_name, img_c[sl], sp_c, ds2, ds1, ds0, do2,
            df2, dt2, dp2, pm1[c_name][sl], pm2[c_name][sl],
            pm_opp_ct, a_new, a_old, edt_e
        )
        if np.any(cand) and len(X_s3) > 0:
            clfs_s3 = s3_models[c_name]
            if len(clfs_s3) >= 3 and getattr(clfs_s3[2], 'n_jobs', 1) not in range(1, 17):
                clfs_s3[2].n_jobs = min(max(os.cpu_count() or 8, 1), 16)
            with joblib.parallel_backend('threading'):
                p_s3 = (
                    0.42 * clfs_s3[0].predict_proba(X_s3)[:, 1]
                    + 0.38 * clfs_s3[1].predict_proba(X_s3)[:, 1]
                    + 0.20 * clfs_s3[2].predict_proba(X_s3)[:, 1]
                ).astype(np.float32)
            s3_scores[c_name] = (cand, p_s3)

    if joint_atlas_priors is not None and joint_alpha > 0.0:
        ap_j = _blend_joint_atlas_priors(ap, joint_atlas_priors, alpha=joint_alpha)
        priors_j = compute_roi_priors_oai(
            sl, sp_c, dts[0], dts[1], dts[2], lm[0], lm[3], lm[4], lm[6], ap_j
        )
        priors_roi = dict(priors_roi)
        priors_roi['a_new_fc'] = priors_j['a_new_fc']
        priors_roi['a_new_tc'] = priors_j['a_new_tc']

    y_pred_c, pv_c = resolve_3cartilage_masks(
        img_c.shape, sl, sp_c, df2, dt2, dp2,
        pm1['femur'][sl], pm1['tibia'][sl], pm1['patella'][sl],
        pm2['femur'][sl], pm2['tibia'][sl], pm2['patella'][sl],
        priors_roi, s3_scores=s3_scores, params=model_dict.get('params')
    )
    return from_canonical_oai(y_pred_c, layout), from_canonical_oai(pv_c, layout)


def _is_oai_domain(img, spacing):
    """Return True if volume is in the OAI 1.0mm isotropic 6-structure domain, False if SKI10 high-res domain."""
    if img is not None and img.shape[:3] == (112, 140, 140):
        return True
    if spacing is not None and len(spacing) >= 3:
        if min(float(spacing[1]), float(spacing[2])) >= 0.65:
            return True
        return False
    return False


def predict_unified_bone_p1(img, spacing, model_dict, ski10_clf_helper=None):
    """Unified SKI10+OAI Pass-1 Bone inference."""
    if _is_oai_domain(img, spacing):
        return predict_oai_bone_p1(img, spacing, model_dict['oai_p1'])
    from .bone_rf import BoneClassifier
    helper = BoneClassifier()
    helper.stage = 'v2_p1'
    helper.meta = model_dict['ski10_p1']
    helper._tierA_cached = model_dict['ski10_p1']['tierA']
    return helper._predict_v2_p1(img, spacing)


def predict_unified_bone_p2(img, prob_map, spacing, model_dict, ski10_clf_helper=None):
    """Unified SKI10+OAI Pass-2 Subchondral Surface Refiner Bone inference."""
    if _is_oai_domain(img, spacing):
        ref_v2_d = model_dict['oai_p2'].get('ref_v2') or model_dict['oai_p2']['models']
        p1_d = model_dict.get('oai_p1')
        if p1_d is None and ski10_clf_helper is not None and getattr(ski10_clf_helper, 'model_dir', None):
            p1_path = os.path.join(ski10_clf_helper.model_dir, 'bone_rf_p1.joblib')
            if os.path.exists(p1_path):
                p1_loaded = joblib.load(p1_path)
                if isinstance(p1_loaded, dict):
                    p1_d = p1_loaded.get('oai_p1', p1_loaded)
        return predict_oai_bone_p2(
            img, prob_map, spacing, ref_v2_d,
            p1_dict=p1_d,
            joint_bone_atlases=model_dict.get('joint_bone_atlases_208'),
            joint_bone_models=model_dict.get('joint_bone_models_208'),
        )
    from .bone_rf import BoneClassifier
    helper = BoneClassifier()
    helper.stage = 'v2_p2'
    helper.model_dir = getattr(ski10_clf_helper, 'model_dir', None) if ski10_clf_helper is not None else None
    ski10_meta = dict(model_dict['ski10_p2'])
    if 'joint_bone_models_208' in model_dict and 'ref_v2' in ski10_meta:
        jb_models = model_dict['joint_bone_models_208']
        jb_atls = model_dict.get('joint_bone_atlases_208', {})
        ref_v2_u = {k: dict(v) for k, v in ski10_meta['ref_v2'].items()}
        if 'tibia' in ref_v2_u and 'tibia' in jb_models:
            ref_v2_u['tibia']['joint_clf'] = jb_models['tibia']
            ref_v2_u['tibia']['joint_atl'] = jb_atls.get('tibia')
            ref_v2_u['tibia']['joint_alpha'] = 0.02
        ski10_meta['ref_v2'] = ref_v2_u
    helper.meta = ski10_meta
    helper._tierA_cached = ski10_meta.get('tierA')
    helper._ref_v2_cached = None
    return helper._predict_v2_refiner(img, spacing, prob_map=prob_map)


def predict_unified_cartilage_p1(img, bone_masks, spacing, model_dict, model_dir=None):
    """Unified SKI10+OAI Pass-1 Cartilage inference."""
    ap_joint = model_dict.get('joint_atlas_priors_208')
    if _is_oai_domain(img, spacing):
        return predict_oai_cartilage_p1(
            img, bone_masks, spacing, model_dict['oai_p1'],
            joint_atlas_priors=ap_joint, joint_alpha=0.015,
        )
    from .cartilage_v6 import predict_cartilage_v6_p1
    ski10_p1 = model_dict['ski10_p1']
    p1_folds = ski10_p1.get('p1_folds') or ski10_p1['m_v6']['p1_folds']
    ap_base = ski10_p1.get('atlas_priors') or ski10_p1['m_v6']['atlas_priors']
    bm_2 = {'femur': bone_masks['femur'], 'tibia': bone_masks['tibia']}
    return predict_cartilage_v6_p1(img, bm_2, spacing, p1_folds, ap_base, model_dir=model_dir)


def predict_unified_cartilage_p2(img, bone_masks, prob_map, spacing, model_dict, model_dir=None):
    """Unified SKI10+OAI Pass-2 Auto-Context + Stage-3 Subchondral Refiner Cartilage inference."""
    ap_joint = model_dict.get('joint_atlas_priors_208')
    if _is_oai_domain(img, spacing):
        return predict_oai_cartilage_p2(
            img, bone_masks, prob_map, spacing, model_dict['oai_p2'],
            joint_atlas_priors=ap_joint, joint_alpha=0.015,
        )
    from .cartilage_v6 import predict_cartilage_v6_p2
    ski10_p2 = model_dict['ski10_p2']
    bm_2 = {'femur': bone_masks['femur'], 'tibia': bone_masks['tibia']}
    return predict_cartilage_v6_p2(img, bm_2, spacing, ski10_p2, prob_map=prob_map, model_dir=model_dir)
