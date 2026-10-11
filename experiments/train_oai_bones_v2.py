"""Train 3-Bone (Femur, Tibia, Patella) Subchondral Surface Refiner & Atlas on OAI."""

import gc
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import joblib
from joblib import Parallel, delayed
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    maximum_filter1d,
    minimum_filter1d,
    zoom,
)
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_EXP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from kneeseg.bone_rf import _compute_shifted_diffs_3d, _postprocess_bone_mask
from kneeseg.features import fast_signed_distance_transform
from kneeseg.io import load_volume
from kneeseg.rf_seg import _compute_articular_landmarks_and_stats


def detect_oai_layout(arr):
    """Detect whether a volume is C-order OAI ('oai_c'), F-order OAI ('oai_f'), or native."""
    if arr.shape[:3] != (112, 140, 140):
        return 'native'
    sub = np.asarray(arr[:32, :64, :64], dtype=np.float32)
    if sub.ndim > 3:
        sub = sub[..., 0]
    mad0 = float(np.mean(np.abs(np.diff(sub, axis=0))))
    mad2 = float(np.mean(np.abs(np.diff(sub, axis=2))))
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
    """Extract Pass-0 strided features (without requiring df_s/dt_s)."""
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

    X_base = np.hstack([
        np.column_stack(cols),
        rx_neg, rx_pos, rz_neg, rz_pos, ry_neg, ry_pos,
        rsid,
    ]).astype(np.float32)
    return X_base


def augment_tierA_3bone_features(X_base, img_s_shape, sp_s, df_s, dt_s, dp_s, c_j, span_z, mask_s, atl_3b, rng_jit=None):
    """Append Joint-Anchored 3-bone spatial priors and jittered SDTs to X_base."""
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

    df_m = df_s[z_idx, y_idx, x_idx]
    dt_m = dt_s[z_idx, y_idx, x_idx]
    dp_m = dp_s[z_idx, y_idx, x_idx]
    if rng_jit is not None:
        N = len(df_m)
        heavy = rng_jit.rand(N) < 0.30
        jf = np.where(heavy, rng_jit.uniform(-9.0, 9.0, size=N), rng_jit.normal(0.0, 2.8, size=N)).astype(np.float32)
        jt = np.where(heavy, rng_jit.uniform(-9.0, 9.0, size=N), rng_jit.normal(0.0, 2.8, size=N)).astype(np.float32)
        jp = np.where(heavy, rng_jit.uniform(-9.0, 9.0, size=N), rng_jit.normal(0.0, 2.8, size=N)).astype(np.float32)
        df_feat = np.clip(df_m + jf, -8.0, 14.0)
        dt_feat = np.clip(dt_m + jt, -8.0, 14.0)
        dp_feat = np.clip(dp_m + jp, -8.0, 14.0)
    else:
        df_feat = np.clip(df_m, -8.0, 14.0)
        dt_feat = np.clip(dt_m, -8.0, 14.0)
        dp_feat = np.clip(dp_m, -8.0, 14.0)

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


def build_patella_fem_atlas(cases_data):
    """Build Distal-Femur-Anchored 3D Patella spatial prior (with LOO/4-fold support)."""
    h_tot1 = np.zeros((24, 32, 24), dtype=np.float32)
    h_pos1 = np.zeros((24, 32, 24), dtype=np.float32)
    h_tot2 = np.zeros((24, 32, 20), dtype=np.float32)
    h_pos2 = np.zeros((24, 32, 20), dtype=np.float32)
    per_case = []
    for item in cases_data:
        lbl, sp, df, dt, fb = item['lbl'], item['sp'], item['df'], item['dt'], item['fb']
        f_min, f_max, f_span = get_distal_femur_frame(fb, dt, sp)
        cand = (df >= 0.0) & (df <= 35.0)
        inds = np.argwhere(cand)
        coords_mm = inds.astype(np.float32) * np.array(sp, dtype=np.float32)
        rel = (coords_mm - f_min) / f_span
        in_box = (
            (rel[:, 0] >= -0.25) & (rel[:, 0] <= 1.05)
            & (rel[:, 1] >= -0.85) & (rel[:, 1] <= 1.35)
            & (rel[:, 2] >= -0.85) & (rel[:, 2] <= 0.15)
        )
        inds = inds[in_box]
        coords_mm = coords_mm[in_box]
        df_m = df[inds[:, 0], inds[:, 1], inds[:, 2]]
        b1, b2, _ = get_patella_fem_bins(coords_mm, df_m, f_min, f_span)
        yp = (lbl[inds[:, 0], inds[:, 1], inds[:, 2]] == 5)
        ct1 = np.zeros((24, 32, 24), dtype=np.float32)
        cp1 = np.zeros((24, 32, 24), dtype=np.float32)
        ct2 = np.zeros((24, 32, 20), dtype=np.float32)
        cp2 = np.zeros((24, 32, 20), dtype=np.float32)
        np.add.at(ct1, b1, 1.0)
        np.add.at(cp1, tuple(b[yp] for b in b1), 1.0)
        np.add.at(ct2, b2, 1.0)
        np.add.at(cp2, tuple(b[yp] for b in b2), 1.0)
        h_tot1 += ct1
        h_pos1 += cp1
        h_tot2 += ct2
        h_pos2 += cp2
        per_case.append((ct1, cp1, ct2, cp2))

    def _fin(t1, p1, t2, p2):
        return (
            (gaussian_filter(p1, 0.9) / (gaussian_filter(t1, 0.9) + 3.0)).astype(np.float32),
            (gaussian_filter(p2, 0.9) / (gaussian_filter(t2, 0.9) + 3.0)).astype(np.float32),
        )

    p_full = _fin(h_tot1, h_pos1, h_tot2, h_pos2)
    n = len(per_case)
    f_sz = n // 4
    p_folds = []
    for fk in range(4):
        s_i, e_i = fk * f_sz, (fk + 1) * f_sz if fk < 3 else n
        sub = per_case[:s_i] + per_case[e_i:]
        p_folds.append(_fin(
            sum(x[0] for x in sub), sum(x[1] for x in sub),
            sum(x[2] for x in sub), sum(x[3] for x in sub),
        ))
    return p_full, p_folds


def extract_patella_specialist_features(img, sp, fb, tb, pb_init, pat_atl, df=None, dt=None, dp_0=None, mask=None, rng_jit=None):
    """Extract full-resolution Femur-Anchored Patella features in the anterior knee."""
    if df is None:
        df = fast_signed_distance_transform(fb > 0, sp)
    if dt is None:
        dt = fast_signed_distance_transform(tb > 0, sp)
    if dp_0 is None:
        dp_0 = fast_signed_distance_transform(pb_init > 0, sp) if np.any(pb_init > 0) else np.full_like(df, 20.0)

    f_min, f_max, f_span = get_distal_femur_frame(fb, dt, sp)
    if mask is None:
        cand = (df >= -0.5) & (df <= 35.0)
        inds_c = np.argwhere(cand)
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
    Z, Y, X = img.shape
    coords_mm = inds.astype(np.float32) * np.array(sp, dtype=np.float32)
    df_m = df[z_idx, y_idx, x_idx]
    dt_m = dt[z_idx, y_idx, x_idx]
    dp_m = dp_0[z_idx, y_idx, x_idx]

    b1, b2, rel_f = get_patella_fem_bins(coords_mm, np.maximum(df_m, 0.0), f_min, f_span)
    p_atl1 = pat_atl[0][b1]
    p_atl2 = pat_atl[1][b2]
    p_atl_ens = 0.55 * p_atl1 + 0.45 * p_atl2

    # Crop around anterior ROI with margin for fast filtering
    margin = [int(np.ceil(26.0 / s)) for s in sp]
    z0, y0, x0 = np.maximum(inds.min(axis=0) - margin, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + margin + 1, img.shape)
    zc, yc, xc = z_idx - z0, y_idx - y0, x_idx - x0
    cz_max, cy_max, cx_max = z1 - z0, y1 - y0, x1 - x0

    p5, p50, p95 = np.percentile(img, [5, 50, 95])
    scale = max(float(p95 - p5), 1.0)
    img_c = ((img[z0:z1, y0:y1, x0:x1].astype(np.float32) - p50) / scale).astype(np.float32)
    df_c = df[z0:z1, y0:y1, x0:x1]

    # Femur normal vector pointing from femur toward patella
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

    # 1D cortical & cartilage enclosure in X, Y, Z at 12mm and 24mm
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
                mxp_x - v_m,  # bright patellofemoral cartilage peak in +X direction!
            ])

    # Ray along +X (posteriorly toward femoral trochlea) and -grad(df) (toward nearest femur surface)
    x_steps = [-8.0, -5.0, -2.5, 2.0, 4.0, 6.0, 8.5, 11.5, 15.0, 19.0]
    rx_v = np.column_stack([
        g1[zc, yc, np.clip(xc + int(round(d / sp[2])), 0, cx_max - 1)] for d in x_steps
    ])
    # Fractional ray from voxel (t=0) to femur surface (t=df_m) and 3mm inside femur (t=df_m+3)
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

    if rng_jit is not None:
        heavy = rng_jit.rand(N) < 0.35
        jp = np.where(heavy, rng_jit.uniform(-8.0, 8.0, size=N), rng_jit.normal(0.0, 2.5, size=N)).astype(np.float32)
        dp_feat = np.clip(dp_m + jp, -7.0, 14.0)
    else:
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
    zf, yf, xf = np.where(fb)
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


def build_bone_atlas_3bone(cases, bone_name, b_lbl):
    h_tot_s = np.zeros((24, 32, 24), dtype=np.float32)
    h_pos_s = np.zeros((24, 32, 24), dtype=np.float32)
    h_tot_m = np.zeros((28, 32, 24), dtype=np.float32)
    h_pos_m = np.zeros((28, 32, 24), dtype=np.float32)
    per_case = []
    for fname, img, lbl, sp, bm_pr, dts_pre in cases:
        d_self, d_other, _, _, max_other = _get_bone_d_self_other(bm_pr, bone_name, sp, dts_pre)
        med_low_z = detect_medial_is_low_z_3bone(bm_pr, sp)
        q50, scale, _ = get_robust_bone_frame_3bone(bm_pr[bone_name], d_other, max_other, sp, med_low_z)
        shell = np.abs(d_self) <= 4.2
        inds = np.argwhere(shell)
        coords_mm = inds.astype(np.float32) * np.array(sp, dtype=np.float32)
        bins_s, bins_m = coords_to_bone_atlas_bins_3bone(coords_mm, q50, scale, med_low_z)
        y_b = (lbl[shell] == b_lbl)
        ct_s = np.zeros((24, 32, 24), dtype=np.float32)
        cp_s = np.zeros((24, 32, 24), dtype=np.float32)
        ct_m = np.zeros((28, 32, 24), dtype=np.float32)
        cp_m = np.zeros((28, 32, 24), dtype=np.float32)
        np.add.at(ct_s, bins_s, 1.0)
        np.add.at(cp_s, tuple(b[y_b] for b in bins_s), 1.0)
        np.add.at(ct_m, bins_m, 1.0)
        np.add.at(cp_m, tuple(b[y_b] for b in bins_m), 1.0)
        h_tot_s += ct_s
        h_pos_s += cp_s
        h_tot_m += ct_m
        h_pos_m += cp_m
        per_case.append((ct_s, cp_s, ct_m, cp_m))

    p_full = (
        gaussian_filter(h_pos_s, sigma=0.8) / (gaussian_filter(h_tot_s, sigma=0.8) + 2.0),
        gaussian_filter(h_pos_m, sigma=0.8) / (gaussian_filter(h_tot_m, sigma=0.8) + 2.0),
    )
    p_loo = [
        (
            gaussian_filter(h_pos_s - ct_i[1], sigma=0.8) / (gaussian_filter(h_tot_s - ct_i[0], sigma=0.8) + 2.0),
            gaussian_filter(h_pos_m - ct_i[3], sigma=0.8) / (gaussian_filter(h_tot_m - ct_i[2], sigma=0.8) + 2.0),
        )
        for ct_i in per_case
    ]
    return p_full, p_loo


def extract_ray_feats_3bone_v2(img, sp, bm_pr, bone_name, bone_atlas_p, dts_pre=None, rng_jit=None):
    """Extract surface-normal ray features on the whole-bone boundary shell |d_self| <= 3.0 mm."""
    d_self, d_other, dt_raw, dp_raw, max_other = _get_bone_d_self_other(bm_pr, bone_name, sp, dts_pre)

    d_norm_ref = gaussian_filter(d_self, sigma=[2.2 / s for s in sp])
    nz, ny, nx = np.gradient(d_norm_ref, *sp)
    norm = np.sqrt(nz**2 + ny**2 + nx**2) + 1e-6
    nz, ny, nx = (nz / norm).astype(np.float32), (ny / norm).astype(np.float32), (nx / norm).astype(np.float32)

    shell = np.abs(d_self) <= 3.0
    if not np.any(shell):
        return shell, d_self, d_other, np.zeros((0, 108), dtype=np.float32), np.zeros((0, 158), dtype=np.float32)
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
    if rng_jit is not None:
        jit = rng_jit.uniform(-0.95, 0.95, size=N).astype(np.float32)
        w_jit = np.clip(1.0 - np.abs(d_self_m) / 2.4, 0.30, 1.0)
        d_self_feat = d_self_m + w_jit * jit
    else:
        d_self_feat = d_self_m

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
            p_atl_m, uz_ml, d_self_feat, d_other_m, dt_m, dp_m,
            idx_peak_rise, val_peak_rise, idx_cort_min, val_cort_min,
            z_bone_crop[zc, yc, xc],
        ]),
        ray_zb[:, 3:16], vy_lat, vyg_lat, vy_zb,
    ]).astype(np.float32)
    return shell, d_self, d_other, X_v1, X_v2


def postprocess_3bone_mask(mask, sp, c_j=None, span_z=65.0, bone_name='femur'):
    cc = _postprocess_bone_mask(mask)
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


def hist_3bone_one(item):
    img_s, y_s, sp_s, c_j, span_z = item['img_s'], item['y_s'], item['sp_s'], item['c_j'], item['span_z']
    f1, f2, fs = get_joint_bins_3bone(img_s.shape, sp_s, c_j, span_z)
    yr = y_s.ravel()
    fb = (yr == 1)
    tb = (yr == 2)
    pb = (yr == 3)
    return (
        np.bincount(f1, minlength=20480).reshape(20, 32, 32).astype(np.float32),
        np.bincount(f1[fb], minlength=20480).reshape(20, 32, 32).astype(np.float32),
        np.bincount(f1[tb], minlength=20480).reshape(20, 32, 32).astype(np.float32),
        np.bincount(f1[pb], minlength=20480).reshape(20, 32, 32).astype(np.float32),
        np.bincount(f2, minlength=30720).reshape(24, 40, 32).astype(np.float32),
        np.bincount(f2[fb], minlength=30720).reshape(24, 40, 32).astype(np.float32),
        np.bincount(f2[tb], minlength=30720).reshape(24, 40, 32).astype(np.float32),
        np.bincount(f2[pb], minlength=30720).reshape(24, 40, 32).astype(np.float32),
        np.bincount(fs, minlength=20480).reshape(16, 40, 32).astype(np.float32),
        np.bincount(fs[fb], minlength=20480).reshape(16, 40, 32).astype(np.float32),
        np.bincount(fs[tb], minlength=20480).reshape(16, 40, 32).astype(np.float32),
        np.bincount(fs[pb], minlength=20480).reshape(16, 40, 32).astype(np.float32),
    )


def fin_3bone_atl(items):
    s = [sum(p[k] for p in items) for k in range(12)]
    return (
        (gaussian_filter(s[1], 0.8) / (gaussian_filter(s[0], 0.8) + 4.0)).ravel().astype(np.float32),
        (gaussian_filter(s[2], 0.8) / (gaussian_filter(s[0], 0.8) + 4.0)).ravel().astype(np.float32),
        (gaussian_filter(s[3], 0.8) / (gaussian_filter(s[0], 0.8) + 4.0)).ravel().astype(np.float32),
        (gaussian_filter(s[5], 0.8) / (gaussian_filter(s[4], 0.8) + 4.0)).ravel().astype(np.float32),
        (gaussian_filter(s[6], 0.8) / (gaussian_filter(s[4], 0.8) + 4.0)).ravel().astype(np.float32),
        (gaussian_filter(s[7], 0.8) / (gaussian_filter(s[4], 0.8) + 4.0)).ravel().astype(np.float32),
        (gaussian_filter(s[9], 0.8) / (gaussian_filter(s[8], 0.8) + 4.0)).ravel().astype(np.float32),
        (gaussian_filter(s[10], 0.8) / (gaussian_filter(s[8], 0.8) + 4.0)).ravel().astype(np.float32),
        (gaussian_filter(s[11], 0.8) / (gaussian_filter(s[8], 0.8) + 4.0)).ravel().astype(np.float32),
    )


def predict_p0_case(img_s, sp_s, full_shape, sp_full, p0_models_list, atl_fov):
    f1 = get_fov_bins(img_s.shape)
    p_fov_sum = (atl_fov[0][f1] + atl_fov[1][f1] + atl_fov[2][f1]).reshape(img_s.shape)
    cand0 = p_fov_sum >= 0.002
    if not np.any(cand0):
        cand0 = np.ones(img_s.shape, dtype=bool)
    X0 = extract_base_strided_features(img_s, sp_s, cand0, atl_fov)
    p0 = sum(m.predict_proba(X0) for m in p0_models_list) / float(len(p0_models_list))
    v0 = np.zeros((*img_s.shape, 4), dtype=np.float32)
    v0[..., 0] = 1.0
    v0[cand0] = p0
    for c in range(4):
        v0[..., c] = gaussian_filter(v0[..., c], sigma=[1.0 / s for s in sp_s])

    fb_s = _postprocess_bone_mask((v0[..., 1] >= 0.42) & (v0[..., 1] >= v0[..., 2]) & (v0[..., 1] >= v0[..., 3]))
    tb_s = _postprocess_bone_mask((v0[..., 2] >= 0.42) & (v0[..., 2] > v0[..., 1]) & (v0[..., 2] >= v0[..., 3]))
    pb_s = _postprocess_bone_mask((v0[..., 3] >= 0.35) & (v0[..., 3] > v0[..., 1]) & (v0[..., 3] > v0[..., 2]))
    if not np.any(pb_s):
        pb_s = _postprocess_bone_mask(v0[..., 3] >= 0.22)
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


def predict_tierA_3bone_case(img, sp, img_s, sp_s, bm_0, df_s, dt_s, dp_s, c_j, span_z, tierA_models_list, atl_3b, pat_models_list=None, pat_atl=None):
    f1, f2, fs = get_joint_bins_3bone(img_s.shape, sp_s, c_j, span_z)
    p_atl = np.maximum(
        atl_3b[3][f2] + atl_3b[4][f2] + atl_3b[5][f2],
        atl_3b[6][fs] + atl_3b[7][fs] + atl_3b[8][fs],
    ).reshape(img_s.shape)
    cand = (df_s <= 10.0) | (dt_s <= 10.0) | (dp_s <= 10.0) | (p_atl >= 0.015)
    if not np.any(cand):
        cand = np.ones(img_s.shape, dtype=bool)
    X_base = extract_base_strided_features(img_s, sp_s, cand, atl_3b[:3])
    X_c = augment_tierA_3bone_features(X_base, img_s.shape, sp_s, df_s, dt_s, dp_s, c_j, span_z, cand, atl_3b, rng_jit=None)
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
            img, sp, fb_mask, tb_mask, pb_mask, pat_atl, df=df_full, dt=dt_full, dp_0=dp_init, mask=None, rng_jit=None
        )
        if np.any(cand_pat):
            p_pat_spec = sum(m.predict_proba(X_pat)[:, 1] for m in pat_models_list) / float(len(pat_models_list))
            vol_pat = 0.20 * pp_up
            vol_pat[cand_pat] = 0.78 * p_pat_spec + 0.22 * pp_up[cand_pat]
            vol_pat = gaussian_filter(vol_pat, sigma=[0.85 / s for s in sp])
            pb_spec = postprocess_3bone_mask((vol_pat >= 0.46) & (fb_mask == 0) & (tb_mask == 0), sp, c_j, span_z, 'patella').astype(np.uint8)
            if np.sum(pb_spec) >= 300:
                pb_mask = pb_spec
                pp_up = vol_pat.astype(np.float32)

    return {'femur': fb_mask, 'tibia': tb_mask, 'patella': pb_mask}, {'femur': pf_up, 'tibia': pt_up, 'patella': pp_up, 'c_j': c_j, 'span_z': span_z}


def refine_one_bone_3bone_v2(img, sp, bm_a, bone_name, m_d, dts_pre=None, return_v1=False):
    if np.sum(bm_a['femur'] > 0) < 20 or np.sum(bm_a['tibia'] > 0) < 20 or np.sum(bm_a.get(bone_name, 0) > 0) < 20:
        m0 = (bm_a.get(bone_name, np.zeros_like(img, dtype=np.uint8)) > 0).astype(np.uint8)
        return (m0, m0) if return_v1 else m0
    shell, d_self, d_other, X1, X2 = extract_ray_feats_3bone_v2(
        img, sp, bm_a, bone_name, m_d['p_full'], dts_pre=dts_pre, rng_jit=None
    )
    if not np.any(shell):
        m0 = (bm_a[bone_name] > 0).astype(np.uint8)
        return (m0, m0) if return_v1 else m0

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

    p_orig = 1.0 / (1.0 + np.exp(np.clip(d_self / 0.45, -20.0, 20.0)))
    sig_s = [0.75 / sp[0], 0.55 / sp[1], 0.65 / sp[2]]
    volb = p_orig.copy()
    volb[shell] = p_bl
    m_v2 = postprocess_3bone_mask(gaussian_filter(volb, sigma=sig_s) >= 0.50, sp, bone_name=bone_name).astype(np.uint8)
    if not return_v1:
        return m_v2
    volb1 = p_orig.copy()
    volb1[shell] = p1
    m_v1 = postprocess_3bone_mask(gaussian_filter(volb1, sigma=sig_s) >= 0.50, sp, bone_name=bone_name).astype(np.uint8)
    return m_v2, m_v1


if __name__ == '__main__':
    t_start = time.time()
    with open(os.path.join(_ROOT_DIR, 'kneeseg', 'data', 'oai_split.json')) as f:
        split = json.load(f)

    data_dir = os.environ.get('OAI_DATA_DIR', '/path/to/OAI_refactored/Data')
    lbl_dir = os.environ.get('OAI_LABEL_DIR', data_dir + '_labels')
    cache_dir = os.path.join(_EXP_DIR, 'cache_oai')
    model_dir = os.path.join(_EXP_DIR, 'models_oai')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    def load_oai_case(fname):
        img, sp = load_volume(os.path.join(data_dir, fname), return_spacing=True)
        lbl = load_volume(os.path.join(lbl_dir, fname.replace('image-', 'labels-')), return_spacing=False).astype(np.uint8)
        layout = detect_oai_layout(img)
        img_c = to_canonical_oai(img, layout=layout).astype(np.float32)
        lbl_c = to_canonical_oai(lbl, layout=layout).astype(np.uint8)
        sp_c = (1.0, 1.0, 1.0)
        img_s = img_c[::1, ::2, ::2].astype(np.float32)
        lbl_s = lbl_c[::1, ::2, ::2]
        y_s = np.zeros_like(lbl_s, dtype=np.uint8)
        y_s[lbl_s == 1] = 1
        y_s[lbl_s == 3] = 2
        y_s[lbl_s == 5] = 3
        sp_s = (1.0, 2.0, 2.0)
        _, _, c_j_gt, span_z_gt = compute_joint_frame_strided(y_s == 1, y_s == 2, sp_s)
        return {
            'fname': fname,
            'img': img_c,
            'lbl': lbl_c,
            'sp': sp_c,
            'img_s': img_s,
            'lbl_s': lbl_s,
            'y_s': y_s,
            'sp_s': sp_s,
            'c_j': c_j_gt,
            'span_z': span_z_gt,
        }

    tr_items = Parallel(n_jobs=16, prefer='threads')(delayed(load_oai_case)(fn) for fn in split['train'])
    ev_items = Parallel(n_jobs=16, prefer='threads')(delayed(load_oai_case)(fn) for fn in split['eval'])
    n_tr = len(tr_items)
    f_sz = n_tr // 4  # 32 scans per fold
    print(f"Loaded {n_tr} OAI train + {len(ev_items)} OAI eval scans in {time.time() - t_start:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # STEP 1: Pass-0 Strided Classifier (No prior mask needed!)
    # ------------------------------------------------------------------
    t0 = time.time()
    per_case_h0 = Parallel(n_jobs=16, prefer='threads')(delayed(hist_3bone_one)(it) for it in tr_items)
    atl_0_full = fin_3bone_atl(per_case_h0)
    atl_0_folds = [
        fin_3bone_atl(per_case_h0[:fk * f_sz] + per_case_h0[(fk + 1) * f_sz:])
        for fk in range(4)
    ]

    def sample_p0_tr(i):
        it = tr_items[i]
        img_s, y_s, lbl_s, sp_s = it['img_s'], it['y_s'], it['lbl_s'], it['sp_s']
        atl = atl_0_folds[i // f_sz][:3]
        dts_gt = [fast_signed_distance_transform(y_s == c, sp_s) for c in [1, 2, 3]]
        min_abs_d = np.minimum(np.minimum(np.abs(dts_gt[0]), np.abs(dts_gt[1])), np.abs(dts_gt[2]))
        p5, p50, p95 = np.percentile(img_s, [5, 50, 95])
        inorm = (img_s - p50) / max(float(p95 - p5), 1.0)
        rng = np.random.RandomState(200 + i)
        cart_mask = (lbl_s == 2) | (lbl_s == 4) | (lbl_s == 6)
        arrs = []
        for pool, n_take in [
            (np.argwhere(min_abs_d <= 5.0), 3200),
            (np.argwhere(y_s == 3), 1200),
            (np.argwhere( np.abs(dts_gt[2]) <= 6.0 ), 1200),
            (np.argwhere(cart_mask), 1000),
            (np.argwhere((y_s > 0) & (inorm > -0.02)), 1400),
            (np.argwhere((y_s > 0) & (min_abs_d > 5.0)), 1400),
            (np.argwhere((y_s == 0) & (min_abs_d > 3.0) & (inorm < -0.02)), 1600),
            (np.argwhere((y_s == 0) & (min_abs_d > 5.0)), 1200),
        ]:
            if len(pool) > 0:
                arrs.append(pool[rng.choice(len(pool), min(len(pool), n_take), replace=False)])
        smask = np.zeros_like(y_s, dtype=bool)
        for a in arrs:
            smask[a[:, 0], a[:, 1], a[:, 2]] = True
        X0 = extract_base_strided_features(img_s, sp_s, smask, atl)
        return X0, y_s[smask]

    tr_p0_s = Parallel(n_jobs=16, prefer='threads')(delayed(sample_p0_tr)(i) for i in range(n_tr))
    X_p0 = np.vstack([s[0] for s in tr_p0_s])
    y_p0 = np.concatenate([s[1] for s in tr_p0_s])
    b_p0 = [0]
    for s in tr_p0_s:
        b_p0.append(b_p0[-1] + len(s[1]))
    del tr_p0_s
    gc.collect()
    print(f"Sampled Pass-0 matrix {X_p0.shape} in {time.time() - t0:.1f}s", flush=True)

    hgb_p0_1 = HistGradientBoostingClassifier(
        max_iter=190, max_leaf_nodes=63, min_samples_leaf=25,
        learning_rate=0.055, max_features=0.70, l2_regularization=1.0,
        early_stopping=False, random_state=42,
    ).fit(X_p0, y_p0)
    hgb_p0_2 = HistGradientBoostingClassifier(
        max_iter=190, max_leaf_nodes=79, min_samples_leaf=30,
        learning_rate=0.055, max_features=0.75, l2_regularization=1.5,
        early_stopping=False, random_state=43,
    ).fit(X_p0, y_p0)

    oof_p0_folds = []
    for fk in range(4):
        s_i, e_i = b_p0[fk * f_sz], b_p0[(fk + 1) * f_sz]
        X_f = np.vstack([X_p0[:s_i], X_p0[e_i:]])
        y_f = np.concatenate([y_p0[:s_i], y_p0[e_i:]])
        clf_f = HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=63, min_samples_leaf=25,
            learning_rate=0.06, max_features=0.70, l2_regularization=1.0,
            early_stopping=False, random_state=140 + fk,
        ).fit(X_f, y_f)
        oof_p0_folds.append(clf_f)
    del X_p0, y_p0
    gc.collect()

    def run_p0_tr(i):
        it = tr_items[i]
        fk = i // f_sz
        return predict_p0_case(it['img_s'], it['sp_s'], it['img'].shape, it['sp'], [oof_p0_folds[fk]], atl_0_folds[fk][:3])

    def run_p0_ev(i):
        it = ev_items[i]
        return predict_p0_case(it['img_s'], it['sp_s'], it['img'].shape, it['sp'], [hgb_p0_1, hgb_p0_2], atl_0_full[:3])

    tr_p0_res = Parallel(n_jobs=16, prefer='threads')(delayed(run_p0_tr)(i) for i in range(n_tr))
    ev_p0_res = Parallel(n_jobs=16, prefer='threads')(delayed(run_p0_ev)(i) for i in range(len(ev_items)))
    for i, (bm_0, df_s, dt_s, dp_s, c_j, span_z) in enumerate(tr_p0_res):
        tr_items[i].update({'bm_0': bm_0, 'df_s': df_s, 'dt_s': dt_s, 'dp_s': dp_s, 'c_j': c_j, 'span_z': span_z})
    for i, (bm_0, df_s, dt_s, dp_s, c_j, span_z) in enumerate(ev_p0_res):
        ev_items[i].update({'bm_0': bm_0, 'df_s': df_s, 'dt_s': dt_s, 'dp_s': dp_s, 'c_j': c_j, 'span_z': span_z})

    for b_lbl, b_name in [(1, 'femur'), (3, 'tibia'), (5, 'patella')]:
        d_oof = [float(2.0 * np.sum((it['lbl'] == b_lbl) & (it['bm_0'][b_name] > 0)) / (np.sum(it['lbl'] == b_lbl) + np.sum(it['bm_0'][b_name] > 0))) for it in tr_items]
        print(f"  [Pass-0 OOF 128-Train] {b_name:8s}: DSC = {np.mean(d_oof):.4f} ± {np.std(d_oof):.4f} (min={min(d_oof):.4f})", flush=True)

    # ------------------------------------------------------------------
    # STEP 2: Pass-1 Tier-A Joint-Anchored 3-Bone Classifier + Patella Specialist
    # ------------------------------------------------------------------
    t_a = time.time()
    per_case_hA = Parallel(n_jobs=16, prefer='threads')(delayed(hist_3bone_one)(it) for it in tr_items)
    atl_A_full = fin_3bone_atl(per_case_hA)
    atl_A_folds = [
        fin_3bone_atl(per_case_hA[:fk * f_sz] + per_case_hA[(fk + 1) * f_sz:])
        for fk in range(4)
    ]

    def sample_tierA_tr(i):
        it = tr_items[i]
        img_s, y_s, lbl_s, sp_s = it['img_s'], it['y_s'], it['lbl_s'], it['sp_s']
        df_s, dt_s, dp_s, c_j, span_z = it['df_s'], it['dt_s'], it['dp_s'], it['c_j'], it['span_z']
        atl = atl_A_folds[i // f_sz]
        dts_gt = [fast_signed_distance_transform(y_s == c, sp_s) for c in [1, 2, 3]]
        min_abs_d = np.minimum(np.minimum(np.abs(dts_gt[0]), np.abs(dts_gt[1])), np.abs(dts_gt[2]))
        p5, p50, p95 = np.percentile(img_s, [5, 50, 95])
        inorm = (img_s - p50) / max(float(p95 - p5), 1.0)
        rng = np.random.RandomState(300 + i)
        cart_mask = (lbl_s == 2) | (lbl_s == 4) | (lbl_s == 6)
        sub_mask = binary_dilation(cart_mask, iterations=2) & (y_s > 0)
        raw_disagree = ((df_s <= 0) != (y_s == 1)) | ((dt_s <= 0) != (y_s == 2)) | ((dp_s <= 0) != (y_s == 3))
        arrs = []
        for pool, n_take in [
            (np.argwhere(min_abs_d <= 5.0), 3000),
            (np.argwhere(raw_disagree), 2000),
            (np.argwhere(np.abs(dts_gt[2]) <= 5.0), 1400),
            (np.argwhere(cart_mask), 1100),
            (np.argwhere(sub_mask), 1100),
            (np.argwhere((y_s > 0) & (inorm > -0.02)), 1400),
            (np.argwhere((y_s > 0) & (min_abs_d > 5.0)), 1200),
            (np.argwhere((y_s == 0) & (min_abs_d > 3.0) & (inorm < -0.02)), 1600),
            (np.argwhere((y_s == 0) & (min_abs_d > 5.0)), 1200),
        ]:
            if len(pool) > 0:
                arrs.append(pool[rng.choice(len(pool), min(len(pool), n_take), replace=False)])
        smask = np.zeros_like(y_s, dtype=bool)
        for a in arrs:
            smask[a[:, 0], a[:, 1], a[:, 2]] = True
        X_base = extract_base_strided_features(img_s, sp_s, smask, atl[:3])
        X_a = augment_tierA_3bone_features(X_base, img_s.shape, sp_s, df_s, dt_s, dp_s, c_j, span_z, smask, atl, rng_jit=rng)
        return X_a, y_s[smask]

    tr_A_s = Parallel(n_jobs=16, prefer='threads')(delayed(sample_tierA_tr)(i) for i in range(n_tr))
    X_tierA = np.vstack([s[0] for s in tr_A_s])
    y_tierA = np.concatenate([s[1] for s in tr_A_s])
    b_A = [0]
    for s in tr_A_s:
        b_A.append(b_A[-1] + len(s[1]))
    del tr_A_s
    gc.collect()
    print(f"Sampled Tier-A matrix {X_tierA.shape} in {time.time() - t_a:.1f}s", flush=True)

    hgb_tierA_1 = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=63, min_samples_leaf=25,
        learning_rate=0.055, max_features=0.70, l2_regularization=1.0,
        early_stopping=False, random_state=42,
    ).fit(X_tierA, y_tierA)
    hgb_tierA_2 = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=79, min_samples_leaf=30,
        learning_rate=0.055, max_features=0.75, l2_regularization=1.5,
        early_stopping=False, random_state=43,
    ).fit(X_tierA, y_tierA)

    oof_tierA_folds = []
    for fk in range(4):
        s_i, e_i = b_A[fk * f_sz], b_A[(fk + 1) * f_sz]
        X_f = np.vstack([X_tierA[:s_i], X_tierA[e_i:]])
        y_f = np.concatenate([y_tierA[:s_i], y_tierA[e_i:]])
        clf_f = HistGradientBoostingClassifier(
            max_iter=190, max_leaf_nodes=63, min_samples_leaf=25,
            learning_rate=0.055, max_features=0.70, l2_regularization=1.0,
            early_stopping=False, random_state=240 + fk,
        ).fit(X_f, y_f)
        oof_tierA_folds.append(clf_f)
    del X_tierA, y_tierA
    gc.collect()

    # Initial Tier-A inference (before Patella full-res specialist) to get fb, tb, pb_init
    def run_tierA_init_tr(i):
        it = tr_items[i]
        fk = i // f_sz
        bm_a, _ = predict_tierA_3bone_case(
            it['img'], it['sp'], it['img_s'], it['sp_s'], it['bm_0'],
            it['df_s'], it['dt_s'], it['dp_s'], it['c_j'], it['span_z'],
            [oof_tierA_folds[fk]], atl_A_folds[fk],
        )
        df = fast_signed_distance_transform(bm_a['femur'] > 0, it['sp'])
        dt = fast_signed_distance_transform(bm_a['tibia'] > 0, it['sp'])
        dp = fast_signed_distance_transform(bm_a['patella'] > 0, it['sp']) if np.any(bm_a['patella'] > 0) else np.full_like(df, 20.0)
        return bm_a, df, dt, dp

    tr_a_init = Parallel(n_jobs=16, prefer='threads')(delayed(run_tierA_init_tr)(i) for i in range(n_tr))
    for i, (bm_a, df, dt, dp) in enumerate(tr_a_init):
        tr_items[i].update({'bm_a_init': bm_a, 'df': df, 'dt': dt, 'dp_0': dp, 'fb': bm_a['femur'], 'tb': bm_a['tibia']})

    # Build Distal-Femur-Anchored Patella Atlas & train Full-Res Patella Specialist
    t_pat = time.time()
    pat_atl_full, pat_atl_folds = build_patella_fem_atlas(tr_items)

    def sample_pat_spec_tr(i):
        it = tr_items[i]
        fk = i // f_sz
        lbl, sp = it['lbl'], it['sp']
        dp_gt = fast_signed_distance_transform(lbl == 5, sp)
        dc_gt = fast_signed_distance_transform(lbl == 6, sp)
        dp_0 = it['dp_0']
        rng = np.random.RandomState(400 + i)

        f_min, f_max, f_span = get_distal_femur_frame(it['fb'], it['dt'], sp)
        cand = (it['df'] >= -0.5) & (it['df'] <= 35.0)
        inds_c = np.argwhere(cand)
        coords_c = inds_c.astype(np.float32) * np.array(sp, dtype=np.float32)
        rel_c = (coords_c - f_min) / f_span
        in_box = (
            (rel_c[:, 0] >= -0.25) & (rel_c[:, 0] <= 1.05)
            & (rel_c[:, 1] >= -0.85) & (rel_c[:, 1] <= 1.35)
            & (rel_c[:, 2] >= -0.85) & (rel_c[:, 2] <= 0.15)
        )
        cand_mask = np.zeros(lbl.shape, dtype=bool)
        kept = inds_c[in_box | (dp_0[inds_c[:, 0], inds_c[:, 1], inds_c[:, 2]] <= 10.0)]
        cand_mask[kept[:, 0], kept[:, 1], kept[:, 2]] = True

        y_p = (lbl == 5)
        disagree = ((dp_0 <= 0) != y_p) & cand_mask
        arrs = []
        for pool, n_take in [
            (np.argwhere(cand_mask & (np.abs(dp_gt) <= 3.5)), 3200),
            (np.argwhere(disagree), 1800),
            (np.argwhere(cand_mask & (dc_gt <= 2.5)), 1200),
            (np.argwhere(cand_mask & y_p & (dp_gt < -3.5)), 1000),
            (np.argwhere(cand_mask & (~y_p) & (dp_gt > 3.5) & (dp_gt <= 10.0)), 1400),
            (np.argwhere(cand_mask & (~y_p) & (dp_gt > 10.0)), 1200),
        ]:
            if len(pool) > 0:
                arrs.append(pool[rng.choice(len(pool), min(len(pool), n_take), replace=False)])
        smask = np.zeros(lbl.shape, dtype=bool)
        for a in arrs:
            smask[a[:, 0], a[:, 1], a[:, 2]] = True
        _, X_p = extract_patella_specialist_features(
            it['img'], sp, it['fb'], it['tb'], it['bm_a_init']['patella'],
            pat_atl_folds[fk], df=it['df'], dt=it['dt'], dp_0=dp_0, mask=smask, rng_jit=rng
        )
        return X_p, y_p[smask].astype(np.uint8)

    tr_pat_s = Parallel(n_jobs=16, prefer='threads')(delayed(sample_pat_spec_tr)(i) for i in range(n_tr))
    X_pat_tr = np.vstack([s[0] for s in tr_pat_s])
    y_pat_tr = np.concatenate([s[1] for s in tr_pat_s])
    b_pat = [0]
    for s in tr_pat_s:
        b_pat.append(b_pat[-1] + len(s[1]))
    del tr_pat_s
    gc.collect()
    print(f"Sampled Full-Res Patella Specialist matrix {X_pat_tr.shape} in {time.time() - t_pat:.1f}s", flush=True)

    hgb_pat_1 = HistGradientBoostingClassifier(
        max_iter=220, max_leaf_nodes=63, min_samples_leaf=25,
        learning_rate=0.05, max_features=0.70, l2_regularization=1.0,
        early_stopping=False, random_state=42,
    ).fit(X_pat_tr, y_pat_tr)
    hgb_pat_2 = HistGradientBoostingClassifier(
        max_iter=220, max_leaf_nodes=79, min_samples_leaf=30,
        learning_rate=0.05, max_features=0.75, l2_regularization=1.5,
        early_stopping=False, random_state=43,
    ).fit(X_pat_tr, y_pat_tr)
    et_pat = ExtraTreesClassifier(
        n_estimators=100, max_depth=24, min_samples_leaf=5, n_jobs=-1, random_state=44
    ).fit(X_pat_tr, y_pat_tr)

    oof_pat_folds = []
    for fk in range(4):
        s_i, e_i = b_pat[fk * f_sz], b_pat[(fk + 1) * f_sz]
        X_f = np.vstack([X_pat_tr[:s_i], X_pat_tr[e_i:]])
        y_f = np.concatenate([y_pat_tr[:s_i], y_pat_tr[e_i:]])
        clf_f = HistGradientBoostingClassifier(
            max_iter=210, max_leaf_nodes=63, min_samples_leaf=25,
            learning_rate=0.05, max_features=0.70, l2_regularization=1.0,
            early_stopping=False, random_state=340 + fk,
        ).fit(X_f, y_f)
        oof_pat_folds.append(clf_f)
    del X_pat_tr, y_pat_tr
    gc.collect()

    def run_tierA_final_tr(i):
        it = tr_items[i]
        fk = i // f_sz
        bm_a, _ = predict_tierA_3bone_case(
            it['img'], it['sp'], it['img_s'], it['sp_s'], it['bm_0'],
            it['df_s'], it['dt_s'], it['dp_s'], it['c_j'], it['span_z'],
            [oof_tierA_folds[fk]], atl_A_folds[fk],
            pat_models_list=[oof_pat_folds[fk]], pat_atl=pat_atl_folds[fk],
        )
        return it['fname'], bm_a

    def run_tierA_final_ev(i):
        it = ev_items[i]
        bm_a, _ = predict_tierA_3bone_case(
            it['img'], it['sp'], it['img_s'], it['sp_s'], it['bm_0'],
            it['df_s'], it['dt_s'], it['dp_s'], it['c_j'], it['span_z'],
            [hgb_tierA_1, hgb_tierA_2], atl_A_full,
            pat_models_list=[hgb_pat_1, hgb_pat_2, et_pat], pat_atl=pat_atl_full,
        )
        return it['fname'], bm_a

    tr_a_fin = Parallel(n_jobs=16, prefer='threads')(delayed(run_tierA_final_tr)(i) for i in range(n_tr))
    ev_a_fin = Parallel(n_jobs=16, prefer='threads')(delayed(run_tierA_final_ev)(i) for i in range(len(ev_items)))
    train128_pred_bones = dict(tr_a_fin)
    eval31_pred_bones = dict(ev_a_fin)
    joblib.dump(train128_pred_bones, os.path.join(cache_dir, 'train128_pred_bones.joblib'), compress=3)
    joblib.dump(eval31_pred_bones, os.path.join(cache_dir, 'eval31_pred_bones.joblib'), compress=3)

    for b_lbl, b_name in [(1, 'femur'), (3, 'tibia'), (5, 'patella')]:
        d_oof = [float(2.0 * np.sum((it['lbl'] == b_lbl) & (train128_pred_bones[it['fname']][b_name] > 0)) / (np.sum(it['lbl'] == b_lbl) + np.sum(train128_pred_bones[it['fname']][b_name] > 0))) for it in tr_items]
        d_ev = [float(2.0 * np.sum((it['lbl'] == b_lbl) & (eval31_pred_bones[it['fname']][b_name] > 0)) / (np.sum(it['lbl'] == b_lbl) + np.sum(eval31_pred_bones[it['fname']][b_name] > 0))) for it in ev_items]
        print(f"  [Tier-A OOF 128-Train] {b_name:8s}: DSC = {np.mean(d_oof):.4f} ± {np.std(d_oof):.4f} (min={min(d_oof):.4f}) | [31-Eval]: DSC = {np.mean(d_ev):.4f} ± {np.std(d_ev):.4f}", flush=True)

    # Save bone_rf_p1.joblib
    joblib.dump({
        'format_version': 'oai_3bone_v2',
        'p0_models': [hgb_p0_1, hgb_p0_2],
        'atl_0_full': atl_0_full[:3],
        'tierA_models': [hgb_tierA_1, hgb_tierA_2],
        'atl_A_full': atl_A_full,
        'pat_models': [hgb_pat_1, hgb_pat_2, et_pat],
        'pat_atl_full': pat_atl_full,
    }, os.path.join(model_dir, 'bone_rf_p1.joblib'), compress=3)
    print(f"Saved bone_rf_p1.joblib in {time.time() - t_start:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # STEP 3: Pass-2 Tier-B Full-Res 3-Bone Boundary & Subchondral Refiner
    # ------------------------------------------------------------------
    def prep_case_dts(it, bm_dict):
        bm = bm_dict[it['fname']]
        sp = it['sp']
        df = fast_signed_distance_transform(bm['femur'] > 0, sp)
        dt = fast_signed_distance_transform(bm['tibia'] > 0, sp)
        dp = fast_signed_distance_transform(bm['patella'] > 0, sp) if np.any(bm['patella'] > 0) else np.full_like(df, 50.0)
        return (it['fname'], it['img'], it['lbl'], sp, bm, (df, dt, dp))

    tr_cases = Parallel(n_jobs=16, prefer='threads')(delayed(prep_case_dts)(it, train128_pred_bones) for it in tr_items)
    ev_cases = Parallel(n_jobs=16, prefer='threads')(delayed(prep_case_dts)(it, eval31_pred_bones) for it in ev_items)
    del tr_items, ev_items
    gc.collect()

    ref_models_all = {}
    ref_oof_all = {}

    for bone_name, b_lbl, c_lbl in [('femur', 1, 2), ('tibia', 3, 4), ('patella', 5, 6)]:
        t_b = time.time()
        max_other = 45.0 if bone_name == 'femur' else 25.0
        p_full, p_loo = build_bone_atlas_3bone(tr_cases, bone_name, b_lbl)

        def sample_tr_ref(i):
            fname, img, lbl, sp, bm_pr, dts_pre = tr_cases[i]
            rng = np.random.RandomState(500 + i)
            shell, d_self, d_other, X1, X2 = extract_ray_feats_3bone_v2(
                img, sp, bm_pr, bone_name, p_loo[i], dts_pre=dts_pre, rng_jit=rng
            )
            d_c = fast_signed_distance_transform(lbl == c_lbl, sp)[shell]
            d_s = d_self[shell]
            d_o = d_other[shell]
            y_all = (lbl[shell] == b_lbl).astype(np.uint8)

            # 1) Whole-shell sample (both articular and non-articular cortex)
            pos_w = np.where((y_all == 1) & (np.abs(d_s) <= 2.4))[0]
            neg_w = np.where((y_all == 0) & (np.abs(d_s) <= 2.4))[0]
            pos_wo = np.where((y_all == 1) & (np.abs(d_s) > 2.4))[0]
            neg_wo = np.where((y_all == 0) & (np.abs(d_s) > 2.4))[0]
            w_list = []
            for pool, n_take in [(pos_w, 2600), (neg_w, 2600), (pos_wo, 900), (neg_wo, 900)]:
                if len(pool) > 0:
                    w_list.append(rng.choice(pool, min(len(pool), n_take), replace=False))
            idx_w = np.concatenate(w_list)

            # 2) Subchondral articular specialist sample
            in_art = (d_o <= max_other) & (d_o >= -3.0)
            pos_int = np.where((y_all == 1) & in_art & (d_c <= 3.0) & (np.abs(d_s) <= 2.0))[0]
            neg_int = np.where((y_all == 0) & in_art & (d_c <= 3.0) & (np.abs(d_s) <= 2.0))[0]
            pos_out = np.where((y_all == 1) & in_art & ((d_c > 3.0) | (np.abs(d_s) > 2.0)))[0]
            neg_out = np.where((y_all == 0) & in_art & ((d_c > 3.0) | (np.abs(d_s) > 2.0)))[0]
            s_list = []
            for pool, n_take in [(pos_int, 2400), (neg_int, 2400), (pos_out, 1200), (neg_out, 1200)]:
                if len(pool) > 0:
                    s_list.append(rng.choice(pool, min(len(pool), n_take), replace=False))
            idx_sub = np.concatenate(s_list) if len(s_list) > 0 else idx_w
            return X1[idx_w], y_all[idx_w], X2[idx_w], y_all[idx_w], X2[idx_sub], y_all[idx_sub]

        tr_s = Parallel(n_jobs=16, prefer='threads')(delayed(sample_tr_ref)(i) for i in range(len(tr_cases)))
        X1_tr = np.vstack([s[0] for s in tr_s])
        y1_tr = np.concatenate([s[1] for s in tr_s])
        X2w_tr = np.vstack([s[2] for s in tr_s])
        y2w_tr = np.concatenate([s[3] for s in tr_s])
        X2s_tr = np.vstack([s[4] for s in tr_s])
        y2s_tr = np.concatenate([s[5] for s in tr_s])

        et_v1 = ExtraTreesClassifier(n_estimators=100, max_depth=24, min_samples_leaf=5, n_jobs=-1, random_state=42).fit(X1_tr, y1_tr)
        mid = len(tr_cases) // 2
        n1_mid = sum(len(s[1]) for s in tr_s[:mid])
        et_v1_a = ExtraTreesClassifier(n_estimators=80, max_depth=24, min_samples_leaf=5, n_jobs=-1, random_state=43).fit(X1_tr[:n1_mid], y1_tr[:n1_mid])
        et_v1_b = ExtraTreesClassifier(n_estimators=80, max_depth=24, min_samples_leaf=5, n_jobs=-1, random_state=44).fit(X1_tr[n1_mid:], y1_tr[n1_mid:])

        hgb_whole_1 = HistGradientBoostingClassifier(
            max_iter=230, max_leaf_nodes=63, min_samples_leaf=25,
            learning_rate=0.05, max_features=0.65, l2_regularization=1.0,
            early_stopping=False, random_state=142,
        ).fit(X2w_tr, y2w_tr)
        hgb_whole_2 = HistGradientBoostingClassifier(
            max_iter=230, max_leaf_nodes=79, min_samples_leaf=30,
            learning_rate=0.05, max_features=0.75, l2_regularization=1.5,
            early_stopping=False, random_state=143,
        ).fit(X2w_tr, y2w_tr)
        et_whole = ExtraTreesClassifier(n_estimators=100, max_depth=24, min_samples_leaf=6, n_jobs=-1, random_state=144).fit(X2w_tr, y2w_tr)

        hgb_sub_1 = HistGradientBoostingClassifier(
            max_iter=240, max_leaf_nodes=63, min_samples_leaf=25,
            learning_rate=0.05, max_features=0.65, l2_regularization=1.0,
            early_stopping=False, random_state=242,
        ).fit(X2s_tr, y2s_tr)
        hgb_sub_2 = HistGradientBoostingClassifier(
            max_iter=240, max_leaf_nodes=79, min_samples_leaf=30,
            learning_rate=0.05, max_features=0.75, l2_regularization=1.5,
            early_stopping=False, random_state=243,
        ).fit(X2s_tr, y2s_tr)
        et_sub = ExtraTreesClassifier(n_estimators=100, max_depth=24, min_samples_leaf=6, n_jobs=-1, random_state=244).fit(X2s_tr, y2s_tr)

        bw = [0]
        bs = [0]
        for s in tr_s:
            bw.append(bw[-1] + len(s[3]))
            bs.append(bs[-1] + len(s[5]))
        oof_w_folds, oof_s_folds = [], []
        for fk in range(4):
            sw_i, ew_i = bw[fk * f_sz], bw[(fk + 1) * f_sz]
            ss_i, es_i = bs[fk * f_sz], bs[(fk + 1) * f_sz]
            oof_w_folds.append(HistGradientBoostingClassifier(
                max_iter=220, max_leaf_nodes=63, min_samples_leaf=25,
                learning_rate=0.05, max_features=0.70, l2_regularization=1.0,
                early_stopping=False, random_state=300 + fk,
            ).fit(np.vstack([X2w_tr[:sw_i], X2w_tr[ew_i:]]), np.concatenate([y2w_tr[:sw_i], y2w_tr[ew_i:]])))
            oof_s_folds.append(HistGradientBoostingClassifier(
                max_iter=220, max_leaf_nodes=63, min_samples_leaf=25,
                learning_rate=0.05, max_features=0.70, l2_regularization=1.0,
                early_stopping=False, random_state=400 + fk,
            ).fit(np.vstack([X2s_tr[:ss_i], X2s_tr[es_i:]]), np.concatenate([y2s_tr[:ss_i], y2s_tr[es_i:]])))

        del tr_s, X1_tr, y1_tr, X2w_tr, y2w_tr, X2s_tr, y2s_tr
        gc.collect()

        ref_models_all[bone_name] = {
            'et_v1': et_v1,
            'hgb_whole_1': hgb_whole_1,
            'hgb_whole_2': hgb_whole_2,
            'et_whole': et_whole,
            'hgb_sub_1': hgb_sub_1,
            'hgb_sub_2': hgb_sub_2,
            'et_sub': et_sub,
            'p_full': p_full,
        }
        ref_oof_all[bone_name] = (et_v1, et_v1_a, et_v1_b, oof_w_folds, oof_s_folds, p_loo)

        dsc_raw, dsc_v1, dsc_v2 = [], [], []
        maes_raw, maes_v2 = [], []
        for fname, img, lbl, sp, bm_pr, dts_pre in ev_cases:
            m_v2, m_v1 = refine_one_bone_3bone_v2(img, sp, bm_pr, bone_name, ref_models_all[bone_name], dts_pre=dts_pre, return_v1=True)
            gt_b = (lbl == b_lbl)
            m_r = (bm_pr[bone_name] > 0)
            dsc_raw.append(float(2.0 * np.sum(gt_b & m_r) / (np.sum(gt_b) + np.sum(m_r))))
            dsc_v1.append(float(2.0 * np.sum(gt_b & (m_v1 > 0)) / (np.sum(gt_b) + np.sum(m_v1 > 0))))
            dsc_v2.append(float(2.0 * np.sum(gt_b & (m_v2 > 0)) / (np.sum(gt_b) + np.sum(m_v2 > 0))))
            d_gt = fast_signed_distance_transform(gt_b, sp)
            d_v2 = fast_signed_distance_transform(m_v2 > 0, sp)
            d_self = dts_pre[0 if bone_name == 'femur' else (1 if bone_name == 'tibia' else 2)]
            m_int = (lbl == c_lbl) & (d_gt <= 1.5)
            if np.any(m_int):
                maes_raw.append(float(np.mean(np.abs(d_self[m_int] - d_gt[m_int]))))
                maes_v2.append(float(np.mean(np.abs(d_v2[m_int] - d_gt[m_int]))))

        print(f"\n================ {bone_name.upper()} 31-EVAL SUMMARY ({time.time() - t_b:.1f}s) ================", flush=True)
        print(f"  TIER-A: DSC = {np.mean(dsc_raw):.4f} ± {np.std(dsc_raw):.4f} (min={min(dsc_raw):.4f}) | Articular MAE = {np.mean(maes_raw):.4f} mm")
        print(f"  REF V1: DSC = {np.mean(dsc_v1):.4f} ± {np.std(dsc_v1):.4f} (min={min(dsc_v1):.4f})")
        print(f"  REF V2: DSC = {np.mean(dsc_v2):.4f} ± {np.std(dsc_v2):.4f} (min={min(dsc_v2):.4f}) | Articular MAE = {np.mean(maes_v2):.4f} mm", flush=True)

    joblib.dump({
        'format_version': 'oai_3bone_refiner_v2',
        'models': ref_models_all,
    }, os.path.join(model_dir, 'bone_rf_p2.joblib'), compress=3)
    print(f"Saved bone_rf_p2.joblib!", flush=True)

    def gen_eval_bones(c):
        fname, img, lbl, sp, bm_pr, dts_pre = c
        bm1, bm2 = {}, {}
        for bone_name in ['femur', 'tibia', 'patella']:
            m_v2, m_v1 = refine_one_bone_3bone_v2(img, sp, bm_pr, bone_name, ref_models_all[bone_name], dts_pre=dts_pre, return_v1=True)
            bm1[bone_name] = m_v1
            bm2[bone_name] = m_v2
        return fname, bm1, bm2

    ev_res = Parallel(n_jobs=12, prefer='threads')(delayed(gen_eval_bones)(c) for c in ev_cases)
    eval31_ref_v1 = {k: v1 for k, v1, _ in ev_res}
    eval31_ref_v2 = {k: v2 for k, _, v2 in ev_res}
    joblib.dump(eval31_ref_v1, os.path.join(cache_dir, 'eval31_ref_bones_v1.joblib'), compress=3)
    joblib.dump(eval31_ref_v2, os.path.join(cache_dir, 'eval31_ref_bones_v2.joblib'), compress=3)

    mid = len(tr_cases) // 2
    def gen_train_bones(i):
        fname, img, lbl, sp, bm_pr, dts_pre = tr_cases[i]
        fk = i // f_sz
        bm1, bm2 = {}, {}
        for bone_name in ['femur', 'tibia', 'patella']:
            et_v1, et_v1_a, et_v1_b, oof_w_folds, oof_s_folds, p_loo = ref_oof_all[bone_name]
            shell, d_self, d_other, X1, X2 = extract_ray_feats_3bone_v2(img, sp, bm_pr, bone_name, p_loo[i], dts_pre=dts_pre, rng_jit=None)
            et_oof = et_v1_b if i < mid else et_v1_a
            with joblib.parallel_backend('threading'):
                p1_oof = et_oof.predict_proba(X1)[:, 1].astype(np.float32)
                p1_full = et_v1.predict_proba(X1)[:, 1].astype(np.float32)
            p1 = 0.50 * p1_full + 0.50 * p1_oof
            pw_oof = oof_w_folds[fk].predict_proba(X2)[:, 1].astype(np.float32)
            ps_oof = oof_s_folds[fk].predict_proba(X2)[:, 1].astype(np.float32)

            max_other = 45.0 if bone_name == 'femur' else 25.0
            d_o_sh = d_other[shell]
            w_art = np.clip((max_other + 4.0 - d_o_sh) / 8.0, 0.0, 1.0) * (d_o_sh >= -3.0)
            p2_art = 0.30 * p1_oof + 0.50 * ps_oof + 0.20 * pw_oof
            p2_non = 0.30 * p1_oof + 0.70 * pw_oof
            p_bl = w_art * p2_art + (1.0 - w_art) * p2_non

            p_orig = 1.0 / (1.0 + np.exp(np.clip(d_self / 0.45, -20.0, 20.0)))
            sig_s = [0.75 / sp[0], 0.55 / sp[1], 0.65 / sp[2]]
            vol1 = p_orig.copy()
            vol1[shell] = p1
            bm1[bone_name] = postprocess_3bone_mask(gaussian_filter(vol1, sigma=sig_s) >= 0.50, sp, bone_name=bone_name).astype(np.uint8)
            volb = p_orig.copy()
            volb[shell] = p_bl
            bm2[bone_name] = postprocess_3bone_mask(gaussian_filter(volb, sigma=sig_s) >= 0.50, sp, bone_name=bone_name).astype(np.uint8)
        return fname, bm1, bm2

    tr_res = Parallel(n_jobs=12, prefer='threads')(delayed(gen_train_bones)(i) for i in range(len(tr_cases)))
    train128_ref_v1 = {k: v1 for k, v1, _ in tr_res}
    train128_ref_v2 = {k: v2 for k, _, v2 in tr_res}
    joblib.dump(train128_ref_v1, os.path.join(cache_dir, 'train128_ref_bones_v1.joblib'), compress=3)
    joblib.dump(train128_ref_v2, os.path.join(cache_dir, 'train128_ref_bones_v2.joblib'), compress=3)
    print(f"Saved OOF V1 and V2 refined bones for all 128 OAI train + 31 eval scans in {time.time() - t_start:.1f}s!", flush=True)
