import gc, json, os, shutil, sys, time, warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    maximum_filter1d,
    minimum_filter1d,
    zoom,
)
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_EXP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
from kneeseg.io import load_volume
from kneeseg.features import fast_signed_distance_transform
from kneeseg.rf_seg import _compute_articular_landmarks_and_stats
from kneeseg.bone_rf import BoneClassifier, _postprocess_bone_mask, _compute_shifted_diffs_3d

data_dir = os.environ.get('SKI10_DATA_DIR', '/path/to/SKI10/TrainingData')
lbl_dir = os.environ.get('SKI10_LABEL_DIR', data_dir + '_labels')
cache_dir = os.path.join(_EXP_DIR, 'cache')
model_dir = os.path.join(_EXP_DIR, 'models')

# ==============================================================================
# TIER A: Joint-Anchored 3D Bone Spatial Prior Atlas + Continuous 1D Cortical
#         Enclosure Ray Global Corrector (4-Fold OOF on Train-80, Full-80 on Eval-20)
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

    g1 = g_dict[1.0]; g2 = g_dict[2.0]; g24 = g_dict[24.0]
    var24 = gaussian_filter((img_norm - g24)**2, sigma=[24.0 / s for s in sp_s])
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
        jz, jy, jx, np.abs(jz), np.sqrt(jy**2 + jx**2),
        pf_fov, pt_fov, pf_j, pt_j, pf_s, pt_s, p_fb, p_tb, log_ft,
    ] + cont_cols

    rng = np.random.RandomState(42)
    offsets = np.vstack([rng.uniform(-14.0, 14.0, size=(10, 3)), rng.uniform(-36.0, 36.0, size=(10, 3))])
    rsid = _compute_shifted_diffs_3d(img_norm, z_idx, y_idx, x_idx, offsets, sp_s)

    df_m = df_s[z_idx, y_idx, x_idx]
    dt_m = dt_s[z_idx, y_idx, x_idx]
    if rng_jit is not None:
        N = len(df_m)
        heavy = rng_jit.rand(N) < 0.30
        jf = np.where(heavy, rng_jit.uniform(-9.0, 9.0, size=N), rng_jit.normal(0.0, 2.8, size=N)).astype(np.float32)
        jt = np.where(heavy, rng_jit.uniform(-9.0, 9.0, size=N), rng_jit.normal(0.0, 2.8, size=N)).astype(np.float32)
        df_feat = np.clip(df_m + jf, -7.0, 10.0)
        dt_feat = np.clip(dt_m + jt, -7.0, 10.0)
    else:
        df_feat = np.clip(df_m, -7.0, 10.0)
        dt_feat = np.clip(dt_m, -7.0, 10.0)

    return np.hstack([
        np.column_stack(cols),
        rx_neg, rx_pos, rz_neg, rz_pos, ry_neg, ry_pos,
        rsid,
        np.column_stack([df_feat, dt_feat, df_feat - dt_feat]),
    ]).astype(np.float32)

def load_full_case(fname, pr_dict):
    img, sp = load_volume(os.path.join(data_dir, fname), return_spacing=True)
    lbl = load_volume(os.path.join(lbl_dir, fname.replace('image-', 'labels-')), return_spacing=False).astype(np.uint8)
    lbl[lbl == 5] = 1
    lbl[lbl == 6] = 3
    return fname, img, lbl, sp, pr_dict[fname]

def prep_strided_case(item):
    fname, img, lbl, sp, bm_orig = item
    img_s = img[::1, ::2, ::2].astype(np.float32)
    lbl_s = lbl[::1, ::2, ::2]
    y_s = np.zeros_like(lbl_s, dtype=np.uint8)
    y_s[lbl_s == 1] = 1
    y_s[lbl_s == 3] = 2
    sp_s = (sp[0], sp[1] * 2.0, sp[2] * 2.0)
    fb_s = bm_orig['femur'][::1, ::2, ::2] > 0
    tb_s = bm_orig['tibia'][::1, ::2, ::2] > 0
    df_s, dt_s, c_j, span_z = compute_joint_frame_strided(fb_s, tb_s, sp_s)
    return fname, img_s, y_s, lbl_s, sp_s, df_s, dt_s, c_j, span_z

def hist_tierA_one(item):
    fname, img_s, y_s, lbl_s, sp_s, df_s, dt_s, c_j, span_z = item
    f1, f2, fs = get_joint_bins(img_s.shape, sp_s, c_j, span_z)
    yr = y_s.ravel()
    fb = (yr == 1); tb = (yr == 2)
    return (
        np.bincount(f1, minlength=20480).reshape(20, 32, 32).astype(np.float32),
        np.bincount(f1[fb], minlength=20480).reshape(20, 32, 32).astype(np.float32),
        np.bincount(f1[tb], minlength=20480).reshape(20, 32, 32).astype(np.float32),
        np.bincount(f2, minlength=30720).reshape(24, 40, 32).astype(np.float32),
        np.bincount(f2[fb], minlength=30720).reshape(24, 40, 32).astype(np.float32),
        np.bincount(f2[tb], minlength=30720).reshape(24, 40, 32).astype(np.float32),
        np.bincount(fs, minlength=20480).reshape(16, 40, 32).astype(np.float32),
        np.bincount(fs[fb], minlength=20480).reshape(16, 40, 32).astype(np.float32),
        np.bincount(fs[tb], minlength=20480).reshape(16, 40, 32).astype(np.float32),
    )

def fin_tierA_atl(items):
    s = [sum(p[k] for p in items) for k in range(9)]
    return (
        (gaussian_filter(s[1], 0.8) / (gaussian_filter(s[0], 0.8) + 4.0)).ravel(),
        (gaussian_filter(s[2], 0.8) / (gaussian_filter(s[0], 0.8) + 4.0)).ravel(),
        (gaussian_filter(s[4], 0.8) / (gaussian_filter(s[3], 0.8) + 4.0)).ravel(),
        (gaussian_filter(s[5], 0.8) / (gaussian_filter(s[3], 0.8) + 4.0)).ravel(),
        (gaussian_filter(s[7], 0.8) / (gaussian_filter(s[6], 0.8) + 4.0)).ravel(),
        (gaussian_filter(s[8], 0.8) / (gaussian_filter(s[6], 0.8) + 4.0)).ravel(),
    )

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
        # Fill axial cross-section holes in shaft/metaphysis away from intercondylar notch
        if (is_femur and jy <= -0.16) or ((not is_femur) and jy >= 0.14):
            crop[:, y_loc, :] = binary_fill_holes(crop[:, y_loc, :])
    for z_loc in range(crop.shape[0]):
        crop[z_loc] = binary_fill_holes(crop[z_loc])
    crop = binary_fill_holes(crop)
    out = np.zeros_like(cc, dtype=bool)
    out[z0:z1, y0:y1, x0:x1] = crop
    return out

def apply_tierA_case(full_item, strided_item, models_list, atl):
    fname, img, lbl, sp, bm_orig = full_item
    _, img_s, y_s, lbl_s, sp_s, df_s, dt_s, c_j, span_z = strided_item
    f1, f2, fs = get_joint_bins(img_s.shape, sp_s, c_j, span_z)
    p_atl = np.maximum(atl[2][f2] + atl[3][f2], atl[4][fs] + atl[5][fs]).reshape(img_s.shape)
    cand = (df_s <= 10.0) | (dt_s <= 10.0) | (p_atl >= 0.02)
    X_c = extract_tierA_features(img_s, sp_s, df_s, dt_s, c_j, span_z, cand, atl, rng_jit=None)
    p_pred = sum(m.predict_proba(X_c) for m in models_list) / len(models_list)

    pf_raw_s = 1.0 / (1.0 + np.exp(np.clip(df_s[cand] / 1.5, -15.0, 15.0)))
    pt_raw_s = 1.0 / (1.0 + np.exp(np.clip(dt_s[cand] / 1.5, -15.0, 15.0)))
    p0_raw_s = np.clip(1.0 - pf_raw_s - pt_raw_s, 0.0, 1.0)
    pr_raw_s = np.column_stack([p0_raw_s, pf_raw_s, pt_raw_s])

    # Adaptive agreement: blend 0.15 raw only where Tier-A and raw agree within 0.45
    disagree = np.max(np.abs(p_pred - pr_raw_s), axis=1, keepdims=True)
    w_raw = 0.15 * np.clip(1.0 - disagree / 0.45, 0.0, 1.0)
    pb_s = (1.0 - w_raw) * p_pred + w_raw * pr_raw_s

    vol_s = np.zeros((*img_s.shape, 3), dtype=np.float32)
    vol_s[..., 0] = 1.0
    vol_s[cand] = pb_s
    for c in range(3):
        vol_s[..., c] = gaussian_filter(vol_s[..., c], sigma=[1.0 / s for s in sp_s])

    zf = [img.shape[d] / img_s.shape[d] for d in range(3)]
    pf_up = zoom(vol_s[..., 1], zf, order=1)
    pt_up = zoom(vol_s[..., 2], zf, order=1)

    # Blend with full-res original SDT near agreeing boundary (|d_orig| <= 2.5mm)
    df_orig = fast_signed_distance_transform(bm_orig['femur'] > 0, sp)
    dt_orig = fast_signed_distance_transform(bm_orig['tibia'] > 0, sp)
    pf_orig = 1.0 / (1.0 + np.exp(np.clip(df_orig / 0.9, -15.0, 15.0)))
    pt_orig = 1.0 / (1.0 + np.exp(np.clip(dt_orig / 0.9, -15.0, 15.0)))

    wf_b = 0.25 * np.clip(1.0 - np.abs(pf_up - pf_orig) / 0.38, 0.0, 1.0) * (np.abs(df_orig) <= 2.5)
    wt_b = 0.25 * np.clip(1.0 - np.abs(pt_up - pt_orig) / 0.38, 0.0, 1.0) * (np.abs(dt_orig) <= 2.5)
    pf_fin = (1.0 - wf_b) * pf_up + wf_b * pf_orig
    pt_fin = (1.0 - wt_b) * pt_up + wt_b * pt_orig

    fb_mask = postprocess_bone_with_shaft_fill((pf_fin >= 0.47) & (pf_fin >= pt_fin), sp, c_j, span_z, is_femur=True)
    tb_mask = postprocess_bone_with_shaft_fill((pt_fin >= 0.47) & (pt_fin > pf_fin), sp, c_j, span_z, is_femur=False)
    return fname, {'femur': fb_mask.astype(np.uint8), 'tibia': tb_mask.astype(np.uint8)}

# ==============================================================================
# TIER B: Full-Resolution Articular Subchondral Refiner (RefBone V1 & V2)
# ==============================================================================
def detect_medial_is_low_z(bm, sp):
    fb = bm['femur'] > 0
    tb = bm['tibia'] > 0
    if np.sum(fb) < 20 or np.sum(tb) < 20:
        return True
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
    bz_ml  = np.clip(((uz_ml + 0.90) / 1.80 * 28).astype(int), 0, 27)
    by = np.clip(((uy + 0.90) / 1.80 * 32).astype(int), 0, 31)
    bx = np.clip(((ux + 0.90) / 1.80 * 24).astype(int), 0, 23)
    return (bz_sym, by, bx), (bz_ml, by, bx)

def build_bone_atlas(cases, bone_name, b_lbl):
    max_other = 45.0 if bone_name == 'femur' else 25.0
    other_name = 'tibia' if bone_name == 'femur' else 'femur'
    h_tot_s = np.zeros((24, 32, 24), dtype=np.float32)
    h_pos_s = np.zeros((24, 32, 24), dtype=np.float32)
    h_tot_m = np.zeros((28, 32, 24), dtype=np.float32)
    h_pos_m = np.zeros((28, 32, 24), dtype=np.float32)
    per_case = []
    for fname, img, lbl, sp, bm_pr in cases:
        d_self = fast_signed_distance_transform(bm_pr[bone_name] > 0, sp)
        d_other = fast_signed_distance_transform(bm_pr[other_name] > 0, sp)
        med_low_z = detect_medial_is_low_z(bm_pr, sp)
        q50, scale, _ = get_robust_bone_frame(bm_pr[bone_name], d_other, max_other, sp, med_low_z)
        shell = (np.abs(d_self) <= 4.5) & (d_other <= max_other) & (d_other >= -3.0)
        inds = np.argwhere(shell)
        coords_mm = inds.astype(np.float32) * np.array(sp, dtype=np.float32)
        bins_s, bins_m = coords_to_bone_atlas_bins(coords_mm, q50, scale, med_low_z)
        y_b = (lbl[shell] == b_lbl)
        ct_s = np.zeros((24, 32, 24), dtype=np.float32); cp_s = np.zeros((24, 32, 24), dtype=np.float32)
        ct_m = np.zeros((28, 32, 24), dtype=np.float32); cp_m = np.zeros((28, 32, 24), dtype=np.float32)
        np.add.at(ct_s, bins_s, 1.0); np.add.at(cp_s, tuple(b[y_b] for b in bins_s), 1.0)
        np.add.at(ct_m, bins_m, 1.0); np.add.at(cp_m, tuple(b[y_b] for b in bins_m), 1.0)
        h_tot_s += ct_s; h_pos_s += cp_s
        h_tot_m += ct_m; h_pos_m += cp_m
        per_case.append((ct_s, cp_s, ct_m, cp_m))

    p_full = (
        gaussian_filter(h_pos_s, sigma=0.8) / (gaussian_filter(h_tot_s, sigma=0.8) + 2.0),
        gaussian_filter(h_pos_m, sigma=0.8) / (gaussian_filter(h_tot_m, sigma=0.8) + 2.0),
    )
    p_loo = [
        (
            gaussian_filter(h_pos_s - ct_s_i[1], sigma=0.8) / (gaussian_filter(h_tot_s - ct_s_i[0], sigma=0.8) + 2.0),
            gaussian_filter(h_pos_m - ct_s_i[3], sigma=0.8) / (gaussian_filter(h_tot_m - ct_s_i[2], sigma=0.8) + 2.0),
        )
        for ct_s_i in per_case
    ]
    return p_full, p_loo

def extract_ray_feats_v2(img, sp, bm_pr, bone_name, bone_atlas_p, rng_jit=None):
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
    i_bn_var  = gaussian_filter(((im_c - i_bn_mean) ** 2) * w_bn, sigma=sig_bn) / w_bn_s

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
    idx_cort_min  = (np.argmin(rn0[:, 5:12], axis=1).astype(np.float32) - 4.0) * 0.5
    val_cort_min  = np.min(rn0[:, 5:12], axis=1)

    dy_steps = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    vy_lat = np.empty((N, 9), dtype=np.float32)
    vy_zb  = np.empty((N, 9), dtype=np.float32)
    for k, dy in enumerate(dy_steps):
        syc = np.clip(yc + dy, 0, cy_max - 1)
        vy_lat[:, k] = (im_c[zc, syc, xc] - i_in[:, 0]) / i_span[:, 0]
        vy_zb[:, k]  = z_bone_crop[zc, syc, xc]
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

    d_self_m = d_self[z_idx, y_idx, x_idx]
    d_other_m = d_other[z_idx, y_idx, x_idx]
    if rng_jit is not None:
        jit = rng_jit.uniform(-0.95, 0.95, size=N).astype(np.float32)
        w_jit = np.clip(1.0 - np.abs(d_self_m) / 2.4, 0.30, 1.0)
        d_self_feat = d_self_m + w_jit * jit
    else:
        d_self_feat = d_self_m

    X_v1 = np.hstack([
        np.column_stack([
            uz, uy, ux, np.abs(rel_f[:, 0]), rel_f[:, 1], rel_f[:, 2],
            p_atl_s, nz_m, ny_m, nx_m, v_s2, (i_out[:, 0] - i_in[:, 0]) / iqr
        ]),
        rn0, rn1, rg0, rg1, rv0_iqr, rv1_iqr
    ]).astype(np.float32)

    X_v2 = np.hstack([
        X_v1,
        np.column_stack([
            p_atl_m, uz_ml, d_self_feat, d_other_m,
            idx_peak_rise, val_peak_rise, idx_cort_min, val_cort_min,
            z_bone_crop[zc, yc, xc],
        ]),
        ray_zb[:, 3:16], vy_lat, vyg_lat, vy_zb
    ]).astype(np.float32)
    return shell, d_self, X_v1, X_v2


def predict_tierA_probs(image, spacing, tierA_or_hgb1, hgb_2=None, atl_full=None, bm_orig=None):
    """Run Tier-A global bone spatial-atlas + cortical enclosure prediction on a single volume."""
    img = np.asarray(image, dtype=np.float32)
    sp = tuple(float(s) for s in spacing)
    img_s = img[::1, ::2, ::2].astype(np.float32)
    sp_s = (sp[0], sp[1] * 2.0, sp[2] * 2.0)
    Z, Y, X = img_s.shape
    if isinstance(tierA_or_hgb1, dict):
        atl = tierA_or_hgb1['atl_full']
        models_list = [tierA_or_hgb1['hgb_1'], tierA_or_hgb1['hgb_2']]
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
    f1, f2, fs = get_joint_bins(img_s.shape, sp_s, c_j, span_z)
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
    c_j = p_a.get('c_j', np.array([0.5 * Z * sp[0], 0.55 * Y * sp[1], 0.5 * X * sp[2]], dtype=np.float32))
    span_z = float(p_a.get('span_z', 65.0))
    fb_mask = postprocess_bone_with_shaft_fill(
        (pf_fin >= 0.47) & (pf_fin >= pt_fin), sp, c_j, span_z, is_femur=True
    )
    tb_mask = postprocess_bone_with_shaft_fill(
        (pt_fin >= 0.47) & (pt_fin > pf_fin), sp, c_j, span_z, is_femur=False
    )
    return {'femur': fb_mask.astype(np.uint8), 'tibia': tb_mask.astype(np.uint8)}


def _refine_one_bone_v2(img, sp, bm_a, bone_name, m_d, return_v1=False):
    if np.sum(bm_a['femur'] > 0) < 20 or np.sum(bm_a['tibia'] > 0) < 20:
        m0 = (bm_a[bone_name] > 0).astype(np.uint8)
        return (m0, m0) if return_v1 else m0
    shell, d_self, X1, X2 = extract_ray_feats_v2(
        img, sp, bm_a, bone_name, m_d['p_full'], rng_jit=None
    )
    if not np.any(shell):
        m0 = (bm_a[bone_name] > 0).astype(np.uint8)
        return (m0, m0) if return_v1 else m0
    p1 = m_d['et_v1'].predict_proba(X1)[:, 1]
    p2 = (
        0.40 * m_d['hgb_v2_1'].predict_proba(X2)[:, 1]
        + 0.35 * m_d['hgb_v2_2'].predict_proba(X2)[:, 1]
        + 0.25 * m_d['et_v2'].predict_proba(X2)[:, 1]
    )
    p_bl = 0.35 * p1 + 0.65 * p2
    p_orig = 1.0 / (1.0 + np.exp(np.clip(d_self / 0.4, -20.0, 20.0)))
    sig_s = [0.8 / sp[0], 0.35 / sp[1], 0.6 / sp[2]]
    volb = p_orig.copy()
    volb[shell] = p_bl
    m_v2 = _postprocess_bone_mask(
        gaussian_filter(volb, sigma=sig_s) >= 0.50
    ).astype(np.uint8)
    if not return_v1:
        return m_v2
    volb1 = p_orig.copy()
    volb1[shell] = p1
    m_v1 = _postprocess_bone_mask(
        gaussian_filter(volb1, sigma=sig_s) >= 0.50
    ).astype(np.uint8)
    return m_v2, m_v1


def refine_bone_v2_single(image, spacing, bm_self_or_dict, bm_other_or_models, bone_name=None, ref_v2_bone=None, return_v1=False):
    """Run Tier-B V2 articular subchondral refiner on a single volume given Tier-A bone masks."""
    img = np.asarray(image, dtype=np.float32)
    sp = tuple(float(s) for s in spacing)
    if bone_name is not None:
        other_name = 'femur' if bone_name == 'tibia' else 'tibia'
        bm_a = {bone_name: bm_self_or_dict, other_name: bm_other_or_models}
        return _refine_one_bone_v2(img, sp, bm_a, bone_name, ref_v2_bone, return_v1=return_v1)
    bm_a = bm_self_or_dict
    v2_models = bm_other_or_models
    if not return_v1:
        return {
            b: _refine_one_bone_v2(img, sp, bm_a, b, v2_models[b], return_v1=False)
            for b in ['femur', 'tibia']
        }
    res = {
        b: _refine_one_bone_v2(img, sp, bm_a, b, v2_models[b], return_v1=True)
        for b in ['femur', 'tibia']
    }
    return (
        {b: res[b][0] for b in ['femur', 'tibia']},
        {b: res[b][1] for b in ['femur', 'tibia']},
    )


if __name__ == '__main__':
    with open(os.path.join(_ROOT_DIR, 'kneeseg', 'data', 'ski10_full_split.json')) as f:
        split = json.load(f)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    eval_raw_path = os.path.join(cache_dir, 'eval_pred_bones.joblib')
    train_raw_path = os.path.join(cache_dir, 'train80_pred_bones.joblib')
    eval_orig_path = os.path.join(cache_dir, 'eval_pred_bones_orig.joblib')
    train_orig_path = os.path.join(cache_dir, 'train80_pred_bones_orig.joblib')

    if not os.path.exists(eval_orig_path) and os.path.exists(eval_raw_path):
        shutil.copyfile(eval_raw_path, eval_orig_path)
    if not os.path.exists(train_orig_path) and os.path.exists(train_raw_path):
        shutil.copyfile(train_raw_path, train_orig_path)

    if os.path.exists(eval_orig_path) and os.path.exists(train_orig_path):
        eval_orig_bones = joblib.load(eval_orig_path)
        train80_orig_bones = joblib.load(train_orig_path)
        print(f"Loaded initial Pass-1/2 bone predictions ({len(train80_orig_bones)} train, {len(eval_orig_bones)} eval)", flush=True)
    else:
        raise FileNotFoundError("Missing initial bone prediction cache.")

    t_tierA = time.time()
    t0 = time.time()
    tr_cases_orig = Parallel(n_jobs=16, prefer='threads')(delayed(load_full_case)(f, train80_orig_bones) for f in split['train'])
    ev_cases_orig = Parallel(n_jobs=16, prefer='threads')(delayed(load_full_case)(f, eval_orig_bones) for f in split['eval'])
    print(f"Loaded 80 train + 20 eval cases in {time.time()-t0:.1f}s", flush=True)

    tr_strided = Parallel(n_jobs=16, prefer='threads')(delayed(prep_strided_case)(it) for it in tr_cases_orig)
    ev_strided = Parallel(n_jobs=16, prefer='threads')(delayed(prep_strided_case)(it) for it in ev_cases_orig)

    per_case_atl = Parallel(n_jobs=16, prefer='threads')(delayed(hist_tierA_one)(it) for it in tr_strided)
    atl_tierA_full = fin_tierA_atl(per_case_atl)
    atl_tierA_folds = [fin_tierA_atl(per_case_atl[:fk * 20] + per_case_atl[(fk + 1) * 20:]) for fk in range(4)]

    def sample_tierA_tr(i):
        fname, img_s, y_s, lbl_s, sp_s, df_s, dt_s, c_j, span_z = tr_strided[i]
        atl = atl_tierA_folds[i // 20]
        dts_gt = [fast_signed_distance_transform(y_s == c, sp_s) for c in [1, 2]]
        min_abs_d = np.minimum(np.abs(dts_gt[0]), np.abs(dts_gt[1]))
        p5, p50, p95 = np.percentile(img_s, [5, 50, 95])
        inorm = (img_s - p50) / max(float(p95 - p5), 1.0)
        rng = np.random.RandomState(100 + i)
        cart_mask = (lbl_s == 2) | (lbl_s == 4)
        sub_mask = binary_dilation(cart_mask, iterations=2) & (y_s > 0)
        raw_disagree = ((df_s <= 0) != (y_s == 1)) | ((dt_s <= 0) != (y_s == 2))
        arrs = []
        for pool, n_take in [
            (np.argwhere(min_abs_d <= 6.0), 3200),
            (np.argwhere(raw_disagree), 1800),
            (np.argwhere(cart_mask), 1100),
            (np.argwhere(sub_mask), 1100),
            (np.argwhere((y_s > 0) & (inorm > -0.02)), 1600),
            (np.argwhere((y_s > 0) & (min_abs_d > 6.0)), 1400),
            (np.argwhere((y_s == 0) & (min_abs_d > 3.0) & (inorm < -0.02)), 2000),
            (np.argwhere((y_s == 0) & (min_abs_d > 6.0)), 1600),
        ]:
            if len(pool) > 0:
                arrs.append(pool[rng.choice(len(pool), min(len(pool), n_take), replace=False)])
        smask = np.zeros_like(y_s, dtype=bool)
        for a in arrs:
            smask[a[:, 0], a[:, 1], a[:, 2]] = True
        return extract_tierA_features(img_s, sp_s, df_s, dt_s, c_j, span_z, smask, atl, rng_jit=rng), y_s[smask]

    tr_tierA_samples = Parallel(n_jobs=16, prefer='threads')(delayed(sample_tierA_tr)(i) for i in range(80))
    X_tierA = np.vstack([s[0] for s in tr_tierA_samples])
    y_tierA = np.concatenate([s[1] for s in tr_tierA_samples])
    fold_bounds_A = [0]
    for s in tr_tierA_samples:
        fold_bounds_A.append(fold_bounds_A[-1] + len(s[1]))
    del tr_tierA_samples
    gc.collect()
    print(f"Sampled Tier-A matrix {X_tierA.shape} in {time.time()-t_tierA:.1f}s", flush=True)

    hgb_tierA_1 = HistGradientBoostingClassifier(
        max_iter=190, max_leaf_nodes=63, min_samples_leaf=25,
        learning_rate=0.055, max_features=0.70, l2_regularization=1.0,
        early_stopping=False, random_state=42
    ).fit(X_tierA, y_tierA)
    hgb_tierA_2 = HistGradientBoostingClassifier(
        max_iter=190, max_leaf_nodes=79, min_samples_leaf=30,
        learning_rate=0.055, max_features=0.75, l2_regularization=1.5,
        early_stopping=False, random_state=43
    ).fit(X_tierA, y_tierA)

    oof_tierA_folds = []
    for fk in range(4):
        s_i, e_i = fold_bounds_A[fk * 20], fold_bounds_A[(fk + 1) * 20]
        X_f = np.vstack([X_tierA[:s_i], X_tierA[e_i:]])
        y_f = np.concatenate([y_tierA[:s_i], y_tierA[e_i:]])
        clf_f = HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=63, min_samples_leaf=25,
            learning_rate=0.06, max_features=0.70, l2_regularization=1.0,
            early_stopping=False, random_state=140 + fk
        ).fit(X_f, y_f)
        oof_tierA_folds.append(clf_f)
    del X_tierA, y_tierA
    gc.collect()

    joblib.dump({
        'hgb_1': hgb_tierA_1,
        'hgb_2': hgb_tierA_2,
        'oof_folds': oof_tierA_folds,
        'atl_full': atl_tierA_full,
        'atl_folds': atl_tierA_folds,
    }, os.path.join(model_dir, 'bone_tierA_models.joblib'), compress=3)
    print(f"Trained & saved Tier-A full + 4-fold OOF models in {time.time()-t_tierA:.1f}s", flush=True)

    t_infA = time.time()
    eval_pred_bones = {}
    for i in range(len(ev_cases_orig)):
        fn, bm_new = apply_tierA_case(ev_cases_orig[i], ev_strided[i], [hgb_tierA_1, hgb_tierA_2], atl_tierA_full)
        eval_pred_bones[fn] = bm_new
    joblib.dump(eval_pred_bones, eval_raw_path, compress=3)
    print(f"Generated & saved Tier-A eval_pred_bones (20 cases) in {time.time()-t_infA:.1f}s", flush=True)

    t_infA_tr = time.time()
    train80_pred_bones = {}
    for i in range(len(tr_cases_orig)):
        fk = i // 20
        fn, bm_new = apply_tierA_case(tr_cases_orig[i], tr_strided[i], [oof_tierA_folds[fk]], atl_tierA_folds[fk])
        train80_pred_bones[fn] = bm_new
        if (i + 1) % 20 == 0:
            print(f"  Generated OOF Tier-A train80_pred_bones {i+1}/80 in {time.time()-t_infA_tr:.1f}s", flush=True)
    joblib.dump(train80_pred_bones, train_raw_path, compress=3)

    tr_cases = [(fn, img, lbl, sp, train80_pred_bones[fn]) for fn, img, lbl, sp, _ in tr_cases_orig]
    ev_cases = [(fn, img, lbl, sp, eval_pred_bones[fn]) for fn, img, lbl, sp, _ in ev_cases_orig]
    del tr_cases_orig, ev_cases_orig, tr_strided, ev_strided
    gc.collect()

    ref_v1_models = {}
    ref_v2_models = {}

    for bone_name, b_lbl, c_lbl in [('tibia', 3, 4), ('femur', 1, 2)]:
        t1 = time.time()
        p_full, p_loo = build_bone_atlas(tr_cases, bone_name, b_lbl)

        def sample_tr_both(i):
            fname, img, lbl, sp, bm_pr = tr_cases[i]
            rng = np.random.RandomState(42 + i)
            shell, d_self, X1, X2 = extract_ray_feats_v2(img, sp, bm_pr, bone_name, p_loo[i], rng_jit=rng)
            d_c = fast_signed_distance_transform(lbl == c_lbl, sp)[shell]
            d_s = d_self[shell]
            y_all = (lbl[shell] == b_lbl).astype(np.uint8)

            valid1 = np.where(d_c <= 4.5)[0]
            if len(valid1) == 0:
                valid1 = np.arange(len(y_all))
            idx1 = rng.choice(len(valid1), min(len(valid1), 7500), replace=False)

            pos_int = np.where((y_all == 1) & (d_c <= 2.5) & (np.abs(d_s) <= 1.8))[0]
            neg_int = np.where((y_all == 0) & (d_c <= 2.5) & (np.abs(d_s) <= 1.8))[0]
            pos_out = np.where((y_all == 1) & ((d_c > 2.5) | (np.abs(d_s) > 1.8)) & (d_c <= 5.0))[0]
            neg_out = np.where((y_all == 0) & ((d_c > 2.5) | (np.abs(d_s) > 1.8)) & (d_c <= 5.0))[0]

            idx2_list = []
            for pool, n_take in [(pos_int, 2600), (neg_int, 2600), (pos_out, 1400), (neg_out, 1400)]:
                if len(pool) > 0:
                    idx2_list.append(rng.choice(pool, min(len(pool), n_take), replace=False))
            idx2 = np.concatenate(idx2_list)
            return X1[valid1[idx1]], y_all[valid1[idx1]], X2[idx2], y_all[idx2]

        tr_s = Parallel(n_jobs=16, prefer='threads')(delayed(sample_tr_both)(i) for i in range(len(tr_cases)))
        X1_tr = np.vstack([s[0] for s in tr_s]); y1_tr = np.concatenate([s[1] for s in tr_s])
        X2_tr = np.vstack([s[2] for s in tr_s]); y2_tr = np.concatenate([s[3] for s in tr_s])

        et_v1 = ExtraTreesClassifier(n_estimators=100, max_depth=24, min_samples_leaf=5, n_jobs=-1, random_state=42).fit(X1_tr, y1_tr)
        mid = len(tr_cases) // 2
        n1_mid = sum(len(s[1]) for s in tr_s[:mid])
        et_v1_a = ExtraTreesClassifier(n_estimators=80, max_depth=24, min_samples_leaf=5, n_jobs=-1, random_state=43).fit(X1_tr[:n1_mid], y1_tr[:n1_mid])
        et_v1_b = ExtraTreesClassifier(n_estimators=80, max_depth=24, min_samples_leaf=5, n_jobs=-1, random_state=44).fit(X1_tr[n1_mid:], y1_tr[n1_mid:])
        ref_v1_models[bone_name] = (et_v1, et_v1_a, et_v1_b, p_full, p_loo)

        hgb_v2_1 = HistGradientBoostingClassifier(
            max_iter=240, max_leaf_nodes=63, min_samples_leaf=25,
            learning_rate=0.05, max_features=0.65, l2_regularization=1.0,
            early_stopping=False, random_state=142
        ).fit(X2_tr, y2_tr)
        hgb_v2_2 = HistGradientBoostingClassifier(
            max_iter=240, max_leaf_nodes=79, min_samples_leaf=30,
            learning_rate=0.05, max_features=0.80, l2_regularization=1.5,
            early_stopping=False, random_state=143
        ).fit(X2_tr, y2_tr)
        et_v2 = ExtraTreesClassifier(n_estimators=120, max_depth=24, min_samples_leaf=6, n_jobs=-1, random_state=144).fit(X2_tr, y2_tr)

        fold_bounds = [0]
        for s in tr_s:
            fold_bounds.append(fold_bounds[-1] + len(s[3]))
        oof_v2_folds = []
        for fk in range(4):
            s_i, e_i = fold_bounds[fk * 20], fold_bounds[(fk + 1) * 20]
            X_f = np.vstack([X2_tr[:s_i], X2_tr[e_i:]])
            y_f = np.concatenate([y2_tr[:s_i], y2_tr[e_i:]])
            clf_f = HistGradientBoostingClassifier(
                max_iter=240, max_leaf_nodes=63, min_samples_leaf=25,
                learning_rate=0.05, max_features=0.70, l2_regularization=1.0,
                early_stopping=False, random_state=200 + fk
            ).fit(X_f, y_f)
            oof_v2_folds.append(clf_f)
        del tr_s, X1_tr, y1_tr, X2_tr, y2_tr; gc.collect()
        ref_v2_models[bone_name] = (hgb_v2_1, hgb_v2_2, et_v2, oof_v2_folds, p_full, p_loo)

        maes_raw, maes_v1, maes_v2, maes_blend = [], [], [], []
        dsc_raw, dsc_v1, dsc_v2, dsc_blend = [], [], [], []
        tc_in_raw, tc_in_v1, tc_in_v2, tc_in_blend = [], [], [], []
        tb_out_raw, tb_out_v1, tb_out_v2, tb_out_blend = [], [], [], []

        for fname, img, lbl, sp, bm_pr in ev_cases:
            shell, d_self, X1, X2 = extract_ray_feats_v2(img, sp, bm_pr, bone_name, p_full, rng_jit=None)
            p1 = et_v1.predict_proba(X1)[:, 1]
            p2 = 0.40 * hgb_v2_1.predict_proba(X2)[:, 1] + 0.35 * hgb_v2_2.predict_proba(X2)[:, 1] + 0.25 * et_v2.predict_proba(X2)[:, 1]
            p_bl = 0.35 * p1 + 0.65 * p2

            p_orig = 1.0 / (1.0 + np.exp(np.clip(d_self / 0.4, -20.0, 20.0)))
            sig_s = [0.8 / sp[0], 0.35 / sp[1], 0.6 / sp[2]]

            vol1 = p_orig.copy(); vol1[shell] = p1
            bm_v1 = _postprocess_bone_mask(gaussian_filter(vol1, sigma=sig_s) >= 0.50)

            vol2 = p_orig.copy(); vol2[shell] = p2
            bm_v2 = _postprocess_bone_mask(gaussian_filter(vol2, sigma=sig_s) >= 0.50)

            volb = p_orig.copy(); volb[shell] = p_bl
            bm_bl = _postprocess_bone_mask(gaussian_filter(volb, sigma=sig_s) >= 0.50)

            gt_b = (lbl == b_lbl)
            d_gt = fast_signed_distance_transform(gt_b, sp)
            d_v1 = fast_signed_distance_transform(bm_v1, sp)
            d_v2 = fast_signed_distance_transform(bm_v2, sp)
            d_bl = fast_signed_distance_transform(bm_bl, sp)

            m_int = (lbl == c_lbl) & (d_gt <= 1.2)
            maes_raw.append(float(np.mean(np.abs(d_self[m_int] - d_gt[m_int]))))
            maes_v1.append(float(np.mean(np.abs(d_v1[m_int] - d_gt[m_int]))))
            maes_v2.append(float(np.mean(np.abs(d_v2[m_int] - d_gt[m_int]))))
            maes_blend.append(float(np.mean(np.abs(d_bl[m_int] - d_gt[m_int]))))

            for lst, bm_x, d_x, tcin_l, tbout_l in [
                (dsc_raw, bm_pr[bone_name] > 0, d_self, tc_in_raw, tb_out_raw),
                (dsc_v1, bm_v1, d_v1, tc_in_v1, tb_out_v1),
                (dsc_v2, bm_v2, d_v2, tc_in_v2, tb_out_v2),
                (dsc_blend, bm_bl, d_bl, tc_in_blend, tb_out_blend),
            ]:
                lst.append(float(2.0 * np.sum(gt_b & bm_x) / (np.sum(gt_b) + np.sum(bm_x))))
                tcin_l.append(int(np.sum((lbl == c_lbl) & (d_x <= 0.0))))
                tbout_l.append(int(np.sum(gt_b & (d_x > 0.0) & (d_x <= 2.0))))

        print(f"\n================ {bone_name.upper()} 20-EVAL SUMMARY ({time.time()-t1:.1f}s) ================", flush=True)
        print(f"  TIER-A: DSC={np.mean(dsc_raw):.4f} (min={min(dsc_raw):.4f}) | Articular MAE={np.mean(maes_raw):.4f} mm | C_in_bone={np.mean(tc_in_raw):6.1f} | Bone_out={np.mean(tb_out_raw):7.1f}")
        print(f"  REF V1: DSC={np.mean(dsc_v1):.4f} (min={min(dsc_v1):.4f}) | Articular MAE={np.mean(maes_v1):.4f} mm | C_in_bone={np.mean(tc_in_v1):6.1f} | Bone_out={np.mean(tb_out_v1):7.1f}")
        print(f"  REF V2: DSC={np.mean(dsc_v2):.4f} (min={min(dsc_v2):.4f}) | Articular MAE={np.mean(maes_v2):.4f} mm | C_in_bone={np.mean(tc_in_v2):6.1f} | Bone_out={np.mean(tb_out_v2):7.1f}")
        print(f"  BLEND : DSC={np.mean(dsc_blend):.4f} (min={min(dsc_blend):.4f}) | Articular MAE={np.mean(maes_blend):.4f} mm | C_in_bone={np.mean(tc_in_blend):6.1f} | Bone_out={np.mean(tb_out_blend):7.1f}", flush=True)

    # Save inference-ready Tier-B refiner models (excluding LOO atlases to keep size compact)
    joblib.dump({
        'tibia': {
            'et_v1': ref_v1_models['tibia'][0],
            'hgb_v2_1': ref_v2_models['tibia'][0],
            'hgb_v2_2': ref_v2_models['tibia'][1],
            'et_v2': ref_v2_models['tibia'][2],
            'p_full': ref_v1_models['tibia'][3],
        },
        'femur': {
            'et_v1': ref_v1_models['femur'][0],
            'hgb_v2_1': ref_v2_models['femur'][0],
            'hgb_v2_2': ref_v2_models['femur'][1],
            'et_v2': ref_v2_models['femur'][2],
            'p_full': ref_v1_models['femur'][3],
        },
    }, os.path.join(model_dir, 'bone_refiner_v2_models.joblib'), compress=3)

    def gen_eval_bones(item):
        fname, img, lbl, sp, bm_pr = item
        bm1, bm2 = {}, {}
        for bone_name in ['femur', 'tibia']:
            et_v1, _, _, p_full, _ = ref_v1_models[bone_name]
            hgb_v2_1, hgb_v2_2, et_v2, _, _, _ = ref_v2_models[bone_name]
            shell, d_self, X1, X2 = extract_ray_feats_v2(img, sp, bm_pr, bone_name, p_full, rng_jit=None)
            p1 = et_v1.predict_proba(X1)[:, 1]
            p2 = 0.40 * hgb_v2_1.predict_proba(X2)[:, 1] + 0.35 * hgb_v2_2.predict_proba(X2)[:, 1] + 0.25 * et_v2.predict_proba(X2)[:, 1]
            p_bl = 0.35 * p1 + 0.65 * p2

            p_orig = 1.0 / (1.0 + np.exp(np.clip(d_self / 0.4, -20.0, 20.0)))
            sig_s = [0.8 / sp[0], 0.35 / sp[1], 0.6 / sp[2]]
            vol1 = p_orig.copy(); vol1[shell] = p1
            bm1[bone_name] = _postprocess_bone_mask(gaussian_filter(vol1, sigma=sig_s) >= 0.50).astype(np.uint8)
            volb = p_orig.copy(); volb[shell] = p_bl
            bm2[bone_name] = _postprocess_bone_mask(gaussian_filter(volb, sigma=sig_s) >= 0.50).astype(np.uint8)
        return fname, bm1, bm2

    ev_res = Parallel(n_jobs=10, prefer='threads')(delayed(gen_eval_bones)(c) for c in ev_cases)
    eval_ref_v1 = {k: v1 for k, v1, _ in ev_res}
    eval_ref_v2 = {k: v2 for k, _, v2 in ev_res}
    joblib.dump(eval_ref_v1, os.path.join(cache_dir, 'eval_ref_bones_80.joblib'), compress=3)
    joblib.dump(eval_ref_v2, os.path.join(cache_dir, 'eval_ref_bones_v2.joblib'), compress=3)

    mid = len(tr_cases) // 2
    def gen_train_bones(i):
        fname, img, lbl, sp, bm_pr = tr_cases[i]
        bm1, bm2 = {}, {}
        for bone_name in ['femur', 'tibia']:
            et_v1, et_v1_a, et_v1_b, _, p_loo = ref_v1_models[bone_name]
            hgb_v2_1, hgb_v2_2, et_v2, oof_v2_folds, _, _ = ref_v2_models[bone_name]
            shell, d_self, X1, X2 = extract_ray_feats_v2(img, sp, bm_pr, bone_name, p_loo[i], rng_jit=None)

            et_oof = et_v1_b if i < mid else et_v1_a
            p1 = 0.5 * et_v1.predict_proba(X1)[:, 1] + 0.5 * et_oof.predict_proba(X1)[:, 1]

            p2_oof = oof_v2_folds[i // 20].predict_proba(X2)[:, 1]
            p_bl = 0.30 * et_oof.predict_proba(X1)[:, 1] + 0.70 * p2_oof

            p_orig = 1.0 / (1.0 + np.exp(np.clip(d_self / 0.4, -20.0, 20.0)))
            sig_s = [0.8 / sp[0], 0.35 / sp[1], 0.6 / sp[2]]
            vol1 = p_orig.copy(); vol1[shell] = p1
            bm1[bone_name] = _postprocess_bone_mask(gaussian_filter(vol1, sigma=sig_s) >= 0.50).astype(np.uint8)
            volb = p_orig.copy(); volb[shell] = p_bl
            bm2[bone_name] = _postprocess_bone_mask(gaussian_filter(volb, sigma=sig_s) >= 0.50).astype(np.uint8)
        return fname, bm1, bm2

    tr_res = []
    for start in range(0, len(tr_cases), 10):
        idxs = list(range(start, min(start + 10, len(tr_cases))))
        tr_res.extend(Parallel(n_jobs=len(idxs), prefer='threads')(delayed(gen_train_bones)(i) for i in idxs))
    train_ref_v1 = {k: v1 for k, v1, _ in tr_res}
    train_ref_v2 = {k: v2 for k, _, v2 in tr_res}
    joblib.dump(train_ref_v1, os.path.join(cache_dir, 'train80_ref_bones.joblib'), compress=3)
    joblib.dump(train_ref_v2, os.path.join(cache_dir, 'train80_ref_bones_v2.joblib'), compress=3)
    print("Saved V1 and V2 refined bones for all 80 train + 20 eval cases to persistent cache_dir!", flush=True)
