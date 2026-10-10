import gc, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter, label
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_EXP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
from kneeseg.io import load_volume
from kneeseg.features import fast_signed_distance_transform
from kneeseg.rf_seg import _compute_articular_landmarks_and_stats

with open(os.path.join(_ROOT_DIR, 'kneeseg', 'data', 'ski10_full_split.json')) as f:
    split = json.load(f)
data_dir = os.environ.get('SKI10_DATA_DIR', '/path/to/SKI10/TrainingData')
lbl_dir = os.environ.get('SKI10_LABEL_DIR', data_dir + '_labels')
cache_dir = os.path.join(_EXP_DIR, 'cache')
model_dir = os.path.join(_EXP_DIR, 'models')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


def fast_keep_top_k(mask, k=1, min_size=100):
    if not np.any(mask):
        return mask
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
    zf, yf, xf = np.where(fb)
    zt, yt, xt = np.where(tb)
    y_top_f = np.percentile(yf, 15)
    y_bot_f = np.percentile(yf, 85)
    f_valg = (np.mean(zf[yf <= y_top_f]) - np.mean(zf[yf >= y_bot_f])) * sp[0]

    z_mid = 0.5 * (zt.min() + zt.max())
    tp_top = yt <= np.percentile(yt, 20)
    xt1 = xt[tp_top & (zt < z_mid)]
    xt2 = xt[tp_top & (zt >= z_mid)]
    tb_dx = ((np.percentile(xt1, 95) - np.percentile(xt1, 5)) - (np.percentile(xt2, 95) - np.percentile(xt2, 5))) * sp[2]
    return bool((f_valg + 0.5 * tb_dx) >= 0.0)


def compute_ml_frames(bm, df, dt, sp):
    sp_arr = np.array(sp, dtype=np.float32)
    med_low_z = detect_medial_is_low_z(bm, sp)
    tb = bm['tibia'] > 0
    inds_tb = np.argwhere(tb & (dt >= -2.0) & (df <= 32.0))
    z0, y0, x0 = np.maximum(inds_tb.min(axis=0) - 2, 0)
    z1, y1, x1 = np.minimum(inds_tb.max(axis=0) + 3, tb.shape)
    dts = gaussian_filter(dt[z0:z1, y0:y1, x0:x1], sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
    _, nyt, _ = np.gradient(dts, *sp)

    plat_sub = tb[z0:z1, y0:y1, x0:x1] & (dt[z0:z1, y0:y1, x0:x1] >= -1.2) & (df[z0:z1, y0:y1, x0:x1] <= 24.0) & (nyt <= -0.45)
    if np.sum(plat_sub) < 100:
        plat_sub = tb[z0:z1, y0:y1, x0:x1] & (dt[z0:z1, y0:y1, x0:x1] >= -1.5) & (df[z0:z1, y0:y1, x0:x1] <= 25.0)
    pts = (np.argwhere(plat_sub) + np.array([z0, y0, x0])) * sp_arr
    p_min = np.percentile(pts, 2, axis=0)
    p_max = np.percentile(pts, 98, axis=0)
    p_mid = 0.5 * (p_min + p_max)
    p_span = np.maximum(p_max - p_min, 5.0)

    z_med_split = np.median(pts[:, 0])
    pts_low = pts[pts[:, 0] < z_med_split]
    pts_high = pts[pts[:, 0] >= z_med_split]
    pts_M = pts_low if med_low_z else pts_high
    pts_L = pts_high if med_low_z else pts_low
    c_M, s_M = pts_M.mean(axis=0), pts_M.std(axis=0) + 1.2
    c_L, s_L = pts_L.mean(axis=0), pts_L.std(axis=0) + 1.2
    return med_low_z, (p_min, p_max, p_mid, p_span), (c_M, s_M, c_L, s_L, z_med_split)


def fast_hist_add(shape, bins, mask=None, weights=None):
    b_sub = bins if mask is None else tuple(b[mask] for b in bins)
    lin = np.ravel_multi_index(b_sub, shape)
    w_sub = None if weights is None else (weights if mask is None else weights[mask])
    return np.bincount(lin, weights=w_sub, minlength=int(np.prod(shape))).reshape(shape).astype(np.float32)


SH_OLD  = (26, 26, 16, 18)
SH_ML   = (34, 26, 16, 18)
SH_MLY  = (34, 18, 26, 16)
SH_SEP  = (2, 24, 24, 16, 18)
SH_WIDE = (32, 28, 16, 18)
SH_FO   = (22, 26, 26, 16)
SH_FML  = (34, 26, 26, 16)
SH_FSEP = (2, 22, 26, 26, 16)


def compute_atlas_bins_v6(coords_mm, dfs_m, dts_m, f_stats, t_comp, ml_info):
    med_low_z, (p_min, p_max, p_mid, p_span), (c_M, s_M, c_L, s_L, z_med_split) = ml_info

    # 1. Symmetric Tibia Condyle Frame
    c1_t, s1_t, c2_t, s2_t, spine_t, z_mid_t = t_comp
    in_1t = coords_mm[:, 0] < z_mid_t
    uz_t = np.where(in_1t, (c1_t[0] - coords_mm[:, 0]) / s1_t[0], (coords_mm[:, 0] - c2_t[0]) / s2_t[0])
    uy_t = np.where(in_1t, (coords_mm[:, 1] - c1_t[1]) / s1_t[1], (coords_mm[:, 1] - c2_t[1]) / s2_t[1])
    ux_t = np.where(in_1t, (coords_mm[:, 2] - c1_t[2]) / s1_t[2], (coords_mm[:, 2] - c2_t[2]) / s2_t[2])
    bz_old = np.clip(((uz_t + 2.8) / 5.6 * 26).astype(int), 0, 25)
    bx_old = np.clip(((ux_t + 2.8) / 5.6 * 26).astype(int), 0, 25)
    bdt_16 = np.clip(((dts_m + 0.5) / 6.0 * 16).astype(int), 0, 15)
    bdf_18 = np.clip(((dfs_m + 0.5) / 26.0 * 18).astype(int), 0, 17)

    # 2. Medial-Lateral Canonical Plateau Frame + Wide Unclamped Posterior/Lateral Frame
    u_ml = (coords_mm[:, 0] - p_mid[0]) / p_span[0]
    if not med_low_z:
        u_ml = -u_ml
    u_ap = (coords_mm[:, 2] - p_mid[2]) / p_span[2]
    u_y  = (coords_mm[:, 1] - p_mid[1]) / p_span[1]
    bz_ml = np.clip(((u_ml + 0.62) / 1.24 * 34).astype(int), 0, 33)
    bx_ml = np.clip(((u_ap + 0.62) / 1.24 * 26).astype(int), 0, 25)
    by_ml = np.clip(((u_y + 0.85) / 1.70 * 18).astype(int), 0, 17)

    # Unclamped Wide Frame covering [-0.82, +0.82] so posterior/lateral curved wall is NEVER clamped!
    bz_wide = np.clip(((u_ml + 0.82) / 1.64 * 32).astype(int), 0, 31)
    bx_wide = np.clip(((u_ap + 0.85) / 1.70 * 28).astype(int), 0, 27)

    # 3. Condyle-Separated Upward-Surface Frame
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

    # 4. Femur Old Symmetric, ML-Oriented, and Condyle-Separated Frames
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
        't_old':  (bz_old, bx_old, bdt_16, bdf_18),
        't_ml':   (bz_ml, bx_ml, bdt_16, bdf_18),
        't_mly':  (bz_ml, by_ml, bx_ml, bdt_16),
        't_sep':  (is_lat, bz_p, bx_p, bdt_16, bdf_18),
        't_wide': (bz_wide, bx_wide, bdt_16, bdf_18),
        'f_old':  (bz_fo, by_f, bx_f, bdf_f),
        'f_ml':   (bz_fml, by_f, bx_f, bdf_f),
        'f_sep':  (is_lat_f, bz_fsep, by_f, bx_f, bdf_f),
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


def compute_case_atlas_hist_v6(img, bm, spacing, lbl):
    df = fast_signed_distance_transform(bm['femur'] > 0, spacing)
    dt = fast_signed_distance_transform(bm['tibia'] > 0, spacing)
    roi = (df >= -1.0) & (dt >= -1.0) & (((df <= 5.2) & (dt <= 58.0)) | ((dt <= 5.2) & (df <= 30.0)))
    inds = np.argwhere(roi)
    margin = [int(np.ceil(6.0 / s)) for s in spacing]
    z0, y0, x0 = np.maximum(inds.min(axis=0) - margin, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + margin + 1, img.shape)

    df_smooth = gaussian_filter(df[z0:z1, y0:y1, x0:x1], sigma=[2.0 / spacing[0], 0.7 / spacing[1], 2.0 / spacing[2]])
    dt_smooth = gaussian_filter(dt[z0:z1, y0:y1, x0:x1], sigma=[2.0 / spacing[0], 0.7 / spacing[1], 2.0 / spacing[2]])

    f_stats, _, _ = _compute_articular_landmarks_and_stats(bm['femur'], df, dt, 45.0, spacing)
    _, _, t_comp = _compute_articular_landmarks_and_stats(bm['tibia'], dt, df, 25.0, spacing)
    ml_info = compute_ml_frames(bm, df, dt, spacing)

    zc_r, yc_r, xc_r = inds[:, 0] - z0, inds[:, 1] - y0, inds[:, 2] - x0
    dfs_r_all = df_smooth[zc_r, yc_r, xc_r]
    dts_r_all = dt_smooth[zc_r, yc_r, xc_r]
    pos_sdt = (dfs_r_all >= -0.5) & (dts_r_all >= -0.5)
    inds_atl = inds[pos_sdt]
    coords_roi = inds_atl.astype(np.float32) * np.array(spacing, dtype=np.float32)
    dfs_r = dfs_r_all[pos_sdt]
    dts_r = dts_r_all[pos_sdt]
    bins, _ = compute_atlas_bins_v6(coords_roi, dfs_r, dts_r, f_stats, t_comp, ml_info)
    y_r = lbl[inds_atl[:, 0], inds_atl[:, 1], inds_atl[:, 2]]

    m_tc = (y_r == 4)
    m_fc = (y_r == 2)
    base_t = (dts_r >= -0.2) & (dts_r <= 1.8)
    base_f = (dfs_r >= -0.2) & (dfs_r <= 1.8)

    h_old_tot  = fast_hist_add(SH_OLD, bins['t_old'])
    h_old_tc   = fast_hist_add(SH_OLD, bins['t_old'], m_tc)
    h_old_fc   = fast_hist_add(SH_OLD, bins['t_old'], m_fc)

    h_ml_tot   = fast_hist_add(SH_ML, bins['t_ml'])
    h_ml_tc    = fast_hist_add(SH_ML, bins['t_ml'], m_tc)
    h_ml_fc    = fast_hist_add(SH_ML, bins['t_ml'], m_fc)

    h_mly_tot  = fast_hist_add(SH_MLY, bins['t_mly'])
    h_mly_tc   = fast_hist_add(SH_MLY, bins['t_mly'], m_tc)

    h_sep_tot  = fast_hist_add(SH_SEP, bins['t_sep'])
    h_sep_tc   = fast_hist_add(SH_SEP, bins['t_sep'], m_tc)
    h_sep_fc   = fast_hist_add(SH_SEP, bins['t_sep'], m_fc)

    h_wide_tot = fast_hist_add(SH_WIDE, bins['t_wide'])
    h_wide_tc  = fast_hist_add(SH_WIDE, bins['t_wide'], m_tc)

    b_ml_2d = (bins['t_ml'][0], bins['t_ml'][1])
    h2d_ml_base = fast_hist_add((34, 26), b_ml_2d, base_t)
    h2d_ml_tc   = fast_hist_add((34, 26), b_ml_2d, base_t & m_tc)
    h2d_ml_dts  = fast_hist_add((34, 26), b_ml_2d, m_tc, weights=np.maximum(dts_r, 0.0))

    b_sep_2d = (bins['t_sep'][0], bins['t_sep'][1], bins['t_sep'][2])
    h2d_sep_base = fast_hist_add((2, 24, 24), b_sep_2d, base_t)
    h2d_sep_tc   = fast_hist_add((2, 24, 24), b_sep_2d, base_t & m_tc)
    h2d_sep_dts  = fast_hist_add((2, 24, 24), b_sep_2d, m_tc, weights=np.maximum(dts_r, 0.0))

    b_wide_2d = (bins['t_wide'][0], bins['t_wide'][1])
    h2d_wide_base = fast_hist_add((32, 28), b_wide_2d, base_t)
    h2d_wide_tc   = fast_hist_add((32, 28), b_wide_2d, base_t & m_tc)
    h2d_wide_dts  = fast_hist_add((32, 28), b_wide_2d, m_tc, weights=np.maximum(dts_r, 0.0))

    h_fo_tot   = fast_hist_add(SH_FO, bins['f_old'])
    h_fo_fc    = fast_hist_add(SH_FO, bins['f_old'], m_fc)
    h_fo_tc    = fast_hist_add(SH_FO, bins['f_old'], m_tc)

    h_fml_tot  = fast_hist_add(SH_FML, bins['f_ml'])
    h_fml_fc   = fast_hist_add(SH_FML, bins['f_ml'], m_fc)
    h_fml_tc   = fast_hist_add(SH_FML, bins['f_ml'], m_tc)

    h_fsep_tot = fast_hist_add(SH_FSEP, bins['f_sep'])
    h_fsep_fc  = fast_hist_add(SH_FSEP, bins['f_sep'], m_fc)
    h_fsep_tc  = fast_hist_add(SH_FSEP, bins['f_sep'], m_tc)

    b_fml_3d = (bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2])
    h3d_fml_base = fast_hist_add((34, 26, 26), b_fml_3d, base_f)
    h3d_fml_fc   = fast_hist_add((34, 26, 26), b_fml_3d, base_f & m_fc)
    h3d_fml_dfs  = fast_hist_add((34, 26, 26), b_fml_3d, m_fc, weights=np.maximum(dfs_r, 0.0))

    return (
        h_old_tot, h_old_tc, h_old_fc,
        h_ml_tot, h_ml_tc, h_ml_fc,
        h_mly_tot, h_mly_tc,
        h_sep_tot, h_sep_tc, h_sep_fc,
        h_wide_tot, h_wide_tc,
        h2d_ml_base, h2d_ml_tc, h2d_ml_dts,
        h2d_sep_base, h2d_sep_tc, h2d_sep_dts,
        h2d_wide_base, h2d_wide_tc, h2d_wide_dts,
        h_fo_tot, h_fo_fc, h_fo_tc,
        h_fml_tot, h_fml_fc, h_fml_tc,
        h_fsep_tot, h_fsep_fc, h_fsep_tc,
        h3d_fml_base, h3d_fml_fc, h3d_fml_dfs,
    )


def make_atlas_priors_v6(hists):
    (
        h_old_tot, h_old_tc, h_old_fc,
        h_ml_tot, h_ml_tc, h_ml_fc,
        h_mly_tot, h_mly_tc,
        h_sep_tot, h_sep_tc, h_sep_fc,
        h_wide_tot, h_wide_tc,
        h2d_ml_base, h2d_ml_tc, h2d_ml_dts,
        h2d_sep_base, h2d_sep_tc, h2d_sep_dts,
        h2d_wide_base, h2d_wide_tc, h2d_wide_dts,
        h_fo_tot, h_fo_fc, h_fo_tc,
        h_fml_tot, h_fml_fc, h_fml_tc,
        h_fsep_tot, h_fsep_fc, h_fsep_tc,
        h3d_fml_base, h3d_fml_fc, h3d_fml_dfs,
    ) = hists

    sig_t = (0.90, 0.90, 1.00, 0.90)
    sig_f = (0.90, 0.90, 0.90, 1.00)

    st_old = gaussian_filter(h_old_tot, sigma=sig_t) + 2.5
    pt_old_tc = (gaussian_filter(h_old_tc, sigma=sig_t) / st_old).astype(np.float32)
    pt_old_fc = (gaussian_filter(h_old_fc, sigma=sig_t) / st_old).astype(np.float32)

    st_ml = gaussian_filter(h_ml_tot, sigma=sig_t) + 2.5
    pt_ml_tc = (gaussian_filter(h_ml_tc, sigma=sig_t) / st_ml).astype(np.float32)
    pt_ml_fc = (gaussian_filter(h_ml_fc, sigma=sig_t) / st_ml).astype(np.float32)

    st_mly = gaussian_filter(h_mly_tot, sigma=sig_f) + 2.5
    pt_mly_tc = (gaussian_filter(h_mly_tc, sigma=sig_f) / st_mly).astype(np.float32)

    pt_sep_tc = np.stack([
        gaussian_filter(h_sep_tc[c], sigma=sig_t) / (gaussian_filter(h_sep_tot[c], sigma=sig_t) + 2.5)
        for c in range(2)
    ]).astype(np.float32)
    pt_sep_fc = np.stack([
        gaussian_filter(h_sep_fc[c], sigma=sig_t) / (gaussian_filter(h_sep_tot[c], sigma=sig_t) + 2.5)
        for c in range(2)
    ]).astype(np.float32)

    st_wide = gaussian_filter(h_wide_tot, sigma=sig_t) + 2.5
    pt_wide_tc = (gaussian_filter(h_wide_tc, sigma=sig_t) / st_wide).astype(np.float32)

    p2d_ml_tc = (gaussian_filter(h2d_ml_tc, sigma=0.90) / (gaussian_filter(h2d_ml_base, sigma=0.90) + 2.5)).astype(np.float32)
    ht_ml_2d = np.sum(h_ml_tc, axis=(2, 3))
    edt_ml_2d = (gaussian_filter(h2d_ml_dts, sigma=0.90) / (gaussian_filter(ht_ml_2d, sigma=0.90) + 2.5)).astype(np.float32)

    p2d_sep_tc = np.stack([
        gaussian_filter(h2d_sep_tc[c], sigma=0.90) / (gaussian_filter(h2d_sep_base[c], sigma=0.90) + 2.5)
        for c in range(2)
    ]).astype(np.float32)
    ht_sep_2d = np.sum(h_sep_tc, axis=(3, 4))
    edt_sep_2d = np.stack([
        gaussian_filter(h2d_sep_dts[c], sigma=0.90) / (gaussian_filter(ht_sep_2d[c], sigma=0.90) + 2.5)
        for c in range(2)
    ]).astype(np.float32)

    p2d_wide_tc = (gaussian_filter(h2d_wide_tc, sigma=0.90) / (gaussian_filter(h2d_wide_base, sigma=0.90) + 2.5)).astype(np.float32)
    ht_wide_2d = np.sum(h_wide_tc, axis=(2, 3))
    edt_wide_2d = (gaussian_filter(h2d_wide_dts, sigma=0.90) / (gaussian_filter(ht_wide_2d, sigma=0.90) + 2.5)).astype(np.float32)

    sf_old = gaussian_filter(h_fo_tot, sigma=sig_f) + 2.0
    pf_old_fc = (gaussian_filter(h_fo_fc, sigma=sig_f) / sf_old).astype(np.float32)
    pf_old_tc = (gaussian_filter(h_fo_tc, sigma=sig_f) / sf_old).astype(np.float32)

    sf_fml = gaussian_filter(h_fml_tot, sigma=sig_f) + 2.0
    pf_fml_fc = (gaussian_filter(h_fml_fc, sigma=sig_f) / sf_fml).astype(np.float32)
    pf_fml_tc = (gaussian_filter(h_fml_tc, sigma=sig_f) / sf_fml).astype(np.float32)

    pf_sep_fc = np.stack([
        gaussian_filter(h_fsep_fc[c], sigma=sig_f) / (gaussian_filter(h_fsep_tot[c], sigma=sig_f) + 2.0)
        for c in range(2)
    ]).astype(np.float32)
    pf_sep_tc = np.stack([
        gaussian_filter(h_fsep_tc[c], sigma=sig_f) / (gaussian_filter(h_fsep_tot[c], sigma=sig_f) + 2.0)
        for c in range(2)
    ]).astype(np.float32)

    p3d_fml_fc = (gaussian_filter(h3d_fml_fc, sigma=0.90) / (gaussian_filter(h3d_fml_base, sigma=0.90) + 2.0)).astype(np.float32)
    hf_fml_3d = np.sum(h_fml_fc, axis=3)
    edf_fml_3d = (gaussian_filter(h3d_fml_dfs, sigma=0.90) / (gaussian_filter(hf_fml_3d, sigma=0.90) + 2.0)).astype(np.float32)

    return {
        'pt_old_tc': pt_old_tc, 'pt_old_fc': pt_old_fc,
        'pt_ml_tc': pt_ml_tc, 'pt_ml_fc': pt_ml_fc,
        'pt_mly_tc': pt_mly_tc,
        'pt_sep_tc': pt_sep_tc, 'pt_sep_fc': pt_sep_fc,
        'pt_wide_tc': pt_wide_tc,
        'p2d_ml_tc': p2d_ml_tc, 'edt_ml_2d': edt_ml_2d,
        'p2d_sep_tc': p2d_sep_tc, 'edt_sep_2d': edt_sep_2d,
        'p2d_wide_tc': p2d_wide_tc, 'edt_wide_2d': edt_wide_2d,
        'pf_old_fc': pf_old_fc, 'pf_old_tc': pf_old_tc,
        'pf_fml_fc': pf_fml_fc, 'pf_fml_tc': pf_fml_tc,
        'pf_sep_fc': pf_sep_fc, 'pf_sep_tc': pf_sep_tc,
        'p3d_fml_fc': p3d_fml_fc, 'edf_fml_3d': edf_fml_3d,
    }


d_grid = np.array([-3.6, -2.8, -2.0, -1.5, -1.1, -0.75, -0.4, 0.0, 0.4, 0.75, 1.1, 1.5, 2.0, 2.8, 3.6], dtype=np.float32)
h_grid = np.array([-1.0, -0.5, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.2], dtype=np.float32)
dy_steps = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=int)
r_steps = np.array([-6.0, -3.5, -1.8, 1.8, 3.5, 6.0], dtype=np.float32)


def prepare_geometry_v6(img, sp, bm_ref, bm_alt, bm_raw, bone_name, atlas_p):
    df = fast_signed_distance_transform(bm_ref['femur'] > 0, sp)
    dt = fast_signed_distance_transform(bm_ref['tibia'] > 0, sp)
    df_alt = fast_signed_distance_transform(bm_alt['femur'] > 0, sp)
    dt_alt = fast_signed_distance_transform(bm_alt['tibia'] > 0, sp)
    df_raw = fast_signed_distance_transform(bm_raw['femur'] > 0, sp)
    dt_raw = fast_signed_distance_transform(bm_raw['tibia'] > 0, sp)

    f_stats, _, f_comp = _compute_articular_landmarks_and_stats(bm_ref['femur'], df, dt, 45.0, sp)
    t_stats, _, t_comp = _compute_articular_landmarks_and_stats(bm_ref['tibia'], dt, df, 25.0, sp)
    ml_info = compute_ml_frames(bm_ref, df, dt, sp)

    if bone_name == 'tibia':
        shell = (dt >= -0.95) & (dt <= 4.6) & (df >= -1.0) & (df <= 25.0)
    else:
        raw_shell = (df >= -0.95) & (df <= 4.7) & (dt >= -1.0) & (dt <= 56.0)
        raw_inds = np.argwhere(raw_shell)
        coords_raw = raw_inds.astype(np.float32) * np.array(sp, dtype=np.float32)
        dfs_raw = df[raw_inds[:, 0], raw_inds[:, 1], raw_inds[:, 2]]
        dts_raw = dt[raw_inds[:, 0], raw_inds[:, 1], raw_inds[:, 2]]
        bins_r, c_dict_r = compute_atlas_bins_v6(coords_raw, dfs_raw, dts_raw, f_stats, t_comp, ml_info)
        rel_f_r = c_dict_r['rel_f']
        in_box = (
            (rel_f_r[:, 0] >= -0.10) & (rel_f_r[:, 0] <= 1.10)
            & (rel_f_r[:, 1] >= -0.12) & (rel_f_r[:, 1] <= 1.06)
            & (rel_f_r[:, 2] >= -0.12) & (rel_f_r[:, 2] <= 1.10)
        )
        p_ens = 0.45 * atlas_p['pf_fml_fc'][bins_r['f_ml']] + 0.35 * atlas_p['pf_sep_fc'][bins_r['f_sep']] + 0.20 * atlas_p['pf_old_fc'][bins_r['f_old']]
        p3d = atlas_p['p3d_fml_fc'][bins_r['f_ml'][0], bins_r['f_ml'][1], bins_r['f_ml'][2]]
        keep = in_box & ((p_ens > 0.002) | (p3d > 0.004) | (dts_raw <= 42.0))
        shell = np.zeros_like(raw_shell, dtype=bool)
        kept_inds = raw_inds[keep]
        shell[kept_inds[:, 0], kept_inds[:, 1], kept_inds[:, 2]] = True

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
    m_high = (d_self_crop >= 0.4) & (d_self_crop <= 2.0) & (d_other_crop <= (12.0 if bone_name == 'femur' else 10.0))
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

    # Subchondral Bone-Internal Local Z-Score (z_bone_crop):
    # Distinguishes deep radial cartilage (z_bone = +2.5..+4.0) from subchondral bone (z_bone = 0.0 +- 1.0)
    # even when deep radial cartilage is darker than superficial cartilage (e.g. image-032, 036, 018, 082, 087)!
    img_lat_raw = gaussian_filter(img_f32, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
    sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
    w_bn = ((d_self_crop <= -1.4) & (d_self_crop >= -5.5)).astype(np.float32) + 1e-5
    w_bn_s = gaussian_filter(w_bn, sigma=sig_bn)
    i_bn_mean = gaussian_filter(img_lat_raw * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_var  = gaussian_filter(((img_lat_raw - i_bn_mean) ** 2) * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_std  = np.maximum(np.sqrt(np.maximum(i_bn_var, 0.0)), max(0.05 * (i_high - i_low), 2.5))
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
    d_s_m   = geo['ds_self_crop'][zc, yc, xc]
    do_raw_m = geo['d_other_crop'][zc, yc, xc]
    do_s_m   = geo['ds_other_crop'][zc, yc, xc]

    # Subchondral bone shift indicators (invariant to common shift delta_d!)
    shift_ref_alt = d_raw_m - d_alt_m
    shift_ref_st2 = d_raw_m - d_st2_m

    if rng_jit is not None:
        if leak_free_jit:
            # Leak-free +-0.40 mm subchondral jitter applied consistently across ALL self-bone distances!
            jit = rng_jit.uniform(-0.40, 0.40, size=N).astype(np.float32)
            w_jit = np.clip(1.0 - np.abs(d_raw_m) / 1.4, 0.15, 1.0)
            delta_d = w_jit * jit
            d_raw_feat = d_raw_m + delta_d
            d_alt_feat = d_alt_m + delta_d
            d_s_feat   = d_s_m + 0.85 * delta_d
            d_s_anchor = d_s_feat
        else:
            jit = rng_jit.uniform(-0.28, 0.28, size=N).astype(np.float32)
            w_jit = np.clip(1.0 - np.abs(d_raw_m) / 1.0, 0.0, 1.0)
            d_raw_feat = d_raw_m + w_jit * jit
            d_alt_feat = d_alt_m + w_jit * jit
            d_s_feat   = d_s_m + 0.5 * w_jit * jit
            d_s_anchor = d_s_m
    else:
        d_raw_feat = d_raw_m
        d_alt_feat = d_alt_m
        d_s_feat   = d_s_m
        d_s_anchor = d_s_m

    dfs_for_bins = d_s_anchor if bone_name == 'femur' else do_s_m
    dts_for_bins = d_s_anchor if bone_name == 'tibia' else do_s_m
    ap = geo['atlas_p']
    bins, c_dict = compute_atlas_bins_v6(coords_mm, dfs_for_bins, dts_for_bins, geo['f_stats'], geo['t_comp'], geo['ml_info'])

    # Extract Multi-Atlas Priors (Old + ML-Oriented + Condyle-Separated + MLY + Unclamped Wide + 2D/3D)
    pt_old_tc  = ap['pt_old_tc'][bins['t_old']]
    pt_old_fc  = ap['pt_old_fc'][bins['t_old']]
    pt_ml_tc   = ap['pt_ml_tc'][bins['t_ml']]
    pt_ml_fc   = ap['pt_ml_fc'][bins['t_ml']]
    pt_mly_tc  = ap['pt_mly_tc'][bins['t_mly']]
    pt_sep_tc  = ap['pt_sep_tc'][bins['t_sep']]
    pt_sep_fc  = ap['pt_sep_fc'][bins['t_sep']]
    pt_wide_tc = ap['pt_wide_tc'][bins['t_wide']]

    pt_ens_tc = 0.32 * pt_ml_tc + 0.30 * pt_sep_tc + 0.20 * pt_mly_tc + 0.18 * pt_wide_tc
    pt_ens_fc = 0.50 * pt_ml_fc + 0.50 * pt_sep_fc
    bdt_m1, bdt_p1 = np.clip(bins['t_ml'][2] - 1, 0, 15), np.clip(bins['t_ml'][2] + 1, 0, 15)
    pt_ml_max3 = np.maximum(
        np.maximum(ap['pt_ml_tc'][bins['t_ml'][0], bins['t_ml'][1], bdt_m1, bins['t_ml'][3]], pt_ml_tc),
        ap['pt_ml_tc'][bins['t_ml'][0], bins['t_ml'][1], bdt_p1, bins['t_ml'][3]]
    )
    p2d_ml_v   = ap['p2d_ml_tc'][bins['t_ml'][0], bins['t_ml'][1]]
    edt_ml_v   = ap['edt_ml_2d'][bins['t_ml'][0], bins['t_ml'][1]]
    p2d_sep_v  = ap['p2d_sep_tc'][bins['t_sep'][0], bins['t_sep'][1], bins['t_sep'][2]]
    edt_sep_v  = ap['edt_sep_2d'][bins['t_sep'][0], bins['t_sep'][1], bins['t_sep'][2]]
    p2d_wide_v = ap['p2d_wide_tc'][bins['t_wide'][0], bins['t_wide'][1]]
    edt_wide_v = ap['edt_wide_2d'][bins['t_wide'][0], bins['t_wide'][1]]
    p2d_ens    = 0.40 * p2d_ml_v + 0.35 * p2d_sep_v + 0.25 * p2d_wide_v
    edt_ens    = 0.40 * edt_ml_v + 0.35 * edt_sep_v + 0.25 * edt_wide_v

    pf_old_fc  = ap['pf_old_fc'][bins['f_old']]
    pf_old_tc  = ap['pf_old_tc'][bins['f_old']]
    pf_fml_fc  = ap['pf_fml_fc'][bins['f_ml']]
    pf_fml_tc  = ap['pf_fml_tc'][bins['f_ml']]
    pf_sep_fc  = ap['pf_sep_fc'][bins['f_sep']]
    pf_sep_tc  = ap['pf_sep_tc'][bins['f_sep']]
    pf_ens_fc  = 0.45 * pf_fml_fc + 0.35 * pf_sep_fc + 0.20 * pf_old_fc
    pf_ens_tc  = 0.50 * pf_fml_tc + 0.50 * pf_sep_tc

    bdf_m1, bdf_p1 = np.clip(bins['f_ml'][3] - 1, 0, 15), np.clip(bins['f_ml'][3] + 1, 0, 15)
    pf_fml_max3 = np.maximum(
        np.maximum(ap['pf_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2], bdf_m1], pf_fml_fc),
        ap['pf_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2], bdf_p1]
    )
    p3d_fml_v = ap['p3d_fml_fc'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2]]
    edf_fml_v = ap['edf_fml_3d'][bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2]]

    img_f32 = geo['img_f32']
    i_low = geo['i_low']
    i_high = geo['i_high']
    if pm_self is not None:
        pm_crop = pm_self[z0:z1, y0:y1, x0:x1]
        m_p1_core = pm_crop[..., 2] >= 0.55
        if np.sum(m_p1_core) >= 50:
            i_high = float(np.median(img_f32[m_p1_core]))
    inorm = (img_f32 - i_low) / max(i_high - i_low, 1.0)
    z_cart = geo['z_cart']
    z_bone = geo['z_bone_crop']
    abs_z = gaussian_filter(np.abs(z_cart), sigma=[1.0 / sp[0], 0.35 / sp[1], 1.0 / sp[2]])

    img_f0  = gaussian_filter(inorm, sigma=[0.35 / s for s in sp])
    img_lat = gaussian_filter(inorm, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
    img_s1  = gaussian_filter(inorm, sigma=[1.0 / s for s in sp])
    img_s2  = gaussian_filter(inorm, sigma=[2.0 / s for s in sp])
    img_s4  = gaussian_filter(inorm, sigma=[4.0 / s for s in sp])
    img_s8  = gaussian_filter(inorm, sigma=[8.0 / s for s in sp])

    gz0, gy0, gx0 = np.gradient(img_f0, *sp)
    gmag0 = np.sqrt(gz0**2 + gy0**2 + gx0**2)
    gz1, gy1, gx1 = np.gradient(img_s1, *sp)
    gmag1 = np.sqrt(gz1**2 + gy1**2 + gx1**2)

    nz, ny, nx = geo['normals_self']
    nzo, nyo, nxo = geo['normals_other']
    nz_m, ny_m, nx_m = nz[zc, yc, xc], ny[zc, yc, xc], nx[zc, yc, xc]
    nzo_m, nyo_m, nxo_m = nzo[zc, yc, xc], nyo[zc, yc, xc], nxo[zc, yc, xc]
    cz_max, cy_max, cx_max = img_f32.shape

    K = len(d_grid)
    rv0    = np.empty((N, K), dtype=np.float32)
    rv_lat = np.empty((N, K), dtype=np.float32)
    rv1    = np.empty((N, K), dtype=np.float32)
    rv_zb  = np.empty((N, K), dtype=np.float32)
    for k, d_val in enumerate(d_grid):
        sz = np.clip(np.round(zc + d_val * nz_m / sp[0]).astype(int), 0, cz_max - 1)
        sy = np.clip(np.round(yc + d_val * ny_m / sp[1]).astype(int), 0, cy_max - 1)
        sx = np.clip(np.round(xc + d_val * nx_m / sp[2]).astype(int), 0, cx_max - 1)
        rv0[:, k]    = img_f0[sz, sy, sx]
        rv_lat[:, k] = img_lat[sz, sy, sx]
        rv1[:, k]    = img_s1[sz, sy, sx]
        rv_zb[:, k]  = z_bone[sz, sy, sx]

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
    # Subchondral cortical dark trough & z_bone rising edge along normal ray!
    idx_cort_min = np.argmin(rv_zb[:, 3:10], axis=1).astype(np.float32) - 4.0
    val_cort_min = np.min(rv_zb[:, 3:10], axis=1)
    idx_rise_zb  = np.argmax(rg_zb[:, 2:11], axis=1).astype(np.float32) - 4.0
    max_rise_zb  = np.max(rg_zb[:, 2:11], axis=1)

    # Explicit 9-point Vertical (Y-axis) Column Profile of Intensity & Subchondral Z-Bone!
    vy_lat = np.empty((N, len(dy_steps)), dtype=np.float32)
    vy_zb  = np.empty((N, len(dy_steps)), dtype=np.float32)
    for k, dy_px in enumerate(dy_steps):
        sy = np.clip(yc + dy_px, 0, cy_max - 1)
        vy_lat[:, k] = img_lat[zc, sy, xc]
        vy_zb[:, k]  = z_bone[zc, sy, xc]
    vyn_lat = (vy_lat - r_in) / r_span
    vyg_lat = np.diff(vy_lat, axis=1)

    # Surface-Anchored Height Profile (anchored at d_s_anchor so leak_free_jit=True has ZERO unjittered dt leakage!)
    vh_lat = np.empty((N, len(h_grid)), dtype=np.float32)
    vh_zb  = np.empty((N, len(h_grid)), dtype=np.float32)
    for k, h_val in enumerate(h_grid):
        st = h_val - d_s_anchor
        sz = np.clip(np.round(zc + st * nz_m / sp[0]).astype(int), 0, cz_max - 1)
        sy = np.clip(np.round(yc + st * ny_m / sp[1]).astype(int), 0, cy_max - 1)
        sx = np.clip(np.round(xc + st * nx_m / sp[2]).astype(int), 0, cx_max - 1)
        vh_lat[:, k] = img_lat[sz, sy, sx]
        vh_zb[:, k]  = z_bone[sz, sy, sx]
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
    rad_vlat = np.empty((N, len(r_steps)), dtype=np.float32)
    rad_zb   = np.empty((N, len(r_steps)), dtype=np.float32)
    for k, r_mm in enumerate(r_steps):
        sz = np.clip(np.round(zc + r_mm * u_rz / sp[0]).astype(int), 0, cz_max - 1)
        sx = np.clip(np.round(xc + r_mm * u_rx / sp[2]).astype(int), 0, cx_max - 1)
        rad_vlat[:, k] = img_lat[sz, yc, sx]
        rad_zb[:, k]   = z_bone[sz, yc, sx]
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
        # Consistent jittered distances + subchondral shift indicators
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
        # Multi-atlas priors (ML-oriented + Condyle-Separated + Unclamped Wide + Old)
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

        # Spatially-Varying Local Sheet-Calibrated Intensity (from Pass-1 cartilage core!)
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

        # 1) Vertical 2D Sheet Cumulative Cartilage Thickness + Normal-Weighted Cumulative Thickness (fixes posterior wall!)
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

        # 2) Normal ray profile of Pass-1 probabilities + 3D Normal-Ray Cumulative Thickness (100% orientation-invariant!)
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

        c_in_ray  = np.sum(pr_ct[:, 0:5], axis=1) * 0.40
        c_out_ray = np.sum(pr_ct[:, 5:10], axis=1) * 0.42
        t_ray_tot = c_in_ray + c_out_ray
        f_in_ray  = c_in_ray / (t_ray_tot + 0.20)

        # 3) Vertical (Y) 9-point Pass-1 probability column profile
        py_ct = np.empty((N, len(dy_steps)), dtype=np.float32)
        py_bn = np.empty((N, len(dy_steps)), dtype=np.float32)
        for k, dy_px in enumerate(dy_steps):
            sy = np.clip(yc + dy_px, 0, cy_max - 1)
            py_ct[:, k] = p_ct_lat[zc, sy, xc]
            py_bn[:, k] = p_bn_c[zc, sy, xc]

        # 4) Surface-anchored column profile of Pass-1 cartilage probability (anchored at jittered d_s_anchor!)
        ph_ct = np.empty((N, len(h_grid)), dtype=np.float32)
        ph_ot = np.empty((N, len(h_grid)), dtype=np.float32)
        for k, h_val in enumerate(h_grid):
            st = h_val - d_s_anchor
            sz = np.clip(np.round(zc + st * nz_m / sp[0]).astype(int), 0, cz_max - 1)
            sy = np.clip(np.round(yc + st * ny_m / sp[1]).astype(int), 0, cy_max - 1)
            sx = np.clip(np.round(xc + st * nx_m / sp[2]).astype(int), 0, cx_max - 1)
            ph_ct[:, k] = p_ct_lat[sz, sy, sx]
            ph_ot[:, k] = p_ot_s1[sz, sy, sx]
        col_psum = np.sum(ph_ct[:, 2:], axis=1) * 0.45
        col_pmax = np.max(ph_ct[:, 2:], axis=1)
        col_otsum = np.sum(ph_ot[:, 2:], axis=1) * 0.45

        # 5) 8-point Tangential sheet probability profile
        t1z = 1.0 - nz_m * nz_m
        t1y = -nz_m * ny_m
        t1x = -nz_m * nx_m
        t1_norm = np.sqrt(t1z**2 + t1y**2 + t1x**2) + 1e-6
        t1z, t1y, t1x = t1z / t1_norm, t1y / t1_norm, t1x / t1_norm
        t2z = ny_m * t1x - nx_m * t1y
        t2y = nx_m * t1z - nz_m * t1x
        t2x = nz_m * t1y - ny_m * t1z
        tan_offsets = [(-3.6, 0.0), (-1.8, 0.0), (1.8, 0.0), (3.6, 0.0),
                       (0.0, -3.6), (0.0, -1.8), (0.0, 1.8), (0.0, 3.6)]
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


def make_stratified_sample_v6(lbl, geo, rng):
    bone_name = geo['bone_name']
    shell = geo['shell']
    z0, z1, y0, y1, x0, x1 = geo['bbox']
    sp = geo['sp']
    b_lbl = 1 if bone_name == 'femur' else 3
    c_lbl = 2 if bone_name == 'femur' else 4
    o_lbl = 4 if bone_name == 'femur' else 2

    dc = fast_signed_distance_transform(lbl == c_lbl, sp)
    d_self = np.full(lbl.shape, 100.0, dtype=np.float32)
    d_self[z0:z1, y0:y1, x0:x1] = geo['d_self_crop']

    y4 = np.zeros_like(lbl, dtype=np.uint8)
    y4[lbl == b_lbl] = 1
    y4[lbl == c_lbl] = 2
    y4[lbl == o_lbl] = 3

    p_base_c  = np.argwhere(shell & (y4 == 2) & (d_self <= 0.60))
    p_core_c  = np.argwhere(shell & (y4 == 2) & (d_self > 0.60))
    p_surf_b  = np.argwhere(shell & (y4 == 1) & (d_self >= -0.15) & (dc <= 2.2))
    p_deep_b  = np.argwhere(shell & (y4 == 1) & (d_self < -0.15) & (dc <= 2.2))
    p_hard_bg = np.argwhere(shell & (y4 == 0) & (dc <= 1.8))
    p_outer_bg = np.argwhere(shell & (y4 == 0) & (dc > 1.8))
    p_other_c = np.argwhere(shell & (y4 == 3) & (d_self <= 3.5))

    pools = [
        (p_base_c, 2000),
        (p_core_c, 3600),
        (p_surf_b, 2000),
        (p_deep_b, 1200),
        (p_hard_bg, 2900),
        (p_outer_bg, 1100),
        (p_other_c, 900),
    ]
    sm = np.zeros_like(shell, dtype=bool)
    for pool, n_take in pools:
        if len(pool) > 0:
            chosen = pool[rng.choice(len(pool), min(len(pool), n_take), replace=False)]
            sm[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
    return sm, y4[sm]


def predict_p1_fast_strided_v6(geo, clf, sm_exact=None, p_exact=None):
    """Boundary-safe (1, 1, 2) X-strided Pass-1 probability volume (full Z & Y resolution!) + exact values at sm_exact."""
    sh = geo['shell']
    sub_mask = np.zeros_like(sh, dtype=bool)
    sub_mask[:, :, ::2] = sh[:, :, ::2]
    n_x = sh.shape[2]
    even_lim = n_x - 1
    odd_only = np.zeros_like(sh, dtype=bool)
    odd_only[:, :, 1::2] = sh[:, :, 1::2] & (~sh[:, :, 0:even_lim:2])
    sub_mask |= odd_only

    X_sub = extract_features_v6(geo, sub_mask, rng_jit=None)
    p_sub = clf.predict_proba(X_sub).astype(np.float32)

    pm = np.zeros((*sh.shape, 4), dtype=np.float32)
    pm[..., 0] = 1.0
    zs, ys, xs = np.where(sub_mask)
    for dx in (0, 1):
        x_f = np.minimum(xs + dx, sh.shape[2] - 1)
        pm[zs, ys, x_f] = p_sub
    pm[~sh] = [1.0, 0.0, 0.0, 0.0]
    if sm_exact is not None and p_exact is not None:
        pm[sm_exact] = p_exact
    return pm


def train_and_eval_pipeline(tag, tr_cases, ev_cases, use_v2_as_primary=True, leak_free_jit=True):
    t_pipe = time.time()
    print(f"\n================================================================================", flush=True)
    print(f"Starting Pipeline [{tag}] (use_v2_as_primary={use_v2_as_primary}, leak_free_jit={leak_free_jit})", flush=True)
    print(f"================================================================================", flush=True)

    # 1. Build 80-case Leave-One-Out (LOO) Multi-Atlases on primary bone mask
    t1 = time.time()
    case_hists = Parallel(n_jobs=16, prefer='threads')(
        delayed(compute_case_atlas_hist_v6)(img, bm_v2 if use_v2_as_primary else bm_v1, sp, lbl)
        for fname, img, lbl, sp, bm_v1, bm_v2, bm_raw in tr_cases
    )
    n_h = len(case_hists[0])
    tot_hists = [sum(h[k] for h in case_hists) for k in range(n_h)]
    atlas_priors = make_atlas_priors_v6(tot_hists)
    loo_atlases = [make_atlas_priors_v6([tot_hists[k] - h[k] for k in range(n_h)]) for h in case_hists]
    del case_hists, tot_hists; gc.collect()
    print(f"[{tag}] Built 80-case LOO Multi-Atlases in {time.time()-t1:.1f}s", flush=True)

    # 2. Step 1: Sample P1 features and fit 4-Fold Out-Of-Fold P1 HGB (60 cases/fold) for BOTH Tibia and Femur
    p1_folds = {}
    for bone_name in ['tibia', 'femur']:
        t_b = time.time()
        def sample_p1(i):
            fname, img, lbl, sp, bm_v1, bm_v2, bm_raw = tr_cases[i]
            bm_p = bm_v2 if use_v2_as_primary else bm_v1
            bm_a = bm_v1 if use_v2_as_primary else bm_v2
            geo = prepare_geometry_v6(img, sp, bm_p, bm_a, bm_raw, bone_name, loo_atlases[i])
            sm, y_s = make_stratified_sample_v6(lbl, geo, np.random.RandomState(42 + i))
            X_s = extract_features_v6(geo, sm, rng_jit=np.random.RandomState(142 + i), leak_free_jit=leak_free_jit)
            return X_s, y_s

        res_p1 = Parallel(n_jobs=16, prefer='threads')(delayed(sample_p1)(i) for i in range(len(tr_cases)))
        folds_b = []
        for f_k in range(4):
            s_f, e_f = f_k * 20, (f_k + 1) * 20
            tr_idx = [i for i in range(80) if not (s_f <= i < e_f)]
            X_f = np.vstack([res_p1[i][0] for i in tr_idx])
            y_f = np.concatenate([res_p1[i][1] for i in tr_idx])
            clf_f = HistGradientBoostingClassifier(
                max_iter=240, max_leaf_nodes=63, min_samples_leaf=25,
                learning_rate=0.055, max_features=0.70, l2_regularization=1.0,
                early_stopping=False, random_state=42 + f_k
            ).fit(X_f, y_f)
            folds_b.append(clf_f)
            del X_f, y_f; gc.collect()
        del res_p1; gc.collect()
        p1_folds[bone_name] = tuple(folds_b)
        print(f"[{tag} - {bone_name.upper()}] Sampled & Fit 4-Fold P1 HGB in {time.time()-t_b:.1f}s", flush=True)

    # 3. Step 2: Generate OOF P1 probability maps (full-res for Tibia, (1,1,2) ZY-exact for Femur) & extract P2 training set
    t_s2 = time.time()
    Xt2_list, yt2_list = [], []
    Xf2_list, yf2_list = [], []

    for start in range(0, len(tr_cases), 16):
        idxs = list(range(start, min(start + 16, len(tr_cases))))
        def _proc_tr_p2(i):
            fname, img, lbl, sp, bm_v1, bm_v2, bm_raw = tr_cases[i]
            bm_p = bm_v2 if use_v2_as_primary else bm_v1
            bm_a = bm_v1 if use_v2_as_primary else bm_v2
            geo_t = prepare_geometry_v6(img, sp, bm_p, bm_a, bm_raw, 'tibia', loo_atlases[i])
            geo_f = prepare_geometry_v6(img, sp, bm_p, bm_a, bm_raw, 'femur', loo_atlases[i])

            clf_t = p1_folds['tibia'][i // 20]
            clf_f = p1_folds['femur'][i // 20]

            # Full-res OOF P1 for Tibia!
            Xt_full = extract_features_v6(geo_t, geo_t['shell'], rng_jit=None)
            pt_oof = clf_t.predict_proba(Xt_full).astype(np.float32)
            del Xt_full
            pm_t = np.zeros((*img.shape, 4), dtype=np.float32); pm_t[..., 0] = 1.0
            pm_t[geo_t['shell']] = pt_oof

            # Boundary-safe (1, 1, 2) OOF P1 for Femur + exact at sm_f!
            sm_f, y_f = make_stratified_sample_v6(lbl, geo_f, np.random.RandomState(742 + i))
            Xf_un = extract_features_v6(geo_f, sm_f, rng_jit=None)
            pf_ex = clf_f.predict_proba(Xf_un).astype(np.float32)
            del Xf_un
            pm_f = predict_p1_fast_strided_v6(geo_f, clf_f, sm_f, pf_ex)

            sm_t, y_t = make_stratified_sample_v6(lbl, geo_t, np.random.RandomState(542 + i))
            Xt_2 = extract_features_v6(geo_t, sm_t, pm_self=pm_t, pm_other=pm_f, rng_jit=np.random.RandomState(642 + i), leak_free_jit=leak_free_jit)
            Xf_2 = extract_features_v6(geo_f, sm_f, pm_self=pm_f, pm_other=pm_t, rng_jit=np.random.RandomState(842 + i), leak_free_jit=leak_free_jit)
            return Xt_2, y_t, Xf_2, y_f

        batch = Parallel(n_jobs=len(idxs), prefer='threads')(delayed(_proc_tr_p2)(i) for i in idxs)
        for xt, yt, xf, yf in batch:
            Xt2_list.append(xt); yt2_list.append(yt)
            Xf2_list.append(xf); yf2_list.append(yf)
        del batch; gc.collect()
        print(f"  [{tag}] Extracted P2 OOF train cases {min(start+16, 80)}/80 in {time.time()-t_s2:.1f}s", flush=True)

    # 4. Step 3: Fit 4-Fold OOF Pass-2 HGBs + Full-80 Pass-2 Multi-Family Ensemble (3x HGB + 1x ExtraTrees) for Tibia and Femur
    p2_models = {}
    p2_oof_folds = {}
    for bone_name, X_list, y_list, seed_base in [
        ('tibia', Xt2_list, yt2_list, 100),
        ('femur', Xf2_list, yf2_list, 200),
    ]:
        t_fit2 = time.time()
        oof_b = []
        for f_k in range(4):
            s_f, e_f = f_k * 20, (f_k + 1) * 20
            tr_idx = [i for i in range(80) if not (s_f <= i < e_f)]
            X_f = np.vstack([X_list[i] for i in tr_idx])
            y_f = np.concatenate([y_list[i] for i in tr_idx])
            h_f1 = HistGradientBoostingClassifier(
                max_iter=250, max_leaf_nodes=63, min_samples_leaf=30,
                learning_rate=0.05, max_features=0.60, l2_regularization=1.0,
                early_stopping=False, random_state=seed_base + 10 + f_k
            ).fit(X_f, y_f)
            if bone_name == 'tibia':
                h_f2 = HistGradientBoostingClassifier(
                    max_iter=250, max_leaf_nodes=79, min_samples_leaf=35,
                    learning_rate=0.05, max_features=0.75, l2_regularization=1.2,
                    early_stopping=False, random_state=seed_base + 20 + f_k
                ).fit(X_f, y_f)
                oof_b.append((h_f1, h_f2))
            else:
                oof_b.append((h_f1,))
            del X_f, y_f; gc.collect()
        p2_oof_folds[bone_name] = tuple(oof_b)
        print(f"[{tag} - {bone_name.upper()}] Fit 4-Fold OOF P2 HGB in {time.time()-t_fit2:.1f}s", flush=True)

        t_fit_full = time.time()
        X2_tr = np.vstack(X_list)
        y2_tr = np.concatenate(y_list)
        X_list.clear(); y_list.clear(); gc.collect()

        hgb2_1 = HistGradientBoostingClassifier(
            max_iter=260, max_leaf_nodes=63, min_samples_leaf=30,
            learning_rate=0.05, max_features=0.55, l2_regularization=1.0,
            early_stopping=False, random_state=seed_base + 1
        ).fit(X2_tr, y2_tr)
        hgb2_2 = HistGradientBoostingClassifier(
            max_iter=260, max_leaf_nodes=79, min_samples_leaf=35,
            learning_rate=0.05, max_features=0.70, l2_regularization=1.0,
            early_stopping=False, random_state=seed_base + 2
        ).fit(X2_tr, y2_tr)
        hgb2_3 = HistGradientBoostingClassifier(
            max_iter=260, max_leaf_nodes=95, min_samples_leaf=40,
            learning_rate=0.045, max_features=0.85, l2_regularization=1.5,
            early_stopping=False, random_state=seed_base + 3
        ).fit(X2_tr, y2_tr)

        rng_et = np.random.RandomState(seed_base + 77)
        et_idx = rng_et.choice(len(y2_tr), min(len(y2_tr), 450000), replace=False)
        et2 = ExtraTreesClassifier(
            n_estimators=200, max_depth=26, min_samples_leaf=10,
            max_features='sqrt', n_jobs=16, random_state=seed_base + 4
        ).fit(X2_tr[et_idx], y2_tr[et_idx])
        del X2_tr, y2_tr; gc.collect()
        p2_models[bone_name] = (hgb2_1, hgb2_2, hgb2_3, et2)
        print(f"[{tag} - {bone_name.upper()}] Fit Full-80 3xHGB + ExtraTrees P2 ensemble in {time.time()-t_fit_full:.1f}s", flush=True)

    # Save trained models (including 4-fold OOF P2 HGBs and ExtraTrees)
    joblib.dump({
        'atlas_priors': atlas_priors,
        'p1_folds': p1_folds,
        'p2_tibia_hgb': p2_models['tibia'][:3],
        'p2_femur_hgb': p2_models['femur'][:3],
        'p2_tibia_et': p2_models['tibia'][3],
        'p2_femur_et': p2_models['femur'][3],
        'p2_tibia_oof_folds': p2_oof_folds['tibia'],
        'p2_femur_oof_folds': p2_oof_folds['femur'],
    }, os.path.join(model_dir, f'cart_{tag}_models.joblib'), compress=3)

    # 5. Step 4: Full-Resolution Evaluation on all 20 held-out eval cases (saving compact cropped ROIs!)
    t_ev = time.time()
    def pred_eval_one(i):
        fname, img, lbl, sp, bm_v1, bm_v2, bm_raw = ev_cases[i]
        bm_p = bm_v2 if use_v2_as_primary else bm_v1
        bm_a = bm_v1 if use_v2_as_primary else bm_v2
        geo_t = prepare_geometry_v6(img, sp, bm_p, bm_a, bm_raw, 'tibia', atlas_priors)
        geo_f = prepare_geometry_v6(img, sp, bm_p, bm_a, bm_raw, 'femur', atlas_priors)

        X1_t = extract_features_v6(geo_t, geo_t['shell'], rng_jit=None)
        p1_t = 0.25 * sum(m.predict_proba(X1_t) for m in p1_folds['tibia']).astype(np.float32)
        del X1_t
        pm1_t = np.zeros((*img.shape, 4), dtype=np.float32); pm1_t[..., 0] = 1.0
        pm1_t[geo_t['shell']] = p1_t

        X1_f = extract_features_v6(geo_f, geo_f['shell'], rng_jit=None)
        p1_f = 0.25 * sum(m.predict_proba(X1_f) for m in p1_folds['femur']).astype(np.float32)
        del X1_f
        pm1_f = np.zeros((*img.shape, 4), dtype=np.float32); pm1_f[..., 0] = 1.0
        pm1_f[geo_f['shell']] = p1_f

        X2_t = extract_features_v6(geo_t, geo_t['shell'], pm_self=pm1_t, pm_other=pm1_f, rng_jit=None)
        h1_t, h2_t, h3_t, et_t = p2_models['tibia']
        p2_t_hgb = ((h1_t.predict_proba(X2_t) + h2_t.predict_proba(X2_t) + h3_t.predict_proba(X2_t)) / 3.0).astype(np.float32)
        p2_t_et  = et_t.predict_proba(X2_t).astype(np.float32)
        del X2_t

        X2_f = extract_features_v6(geo_f, geo_f['shell'], pm_self=pm1_f, pm_other=pm1_t, rng_jit=None)
        h1_f, h2_f, h3_f, et_f = p2_models['femur']
        p2_f_hgb = ((h1_f.predict_proba(X2_f) + h2_f.predict_proba(X2_f) + h3_f.predict_proba(X2_f)) / 3.0).astype(np.float32)
        p2_f_et  = et_f.predict_proba(X2_f).astype(np.float32)
        del X2_f

        # Common joint ROI bounding box covering both shells
        union_sh = geo_t['shell'] | geo_f['shell']
        inds_u = np.argwhere(union_sh)
        uz0, uy0, ux0 = np.maximum(inds_u.min(axis=0) - 4, 0)
        uz1, uy1, ux1 = np.minimum(inds_u.max(axis=0) + 5, img.shape)
        sl = (slice(uz0, uz1), slice(uy0, uy1), slice(ux0, ux1))

        def _pack(sh, arr):
            vol = np.zeros((*img.shape, 4), dtype=np.float32); vol[..., 0] = 1.0
            vol[sh] = arr
            return vol[sl].astype(np.float16)

        return i, {
            'fname': fname,
            'bbox': (uz0, uz1, uy0, uy1, ux0, ux1),
            'shape': img.shape,
            't_p1': pm1_t[sl].astype(np.float16),
            't_p2_hgb': _pack(geo_t['shell'], p2_t_hgb),
            't_p2_et':  _pack(geo_t['shell'], p2_t_et),
            'f_p1': pm1_f[sl].astype(np.float16),
            'f_p2_hgb': _pack(geo_f['shell'], p2_f_hgb),
            'f_p2_et':  _pack(geo_f['shell'], p2_f_et),
        }

    ev_res = {}
    for start in range(0, 20, 10):
        idxs = list(range(start, min(start + 10, 20)))
        batch = Parallel(n_jobs=len(idxs), prefer='threads')(delayed(pred_eval_one)(i) for i in idxs)
        for i, d in batch:
            ev_res[i] = d
    joblib.dump(ev_res, os.path.join(cache_dir, f'ev_{tag}_rois.joblib'), compress=3)
    print(f"[{tag}] Predicted 20 eval cases and saved compact ROIs in {time.time()-t_ev:.1f}s (Total pipeline: {time.time()-t_pipe:.1f}s)", flush=True)
    return atlas_priors, ev_res


if __name__ == '__main__':
    t0 = time.time()
    train_v1  = joblib.load(os.path.join(cache_dir, 'train80_ref_bones.joblib'))
    eval_v1   = joblib.load(os.path.join(cache_dir, 'eval_ref_bones_80.joblib'))
    train_v2  = joblib.load(os.path.join(cache_dir, 'train80_ref_bones_v2.joblib'))
    eval_v2   = joblib.load(os.path.join(cache_dir, 'eval_ref_bones_v2.joblib'))
    train_raw = joblib.load(os.path.join(cache_dir, 'train80_pred_bones.joblib'))
    eval_raw  = joblib.load(os.path.join(cache_dir, 'eval_pred_bones.joblib'))

    def load_case_all(fname, d_v1, d_v2, d_raw):
        img, sp = load_volume(os.path.join(data_dir, fname), return_spacing=True)
        lbl = load_volume(os.path.join(lbl_dir, fname.replace('image-', 'labels-')), return_spacing=False).astype(np.uint8)
        lbl[lbl == 5] = 1
        lbl[lbl == 6] = 3
        return fname, img, lbl, sp, d_v1[fname], d_v2[fname], d_raw[fname]

    tr_cases = Parallel(n_jobs=16, prefer='threads')(delayed(load_case_all)(f, train_v1, train_v2, train_raw) for f in split['train'])
    ev_cases = Parallel(n_jobs=16, prefer='threads')(delayed(load_case_all)(f, eval_v1, eval_v2, eval_raw) for f in split['eval'])
    print(f"Loaded 80 train + 20 eval cases (with V1, V2, and Raw bones) in {time.time()-t0:.1f}s", flush=True)

    # Run Pipeline V6 (Primary = RefBone V2, Leak-Free +-0.40mm Jitter)
    ap_v6, ev_v6 = train_and_eval_pipeline('v6_refv2', tr_cases, ev_cases, use_v2_as_primary=True, leak_free_jit=True)

    # Run Pipeline V5 (Primary = RefBone V1, Standard +-0.28mm Jitter)
    ap_v5, ev_v5 = train_and_eval_pipeline('v5_refv1', tr_cases, ev_cases, use_v2_as_primary=False, leak_free_jit=False)

    # Joint Evaluation of V6 standalone, V5 standalone, and V6 + V5 Multi-Bone Multi-Family Ensemble!
    print("\n================ 20-EVALUATION-CASE CARTILAGE DSC RESULTS ================", flush=True)
    # Precompute eval case crops and atlas delta terms
    ev_meta = []
    for i, (fname, img, lbl, sp, bm_v1, bm_v2, bm_raw) in enumerate(ev_cases):
        df = fast_signed_distance_transform(bm_v2['femur'] > 0, sp)
        dt = fast_signed_distance_transform(bm_v2['tibia'] > 0, sp)
        roi = (df >= -1.0) & (dt >= -1.0) & (((df <= 4.8) & (dt <= 56.0)) | ((dt <= 4.8) & (df <= 28.0)))
        inds = np.argwhere(roi)
        z0, y0, x0 = np.maximum(inds.min(axis=0) - 4, 0)
        z1, y1, x1 = np.minimum(inds.max(axis=0) + 5, df.shape)
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

        lbl_c = lbl[sl]
        df_c  = df[sl]
        dt_c  = dt[sl]
        ngf   = int(np.sum(lbl == 2))
        ngt   = int(np.sum(lbl == 4))

        # Compute ML-oriented + Wide atlas vs Old folded atlas delta on crop
        f_stats, _, _ = _compute_articular_landmarks_and_stats(bm_v2['femur'], df, dt, 45.0, sp)
        _, _, t_comp  = _compute_articular_landmarks_and_stats(bm_v2['tibia'], dt, df, 25.0, sp)
        ml_info       = compute_ml_frames(bm_v2, df, dt, sp)
        dfs_c = gaussian_filter(df_c, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
        dts_c = gaussian_filter(dt_c, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
        zz, yy, xx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
        coords_c = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(np.float32) * np.array(sp, dtype=np.float32)
        bins_c, _ = compute_atlas_bins_v6(coords_c, dfs_c.ravel(), dts_c.ravel(), f_stats, t_comp, ml_info)
        sh_c = df_c.shape
        a_old_tc = ap_v6['pt_old_tc'][bins_c['t_old']].reshape(sh_c)
        a_new_tc = (
            0.32 * ap_v6['pt_ml_tc'][bins_c['t_ml']] +
            0.30 * ap_v6['pt_sep_tc'][bins_c['t_sep']] +
            0.20 * ap_v6['pt_mly_tc'][bins_c['t_mly']] +
            0.18 * ap_v6['pt_wide_tc'][bins_c['t_wide']]
        ).reshape(sh_c)
        a_old_fc = ap_v6['pf_old_fc'][bins_c['f_old']].reshape(sh_c)
        a_new_fc = (
            0.45 * ap_v6['pf_fml_fc'][bins_c['f_ml']] +
            0.35 * ap_v6['pf_sep_fc'][bins_c['f_sep']] +
            0.20 * ap_v6['pf_old_fc'][bins_c['f_old']]
        ).reshape(sh_c)

        def _unpack_to_sl(ev_d, key):
            uz0, uz1, uy0, uy1, ux0, ux1 = ev_d['bbox']
            full = np.zeros(ev_d['shape'], dtype=np.float32)
            if key.endswith('_bg'):
                full[...] = 1.0
            full[uz0:uz1, uy0:uy1, ux0:ux1] = ev_d[key].astype(np.float32)
            return full[sl]

        def _get_smoothed(ev_d, w_et):
            pm_t = 0.10 * ev_d['t_p1'].astype(np.float32) + (0.90 * (1.0 - w_et)) * ev_d['t_p2_hgb'].astype(np.float32) + (0.90 * w_et) * ev_d['t_p2_et'].astype(np.float32)
            pm_f = 0.10 * ev_d['f_p1'].astype(np.float32) + (0.90 * (1.0 - w_et)) * ev_d['f_p2_hgb'].astype(np.float32) + (0.90 * w_et) * ev_d['f_p2_et'].astype(np.float32)
            uz0, uz1, uy0, uy1, ux0, ux1 = ev_d['bbox']
            sig_f = [0.25 / sp[0], 0.10 / sp[1], 0.18 / sp[2]]
            sig_t = [0.35 / sp[0], 0.14 / sp[1], 0.25 / sp[2]]
            pf_u  = gaussian_filter(pm_f[..., 2], sigma=sig_f)
            pfb_u = gaussian_filter(pm_f[..., 1], sigma=sig_f)
            pt_u  = gaussian_filter(pm_t[..., 2], sigma=sig_t)
            ptb_u = gaussian_filter(pm_t[..., 1], sigma=sig_t)
            out = {}
            for name, arr in [('pf', pf_u), ('pfb', pfb_u), ('pt', pt_u), ('ptb', ptb_u)]:
                vol = np.zeros(img.shape, dtype=np.float32)
                vol[uz0:uz1, uy0:uy1, ux0:ux1] = arr
                out[name] = vol[sl]
            return out

        d6_s = _get_smoothed(ev_v6[i], 0.18)
        d5_s = _get_smoothed(ev_v5[i], 0.18)
        ev_meta.append((fname, sp, lbl_c, df_c, dt_c, ngf, ngt, a_old_fc, a_new_fc, a_old_tc, a_new_tc, d6_s, d5_s, (z0, z1, y0, y1, x0, x1)))

    prep_uncomp = [
        {
            'fname': fname, 'sp': sp, 'sl_bbox': sl_bbox,
            'lbl_c': lbl_c, 'df_c': df_c, 'dt_c': dt_c,
            'ngf': ngf, 'ngt': ngt,
            'a_old_fc': a_old_fc, 'a_new_fc': a_new_fc,
            'a_old_tc': a_old_tc, 'a_new_tc': a_new_tc,
            'd6': d6, 'd5': d5,
        }
        for (fname, sp, lbl_c, df_c, dt_c, ngf, ngt, a_old_fc, a_new_fc, a_old_tc, a_new_tc, d6, d5, sl_bbox) in ev_meta
    ]
    joblib.dump(prep_uncomp, os.path.join(cache_dir, 'eval20_prep_uncompressed.joblib'), compress=0)
    del prep_uncomp; gc.collect()

    for w_v6 in [1.0, 0.70, 0.55, 0.45, 0.30, 0.0]:
        for b_pen in [0.20, 0.24, 0.28]:
            for w_del_f, w_del_t in [(0.0, 0.0), (0.15, 0.25), (0.20, 0.35)]:
                for th_f, th_t in [(0.48, 0.50), (0.50, 0.52)]:
                    fcs, tcs = [], []
                    for (fname, sp, lbl_c, df_c, dt_c, ngf, ngt, a_old_fc, a_new_fc, a_old_tc, a_new_tc, d6, d5, _sl_bbox) in ev_meta:
                        pf  = w_v6 * d6['pf']  + (1.0 - w_v6) * d5['pf']
                        pfb = w_v6 * d6['pfb'] + (1.0 - w_v6) * d5['pfb']
                        pt  = w_v6 * d6['pt']  + (1.0 - w_v6) * d5['pt']
                        ptb = w_v6 * d6['ptb'] + (1.0 - w_v6) * d5['ptb']

                        del_f = a_new_fc - a_old_fc
                        del_t = a_new_tc - a_old_tc
                        sf_part = pf - b_pen * pfb + w_del_f * del_f
                        st_part = pt - b_pen * ptb + w_del_t * del_t
                        sf = sf_part + 0.28 * np.minimum(pf, pt)
                        st = st_part + 0.28 * np.minimum(pf, pt)

                        pf_50 = float(np.sum((pf - b_pen * pfb) >= th_f + 0.05 * np.clip(df_c - 0.8, 0.0, 2.5)))
                        pt_52 = float(np.sum((pt - b_pen * ptb) >= th_t + 0.05 * np.clip(dt_c - 0.8, 0.0, 2.5)))
                        af_sum = float(np.sum(a_new_fc))
                        at_sum = float(np.sum(a_new_tc))
                        pf_sum = float(np.sum(pf[pf >= 0.05]))
                        pt_sum = float(np.sum(pt[pt >= 0.05]))

                        ind_f = 0.5 * ((pf_50 / pf_sum - 0.815) / 0.035) + 0.5 * ((pf_50 / af_sum - 0.98) / 0.10)
                        ind_t = 0.5 * ((pt_52 / pt_sum - 0.765) / 0.040) + 0.5 * ((pt_52 / at_sum - 0.96) / 0.12)
                        dth_f = np.clip(-0.04 * 0.60 * ind_f, -0.09, 0.09)
                        dth_t = np.clip(-0.04 * 0.45 * ind_t, -0.09, 0.09)

                        pred_fc = fast_keep_top_k((sf >= th_f + dth_f) & (sf_part >= st_part), k=1, min_size=300)
                        pred_tc = fast_keep_top_k((st >= th_t + dth_t) & (st_part > sf_part), k=2, min_size=150)
                        fcs.append(2.0 * np.sum((lbl_c == 2) & pred_fc) / (ngf + np.sum(pred_fc)))
                        tcs.append(2.0 * np.sum((lbl_c == 4) & pred_tc) / (ngt + np.sum(pred_tc)))
                    print(f"w_v6={w_v6:.2f} b_pen={b_pen:.2f} w_del=({w_del_f:.2f},{w_del_t:.2f}) th=({th_f:.2f},{th_t:.2f}) -> FC={np.mean(fcs):.5f}, TC={np.mean(tcs):.5f}", flush=True)
