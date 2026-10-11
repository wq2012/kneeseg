"""Train 3-Cartilage (Femoral, Tibial, Patellar) Subchondral Atlas & Auto-Context on OAI."""

import gc
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import joblib
from joblib import Parallel, delayed
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_EXP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from kneeseg.features import fast_signed_distance_transform
from kneeseg.io import load_volume
from kneeseg.rf_seg import _compute_articular_landmarks_and_stats
from experiments.train_oai_bones_v2 import detect_medial_is_low_z_3bone, detect_oai_layout, to_canonical_oai


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


def fast_hist_add(shape, bins, mask=None, weights=None):
    b_sub = bins if mask is None else tuple(b[mask] for b in bins)
    lin = np.ravel_multi_index(b_sub, shape)
    w_sub = None if weights is None else (weights if mask is None else weights[mask])
    return np.bincount(lin, weights=w_sub, minlength=int(np.prod(shape))).reshape(shape).astype(np.float32)


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


def compute_atlas_bins_oai_v6(coords_mm, dfs_m, dts_m, dps_m, f_stats, t_comp, p_stats, ml_info):
    med_low_z, (p_min, p_max, p_mid, p_span), (c_M, s_M, c_L, s_L, z_med_split) = ml_info

    # 1. Symmetric Tibia Condyle Frame
    c1_t, s1_t, c2_t, s2_t, spine_t, z_mid_t = t_comp
    in_1t = coords_mm[:, 0] < z_mid_t
    uz_t = np.where(in_1t, (c1_t[0] - coords_mm[:, 0]) / s1_t[0], (coords_mm[:, 0] - c2_t[0]) / s2_t[0])
    uy_t = np.where(in_1t, (coords_mm[:, 1] - c1_t[1]) / s1_t[1], (coords_mm[:, 1] - c2_t[1]) / s2_t[1])
    ux_t = np.where(in_1t, (coords_mm[:, 2] - c1_t[2]) / s1_t[2], (coords_mm[:, 2] - c2_t[2]) / s2_t[2])
    bz_old = np.clip(((uz_t + 2.8) / 5.6 * 26).astype(int), 0, 25)
    bx_old = np.clip(((ux_t + 2.8) / 5.6 * 26).astype(int), 0, 25)
    bdt_16 = np.clip(((dts_m + 0.8) / 7.6 * 16).astype(int), 0, 15)
    bdf_18 = np.clip(((dfs_m + 0.8) / 27.0 * 18).astype(int), 0, 17)

    # 2. Medial-Lateral Canonical Plateau Frame + Wide Frame
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
    bdf_f = np.clip(((dfs_m + 0.8) / 7.6 * 16).astype(int), 0, 15)

    rel_fz_ml = rel_f[:, 0] if med_low_z else (1.0 - rel_f[:, 0])
    bz_fml = np.clip(((rel_fz_ml + 0.10) / 1.20 * 34).astype(int), 0, 33)
    is_lat_f = (rel_fz_ml >= 0.5).astype(int)
    uz_fsep = np.where(is_lat_f == 1, (rel_fz_ml - 0.5) * 2.0, (0.5 - rel_fz_ml) * 2.0)
    bz_fsep = np.clip((uz_fsep / 1.25 * 22).astype(int), 0, 21)

    # 5. Patella Symmetric, ML-Oriented, and Facet-Separated Frames
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


def compute_case_atlas_hist_oai_v6(img, bm, spacing, lbl):
    df = fast_signed_distance_transform(bm['femur'] > 0, spacing)
    dt = fast_signed_distance_transform(bm['tibia'] > 0, spacing)
    dp = (
        fast_signed_distance_transform(bm['patella'] > 0, spacing)
        if ('patella' in bm and np.any(bm['patella'] > 0))
        else np.full_like(df, 50.0)
    )
    roi = (
        (df >= -1.2) & (dt >= -1.2) & (dp >= -1.2)
        & (
            ((df <= 6.8) & ((dt <= 58.0) | (dp <= 26.0)))
            | ((dt <= 6.6) & (df <= 30.0))
            | ((dp <= 7.8) & (df <= 24.0))
        )
    )
    inds = np.argwhere(roi)
    margin = [int(np.ceil(6.0 / s)) for s in spacing]
    z0, y0, x0 = np.maximum(inds.min(axis=0) - margin, 0)
    z1, y1, x1 = np.minimum(inds.max(axis=0) + margin + 1, img.shape)

    sig_s = [2.0 / spacing[0], 0.7 / spacing[1], 2.0 / spacing[2]]
    df_smooth = gaussian_filter(df[z0:z1, y0:y1, x0:x1], sigma=sig_s)
    dt_smooth = gaussian_filter(dt[z0:z1, y0:y1, x0:x1], sigma=sig_s)
    dp_smooth = gaussian_filter(dp[z0:z1, y0:y1, x0:x1], sigma=sig_s)

    f_stats, _, _ = _compute_articular_landmarks_and_stats(bm['femur'], df, np.minimum(dt, dp), 45.0, spacing)
    _, _, t_comp = _compute_articular_landmarks_and_stats(bm['tibia'], dt, df, 25.0, spacing)
    p_stats, _, _ = _compute_articular_landmarks_and_stats(bm.get('patella', bm['tibia']), dp, df, 25.0, spacing)
    ml_info = compute_ml_frames_3bone(bm, df, dt, spacing)

    zc_r, yc_r, xc_r = inds[:, 0] - z0, inds[:, 1] - y0, inds[:, 2] - x0
    dfs_r_all = df_smooth[zc_r, yc_r, xc_r]
    dts_r_all = dt_smooth[zc_r, yc_r, xc_r]
    dps_r_all = dp_smooth[zc_r, yc_r, xc_r]
    pos_sdt = (dfs_r_all >= -0.8) & (dts_r_all >= -0.8) & (dps_r_all >= -0.8)
    inds_atl = inds[pos_sdt]
    coords_roi = inds_atl.astype(np.float32) * np.array(spacing, dtype=np.float32)
    dfs_r = dfs_r_all[pos_sdt]
    dts_r = dts_r_all[pos_sdt]
    dps_r = dps_r_all[pos_sdt]

    bins, _ = compute_atlas_bins_oai_v6(coords_roi, dfs_r, dts_r, dps_r, f_stats, t_comp, p_stats, ml_info)
    y_r = lbl[inds_atl[:, 0], inds_atl[:, 1], inds_atl[:, 2]]

    m_fc = (y_r == 2)
    m_tc = (y_r == 4)
    m_pc = (y_r == 6)
    base_t = (dts_r >= -0.3) & (dts_r <= 2.2)
    base_f = (dfs_r >= -0.3) & (dfs_r <= 2.2)
    base_p = (dps_r >= -0.3) & (dps_r <= 2.4)

    h_old_tot = fast_hist_add(SH_OLD, bins['t_old'])
    h_old_tc = fast_hist_add(SH_OLD, bins['t_old'], m_tc)
    h_old_fc = fast_hist_add(SH_OLD, bins['t_old'], m_fc)

    h_ml_tot = fast_hist_add(SH_ML, bins['t_ml'])
    h_ml_tc = fast_hist_add(SH_ML, bins['t_ml'], m_tc)
    h_ml_fc = fast_hist_add(SH_ML, bins['t_ml'], m_fc)

    h_mly_tot = fast_hist_add(SH_MLY, bins['t_mly'])
    h_mly_tc = fast_hist_add(SH_MLY, bins['t_mly'], m_tc)

    h_sep_tot = fast_hist_add(SH_SEP, bins['t_sep'])
    h_sep_tc = fast_hist_add(SH_SEP, bins['t_sep'], m_tc)
    h_sep_fc = fast_hist_add(SH_SEP, bins['t_sep'], m_fc)

    h_wide_tot = fast_hist_add(SH_WIDE, bins['t_wide'])
    h_wide_tc = fast_hist_add(SH_WIDE, bins['t_wide'], m_tc)

    b_ml_2d = (bins['t_ml'][0], bins['t_ml'][1])
    h2d_ml_base = fast_hist_add((34, 26), b_ml_2d, base_t)
    h2d_ml_tc = fast_hist_add((34, 26), b_ml_2d, base_t & m_tc)
    h2d_ml_dts = fast_hist_add((34, 26), b_ml_2d, m_tc, weights=np.maximum(dts_r, 0.0))

    b_sep_2d = (bins['t_sep'][0], bins['t_sep'][1], bins['t_sep'][2])
    h2d_sep_base = fast_hist_add((2, 24, 24), b_sep_2d, base_t)
    h2d_sep_tc = fast_hist_add((2, 24, 24), b_sep_2d, base_t & m_tc)
    h2d_sep_dts = fast_hist_add((2, 24, 24), b_sep_2d, m_tc, weights=np.maximum(dts_r, 0.0))

    b_wide_2d = (bins['t_wide'][0], bins['t_wide'][1])
    h2d_wide_base = fast_hist_add((32, 28), b_wide_2d, base_t)
    h2d_wide_tc = fast_hist_add((32, 28), b_wide_2d, base_t & m_tc)
    h2d_wide_dts = fast_hist_add((32, 28), b_wide_2d, m_tc, weights=np.maximum(dts_r, 0.0))

    h_fo_tot = fast_hist_add(SH_FO, bins['f_old'])
    h_fo_fc = fast_hist_add(SH_FO, bins['f_old'], m_fc)
    h_fo_tc = fast_hist_add(SH_FO, bins['f_old'], m_tc)
    h_fo_pc = fast_hist_add(SH_FO, bins['f_old'], m_pc)

    h_fml_tot = fast_hist_add(SH_FML, bins['f_ml'])
    h_fml_fc = fast_hist_add(SH_FML, bins['f_ml'], m_fc)
    h_fml_tc = fast_hist_add(SH_FML, bins['f_ml'], m_tc)
    h_fml_pc = fast_hist_add(SH_FML, bins['f_ml'], m_pc)

    h_fsep_tot = fast_hist_add(SH_FSEP, bins['f_sep'])
    h_fsep_fc = fast_hist_add(SH_FSEP, bins['f_sep'], m_fc)
    h_fsep_tc = fast_hist_add(SH_FSEP, bins['f_sep'], m_tc)
    h_fsep_pc = fast_hist_add(SH_FSEP, bins['f_sep'], m_pc)

    b_fml_3d = (bins['f_ml'][0], bins['f_ml'][1], bins['f_ml'][2])
    h3d_fml_base = fast_hist_add((34, 26, 26), b_fml_3d, base_f)
    h3d_fml_fc = fast_hist_add((34, 26, 26), b_fml_3d, base_f & m_fc)
    h3d_fml_dfs = fast_hist_add((34, 26, 26), b_fml_3d, m_fc, weights=np.maximum(dfs_r, 0.0))

    h_po_tot = fast_hist_add(SH_PO, bins['p_old'])
    h_po_pc = fast_hist_add(SH_PO, bins['p_old'], m_pc)
    h_po_fc = fast_hist_add(SH_PO, bins['p_old'], m_fc)

    h_pml_tot = fast_hist_add(SH_PML, bins['p_ml'])
    h_pml_pc = fast_hist_add(SH_PML, bins['p_ml'], m_pc)
    h_pml_fc = fast_hist_add(SH_PML, bins['p_ml'], m_fc)

    h_psep_tot = fast_hist_add(SH_PSEP, bins['p_sep'])
    h_psep_pc = fast_hist_add(SH_PSEP, bins['p_sep'], m_pc)
    h_psep_fc = fast_hist_add(SH_PSEP, bins['p_sep'], m_fc)

    b_pml_2d = (bins['p_ml'][0], bins['p_ml'][1])
    h2d_pml_base = fast_hist_add((28, 24), b_pml_2d, base_p)
    h2d_pml_pc = fast_hist_add((28, 24), b_pml_2d, base_p & m_pc)
    h2d_pml_dps = fast_hist_add((28, 24), b_pml_2d, m_pc, weights=np.maximum(dps_r, 0.0))

    return (
        h_old_tot, h_old_tc, h_old_fc,
        h_ml_tot, h_ml_tc, h_ml_fc,
        h_mly_tot, h_mly_tc,
        h_sep_tot, h_sep_tc, h_sep_fc,
        h_wide_tot, h_wide_tc,
        h2d_ml_base, h2d_ml_tc, h2d_ml_dts,
        h2d_sep_base, h2d_sep_tc, h2d_sep_dts,
        h2d_wide_base, h2d_wide_tc, h2d_wide_dts,
        h_fo_tot, h_fo_fc, h_fo_tc, h_fo_pc,
        h_fml_tot, h_fml_fc, h_fml_tc, h_fml_pc,
        h_fsep_tot, h_fsep_fc, h_fsep_tc, h_fsep_pc,
        h3d_fml_base, h3d_fml_fc, h3d_fml_dfs,
        h_po_tot, h_po_pc, h_po_fc,
        h_pml_tot, h_pml_pc, h_pml_fc,
        h_psep_tot, h_psep_pc, h_psep_fc,
        h2d_pml_base, h2d_pml_pc, h2d_pml_dps,
    )


def make_atlas_priors_oai_v6(hists):
    (
        h_old_tot, h_old_tc, h_old_fc,
        h_ml_tot, h_ml_tc, h_ml_fc,
        h_mly_tot, h_mly_tc,
        h_sep_tot, h_sep_tc, h_sep_fc,
        h_wide_tot, h_wide_tc,
        h2d_ml_base, h2d_ml_tc, h2d_ml_dts,
        h2d_sep_base, h2d_sep_tc, h2d_sep_dts,
        h2d_wide_base, h2d_wide_tc, h2d_wide_dts,
        h_fo_tot, h_fo_fc, h_fo_tc, h_fo_pc,
        h_fml_tot, h_fml_fc, h_fml_tc, h_fml_pc,
        h_fsep_tot, h_fsep_fc, h_fsep_tc, h_fsep_pc,
        h3d_fml_base, h3d_fml_fc, h3d_fml_dfs,
        h_po_tot, h_po_pc, h_po_fc,
        h_pml_tot, h_pml_pc, h_pml_fc,
        h_psep_tot, h_psep_pc, h_psep_fc,
        h2d_pml_base, h2d_pml_pc, h2d_pml_dps,
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
    pf_old_pc = (gaussian_filter(h_fo_pc, sigma=sig_f) / sf_old).astype(np.float32)

    sf_fml = gaussian_filter(h_fml_tot, sigma=sig_f) + 2.0
    pf_fml_fc = (gaussian_filter(h_fml_fc, sigma=sig_f) / sf_fml).astype(np.float32)
    pf_fml_tc = (gaussian_filter(h_fml_tc, sigma=sig_f) / sf_fml).astype(np.float32)
    pf_fml_pc = (gaussian_filter(h_fml_pc, sigma=sig_f) / sf_fml).astype(np.float32)

    pf_sep_fc = np.stack([
        gaussian_filter(h_fsep_fc[c], sigma=sig_f) / (gaussian_filter(h_fsep_tot[c], sigma=sig_f) + 2.0)
        for c in range(2)
    ]).astype(np.float32)
    pf_sep_tc = np.stack([
        gaussian_filter(h_fsep_tc[c], sigma=sig_f) / (gaussian_filter(h_fsep_tot[c], sigma=sig_f) + 2.0)
        for c in range(2)
    ]).astype(np.float32)
    pf_sep_pc = np.stack([
        gaussian_filter(h_fsep_pc[c], sigma=sig_f) / (gaussian_filter(h_fsep_tot[c], sigma=sig_f) + 2.0)
        for c in range(2)
    ]).astype(np.float32)

    p3d_fml_fc = (gaussian_filter(h3d_fml_fc, sigma=0.90) / (gaussian_filter(h3d_fml_base, sigma=0.90) + 2.0)).astype(np.float32)
    hf_fml_3d = np.sum(h_fml_fc, axis=3)
    edf_fml_3d = (gaussian_filter(h3d_fml_dfs, sigma=0.90) / (gaussian_filter(hf_fml_3d, sigma=0.90) + 2.0)).astype(np.float32)

    sp_old = gaussian_filter(h_po_tot, sigma=sig_t) + 2.0
    pp_old_pc = (gaussian_filter(h_po_pc, sigma=sig_t) / sp_old).astype(np.float32)
    pp_old_fc = (gaussian_filter(h_po_fc, sigma=sig_t) / sp_old).astype(np.float32)

    sp_ml = gaussian_filter(h_pml_tot, sigma=sig_t) + 2.0
    pp_ml_pc = (gaussian_filter(h_pml_pc, sigma=sig_t) / sp_ml).astype(np.float32)
    pp_ml_fc = (gaussian_filter(h_pml_fc, sigma=sig_t) / sp_ml).astype(np.float32)

    pp_sep_pc = np.stack([
        gaussian_filter(h_psep_pc[c], sigma=sig_t) / (gaussian_filter(h_psep_tot[c], sigma=sig_t) + 2.0)
        for c in range(2)
    ]).astype(np.float32)
    pp_sep_fc = np.stack([
        gaussian_filter(h_psep_fc[c], sigma=sig_t) / (gaussian_filter(h_psep_tot[c], sigma=sig_t) + 2.0)
        for c in range(2)
    ]).astype(np.float32)

    p2d_pml_pc = (gaussian_filter(h2d_pml_pc, sigma=0.90) / (gaussian_filter(h2d_pml_base, sigma=0.90) + 2.0)).astype(np.float32)
    hp_ml_2d = np.sum(h_pml_pc, axis=(2, 3))
    edp_pml_2d = (gaussian_filter(h2d_pml_dps, sigma=0.90) / (gaussian_filter(hp_ml_2d, sigma=0.90) + 2.0)).astype(np.float32)

    return {
        'pt_old_tc': pt_old_tc, 'pt_old_fc': pt_old_fc,
        'pt_ml_tc': pt_ml_tc, 'pt_ml_fc': pt_ml_fc,
        'pt_mly_tc': pt_mly_tc,
        'pt_sep_tc': pt_sep_tc, 'pt_sep_fc': pt_sep_fc,
        'pt_wide_tc': pt_wide_tc,
        'p2d_ml_tc': p2d_ml_tc, 'edt_ml_2d': edt_ml_2d,
        'p2d_sep_tc': p2d_sep_tc, 'edt_sep_2d': edt_sep_2d,
        'p2d_wide_tc': p2d_wide_tc, 'edt_wide_2d': edt_wide_2d,
        'pf_old_fc': pf_old_fc, 'pf_old_tc': pf_old_tc, 'pf_old_pc': pf_old_pc,
        'pf_fml_fc': pf_fml_fc, 'pf_fml_tc': pf_fml_tc, 'pf_fml_pc': pf_fml_pc,
        'pf_sep_fc': pf_sep_fc, 'pf_sep_tc': pf_sep_tc, 'pf_sep_pc': pf_sep_pc,
        'p3d_fml_fc': p3d_fml_fc, 'edf_fml_3d': edf_fml_3d,
        'pp_old_pc': pp_old_pc, 'pp_old_fc': pp_old_fc,
        'pp_ml_pc': pp_ml_pc, 'pp_ml_fc': pp_ml_fc,
        'pp_sep_pc': pp_sep_pc, 'pp_sep_fc': pp_sep_fc,
        'p2d_pml_pc': p2d_pml_pc, 'edp_pml_2d': edp_pml_2d,
    }


_D_GRID = np.array([-3.6, -2.8, -2.0, -1.4, -0.9, -0.45, 0.0, 0.45, 0.9, 1.4, 2.0, 2.7, 3.5, 4.5, 5.6], dtype=np.float32)
_H_GRID = np.array([-1.0, -0.4, 0.0, 0.5, 1.0, 1.6, 2.2, 3.0, 4.0, 5.2], dtype=np.float32)
_DY_STEPS = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=int)
_DX_STEPS = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=int)
_R_STEPS = np.array([-6.0, -3.5, -1.8, 1.8, 3.5, 6.0], dtype=np.float32)


def prepare_geometry_oai_v6(img, sp, bm_ref, bm_alt, bm_raw, bone_name, atlas_p, dts_pre=None, landmarks_pre=None):
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


def extract_features_oai_v6(geo, target_mask, pm_self=None, pm_other=None, rng_jit=None):
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

    if rng_jit is not None:
        jit = rng_jit.uniform(-0.30, 0.30, size=N).astype(np.float32)
        w_jit = np.clip(1.0 - np.abs(d_s_m) / 2.2, 0.0, 1.0)
        d_raw_feat = d_raw_m + w_jit * jit
        d_alt_feat = d_alt_m + w_jit * jit
        d_s_feat = d_s_m + w_jit * jit
        d_s_anchor = d_s_m + 0.5 * w_jit * jit
    else:
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

    inorm_0 = geo['inorm']
    base_span = max(i_high - i_low, 1.0)
    if pm_self is not None:
        pm_crop = pm_self[z0:z1, y0:y1, x0:x1]
        m_p1_core = pm_crop[..., 2] >= 0.55
        if np.sum(m_p1_core) >= 40:
            i_high = float(np.median(img_f32[m_p1_core]))

    scale = float(base_span / max(i_high - i_low, 1.0))
    inorm_s = inorm_0 if abs(scale - 1.0) < 1e-7 else (inorm_0 * scale)
    abs_z = gaussian_filter(np.abs(z_cart), sigma=[1.0 / sp[0], 0.35 / sp[1], 1.0 / sp[2]])
    img_f0 = gaussian_filter(inorm_s, sigma=[0.35 / s for s in sp])
    img_lat = gaussian_filter(inorm_s, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
    img_s1 = gaussian_filter(inorm_s, sigma=[1.0 / s for s in sp])
    img_s2 = gaussian_filter(inorm_s, sigma=[2.0 / s for s in sp])
    img_s4 = gaussian_filter(inorm_s, sigma=[4.0 / s for s in sp])
    img_s8 = gaussian_filter(inorm_s, sigma=[8.0 / s for s in sp])
    gz0, gy0, gx0 = np.gradient(img_f0, *sp)
    gmag0 = np.sqrt(gz0**2 + gy0**2 + gx0**2)
    gz1, gy1, gx1 = np.gradient(img_s1, *sp)
    gmag1 = np.sqrt(gz1**2 + gy1**2 + gx1**2)

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

        # Also X-axis sheet integral for vertical patellofemoral cartilage
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
    """Combine Tibia and Patella 4-class probability maps into opposing map for Femur."""
    use_p = dp < dt
    out = np.where(use_p[..., None], pm_p, pm_t).astype(np.float32)
    out[..., 1] = np.maximum(pm_t[..., 1], pm_p[..., 1])
    out[..., 2] = np.maximum(pm_t[..., 2], pm_p[..., 2])
    out[..., 0] = np.clip(1.0 - out[..., 1] - out[..., 2] - out[..., 3], 0.0, 1.0)
    return out


def _p1_folds_screened_oai(geo, clfs, return_folds=True):
    sh = geo['shell']
    if not np.any(sh):
        z4 = np.zeros((*sh.shape, 4), dtype=np.float32)
        z4[..., 0] = 1.0
        return z4, ([z4.copy() for _ in clfs] if return_folds else [])
    X = extract_features_oai_v6(geo, sh, rng_jit=None)
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
    pm_f_list = []
    if return_folds:
        for pf in p_folds:
            v = np.zeros((*sh.shape, 4), dtype=np.float32)
            v[..., 0] = 1.0
            v[sh] = pf
            pm_f_list.append(v)
    return pm_avg, pm_f_list


def _p2_screened_oai(geo, pm1_s, pm1_o, clfs_hgb, clf_et):
    sh = geo['shell']
    act = sh & ((pm1_s[..., 2] >= 0.010) | (pm1_s[..., 3] >= 0.010) | (pm1_o[..., 2] >= 0.015))
    if not np.any(act):
        return pm1_s.copy(), pm1_s.copy(), pm1_s.copy()
    X2 = extract_features_oai_v6(geo, act, pm_self=pm1_s, pm_other=pm1_o, rng_jit=None)
    p0 = clfs_hgb[0].predict_proba(X2).astype(np.float32)
    bnd = (p0[:, 2] >= 0.012) & (p0[:, 2] <= 0.988)
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
    """Extract Stage-3 boundary & subchondral refinement features on ROI crop for one compartment."""
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
            v_s3_f[cand_f] = (1.0 - w_s3_f) * sf_base[cand_f] + w_s3_f * (p_s3_f + 0.06)
            sf_fin = gaussian_filter(v_s3_f, sigma=sig_f)
        else:
            sf_fin = sf_base
        if 'tibia' in s3_scores and s3_scores['tibia'] is not None:
            cand_t, p_s3_t = s3_scores['tibia']
            v_s3_t = st_base.copy()
            v_s3_t[cand_t] = (1.0 - w_s3_t) * st_base[cand_t] + w_s3_t * (p_s3_t + 0.06)
            st_fin = gaussian_filter(v_s3_t, sigma=sig_t)
        else:
            st_fin = st_base
        if 'patella' in s3_scores and s3_scores['patella'] is not None:
            cand_p, p_s3_p = s3_scores['patella']
            v_s3_p = sp_base.copy()
            v_s3_p[cand_p] = (1.0 - w_s3_p) * sp_base[cand_p] + w_s3_p * (p_s3_p + 0.06)
            sp_fin = gaussian_filter(v_s3_p, sigma=sig_p)
        else:
            sp_fin = sp_base
    else:
        sf_fin, st_fin, sp_fin = sf_base, st_base, sp_base

    # Adaptive per-scan volume ratio threshold calibration
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

    # Partition competition:
    # Femoral cartilage beats Tibial cartilage where sf_part >= st_part, and beats Patellar cartilage where sf_part >= sp_part
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

    pv = np.zeros((*img_shape, 4), dtype=np.float32)
    pv[sl][..., 1] = np.clip(sf_fin, 0.0, 1.0)
    pv[sl][..., 2] = np.clip(st_fin, 0.0, 1.0)
    pv[sl][..., 3] = np.clip(sp_fin, 0.0, 1.0)
    pv[y_pred == 2, 1] = np.maximum(pv[y_pred == 2, 1], 0.75)
    pv[y_pred == 4, 2] = np.maximum(pv[y_pred == 4, 2], 0.75)
    pv[y_pred == 6, 3] = np.maximum(pv[y_pred == 6, 3], 0.75)
    s_fg = pv[..., 1] + pv[..., 2] + pv[..., 3]
    over = s_fg > 1.0
    if np.any(over):
        pv[over, 1] /= s_fg[over]
        pv[over, 2] /= s_fg[over]
        pv[over, 3] /= s_fg[over]
    pv[..., 0] = np.clip(1.0 - pv[..., 1] - pv[..., 2] - pv[..., 3], 0.0, 1.0)
    return y_pred, pv


_MP_STATE = {}


def _fit_p1_fold(fk):
    X_all = _MP_STATE['X']
    y_all = _MP_STATE['y']
    b = _MP_STATE['b']
    f_sz = _MP_STATE['f_sz']
    leaf_grid = [55, 63, 59, 67]
    s_i, e_i = b[fk * f_sz], b[(fk + 1) * f_sz]
    X_f = np.vstack([X_all[:s_i], X_all[e_i:]])
    y_f = np.concatenate([y_all[:s_i], y_all[e_i:]])
    with threadpool_limits(limits=14, user_api='openmp'):
        return HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=leaf_grid[fk], min_samples_leaf=25,
            learning_rate=0.06, max_features=0.45, l2_regularization=1.2,
            early_stopping=False, random_state=100 + fk * 10,
        ).fit(X_f, y_f)


def _fit_p2_model(idx):
    X_all = _MP_STATE['X']
    y_all = _MP_STATE['y']
    b = _MP_STATE['b']
    f_sz = _MP_STATE['f_sz']
    with threadpool_limits(limits=10, user_api='openmp'):
        if idx == 0:
            return HistGradientBoostingClassifier(
                max_iter=155, max_leaf_nodes=55, min_samples_leaf=25,
                learning_rate=0.06, max_features=0.45, l2_regularization=1.0,
                early_stopping=False, random_state=201,
            ).fit(X_all, y_all)
        elif idx == 1:
            return HistGradientBoostingClassifier(
                max_iter=155, max_leaf_nodes=67, min_samples_leaf=30,
                learning_rate=0.06, max_features=0.50, l2_regularization=1.5,
                early_stopping=False, random_state=202,
            ).fit(X_all, y_all)
        else:
            fk = idx - 2
            s_i, e_i = b[fk * f_sz], b[(fk + 1) * f_sz]
            X_f = np.vstack([X_all[:s_i], X_all[e_i:]])
            y_f = np.concatenate([y_all[:s_i], y_all[e_i:]])
            return HistGradientBoostingClassifier(
                max_iter=145, max_leaf_nodes=59, min_samples_leaf=25,
                learning_rate=0.06, max_features=0.45, l2_regularization=1.2,
                early_stopping=False, random_state=301 + fk * 10,
            ).fit(X_f, y_f)


def _fit_s3_model(idx):
    X_all = _MP_STATE['X']
    y_all = _MP_STATE['y']
    b = _MP_STATE['b']
    f_sz = _MP_STATE['f_sz']
    with threadpool_limits(limits=10, user_api='openmp'):
        if idx == 0:
            return HistGradientBoostingClassifier(
                max_iter=155, max_leaf_nodes=51, min_samples_leaf=30,
                learning_rate=0.055, max_features=0.50, l2_regularization=1.5,
                early_stopping=False, random_state=501,
            ).fit(X_all, y_all)
        elif idx == 1:
            return HistGradientBoostingClassifier(
                max_iter=155, max_leaf_nodes=63, min_samples_leaf=35,
                learning_rate=0.055, max_features=0.55, l2_regularization=2.0,
                early_stopping=False, random_state=502,
            ).fit(X_all, y_all)
        else:
            fk = idx - 2
            s_i, e_i = b[fk * f_sz], b[(fk + 1) * f_sz]
            X_f = np.vstack([X_all[:s_i], X_all[e_i:]])
            y_f = np.concatenate([y_all[:s_i], y_all[e_i:]])
            return HistGradientBoostingClassifier(
                max_iter=145, max_leaf_nodes=51, min_samples_leaf=30,
                learning_rate=0.055, max_features=0.50, l2_regularization=1.5,
                early_stopping=False, random_state=600 + fk,
            ).fit(X_f, y_f)


def _p1_tr_worker(i):
    fk = i // _MP_STATE['f_sz']
    with threadpool_limits(limits=1, user_api='openmp'):
        pm_oof, _ = _p1_folds_screened_oai(_MP_STATE['geos'][i], [_MP_STATE['fold_clfs'][fk]], return_folds=False)
    return pm_oof


def _p1_ev_worker(i):
    with threadpool_limits(limits=1, user_api='openmp'):
        pm_ev, _ = _p1_folds_screened_oai(_MP_STATE['geos'][i], _MP_STATE['fold_clfs'], return_folds=False)
    return pm_ev


def _p2_tr_worker(i):
    c = _MP_STATE['cases'][i]
    c_name = _MP_STATE['c_name']
    fk = i // _MP_STATE['f_sz']
    if c_name == 'tibia' or c_name == 'patella':
        pm_o = c['pm1_femur']
    else:
        pm_o = combine_tp_for_femur(c['pm1_tibia'], c['pm1_patella'], c['dts'][1], c['dts'][2])
    with threadpool_limits(limits=1, user_api='openmp'):
        pm2_oof, _, _ = _p2_screened_oai(
            _MP_STATE['geos'][i], c[f'pm1_{c_name}'], pm_o, _MP_STATE['oof_folds'][fk], None
        )
    return pm2_oof


def _p2_ev_worker(i):
    c = _MP_STATE['cases'][i]
    c_name = _MP_STATE['c_name']
    if c_name == 'tibia' or c_name == 'patella':
        pm_o = c['pm1_femur']
    else:
        pm_o = combine_tp_for_femur(c['pm1_tibia'], c['pm1_patella'], c['dts'][1], c['dts'][2])
    with threadpool_limits(limits=1, user_api='openmp'):
        pm2_ev, _, _ = _p2_screened_oai(
            _MP_STATE['geos'][i], c[f'pm1_{c_name}'], pm_o, _MP_STATE['hgb'], _MP_STATE['et']
        )
    return pm2_ev


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

    tr_raw = joblib.load(os.path.join(cache_dir, 'train128_pred_bones.joblib'))
    tr_v1 = joblib.load(os.path.join(cache_dir, 'train128_ref_bones_v1.joblib'))
    tr_v2 = joblib.load(os.path.join(cache_dir, 'train128_ref_bones_v2.joblib'))
    ev_raw = joblib.load(os.path.join(cache_dir, 'eval31_pred_bones.joblib'))
    ev_v1 = joblib.load(os.path.join(cache_dir, 'eval31_ref_bones_v1.joblib'))
    ev_v2 = joblib.load(os.path.join(cache_dir, 'eval31_ref_bones_v2.joblib'))

    def load_case_all(fname, d_raw, d_v1, d_v2):
        img, _ = load_volume(os.path.join(data_dir, fname), return_spacing=True)
        lbl = load_volume(os.path.join(lbl_dir, fname.replace('image-', 'labels-')), return_spacing=False).astype(np.uint8)
        layout = detect_oai_layout(img)
        img_c = to_canonical_oai(img, layout=layout).astype(np.float32)
        lbl_c = to_canonical_oai(lbl, layout=layout).astype(np.uint8)
        sp_c = (1.0, 1.0, 1.0)
        bm_v2 = d_v2[fname]
        bm_v1 = d_v1[fname]
        bm_r = d_raw[fname]
        df2 = fast_signed_distance_transform(bm_v2['femur'] > 0, sp_c)
        dt2 = fast_signed_distance_transform(bm_v2['tibia'] > 0, sp_c)
        dp2 = fast_signed_distance_transform(bm_v2['patella'] > 0, sp_c)
        df1 = fast_signed_distance_transform(bm_v1['femur'] > 0, sp_c)
        dt1 = fast_signed_distance_transform(bm_v1['tibia'] > 0, sp_c)
        dp1 = fast_signed_distance_transform(bm_v1['patella'] > 0, sp_c)
        df0 = fast_signed_distance_transform(bm_r['femur'] > 0, sp_c)
        dt0 = fast_signed_distance_transform(bm_r['tibia'] > 0, sp_c)
        dp0 = fast_signed_distance_transform(bm_r['patella'] > 0, sp_c)
        dts = (df2, dt2, dp2, df1, dt1, dp1, df0, dt0, dp0)
        f_st, _, f_cp = _compute_articular_landmarks_and_stats(bm_v2['femur'], df2, np.minimum(dt2, dp2), 45.0, sp_c)
        t_st, _, t_cp = _compute_articular_landmarks_and_stats(bm_v2['tibia'], dt2, df2, 25.0, sp_c)
        p_st, _, p_cp = _compute_articular_landmarks_and_stats(bm_v2['patella'], dp2, df2, 25.0, sp_c)
        ml_i = compute_ml_frames_3bone(bm_v2, df2, dt2, sp_c)
        lm = (f_st, f_cp, t_st, t_cp, p_st, p_cp, ml_i)
        return {
            'fname': fname,
            'img': img_c,
            'lbl': lbl_c,
            'sp': sp_c,
            'bm_v2': bm_v2,
            'bm_v1': bm_v1,
            'bm_raw': bm_r,
            'dts': dts,
            'lm': lm,
        }

    tr_cases = Parallel(n_jobs=16, prefer='threads')(
        delayed(load_case_all)(fn, tr_raw, tr_v1, tr_v2) for fn in split['train']
    )
    ev_cases = Parallel(n_jobs=16, prefer='threads')(
        delayed(load_case_all)(fn, ev_raw, ev_v1, ev_v2) for fn in split['eval']
    )
    n_tr = len(tr_cases)
    n_ev = len(ev_cases)
    f_sz = n_tr // 4  # 32 scans per fold
    print(f"Loaded {n_tr} train + {n_ev} eval cases with 9 SDTs & landmarks in {time.time() - t_start:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # STEP 1: Build 4D Articular Coordinate Probabilistic Atlases (Full + LOO)
    # ------------------------------------------------------------------
    t_atl = time.time()
    per_case_h = Parallel(n_jobs=16, prefer='threads')(
        delayed(compute_case_atlas_hist_oai_v6)(c['img'], c['bm_v2'], c['sp'], c['lbl'])
        for c in tr_cases
    )
    n_h = len(per_case_h[0])
    tot_h = [sum(h[k] for h in per_case_h) for k in range(n_h)]
    ap_full = make_atlas_priors_oai_v6(tot_h)
    ap_loo = Parallel(n_jobs=16, prefer='threads')(
        delayed(make_atlas_priors_oai_v6)([tot_h[k] - per_case_h[i][k] for k in range(n_h)])
        for i in range(n_tr)
    )
    del per_case_h, tot_h
    gc.collect()
    print(f"Built Full + 128-LOO 4D Articular Atlases in {time.time() - t_atl:.1f}s", flush=True)

    # Initialize union shell bounding box tracker
    for c in tr_cases + ev_cases:
        c['u_sh'] = np.zeros(c['img'].shape, dtype=bool)

    # ------------------------------------------------------------------
    # STEP 2: Train Pass-1 4-Fold Specialist Ensembles (femur, tibia, patella)
    # ------------------------------------------------------------------
    p1_ckpt_path = os.path.join(model_dir, 'cartilage_rf_p1.joblib')
    p1_folds_saved = None
    if os.path.exists(p1_ckpt_path):
        try:
            d_p1 = joblib.load(p1_ckpt_path)
            if d_p1.get('format_version') == 'oai_cartilage_v6_p1' and 'p1_folds' in d_p1:
                p1_folds_saved = d_p1['p1_folds']
                print("Loaded existing cartilage_rf_p1.joblib fold models!", flush=True)
        except Exception:
            p1_folds_saved = None

    p1_folds = {}
    spec_specs = [
        ('femur', 1, 2, (4, 6)),
        ('tibia', 3, 4, (2,)),
        ('patella', 5, 6, (2,)),
    ]

    for c_name, b_lbl, c_lbl, opp_c_lbls in spec_specs:
        t_p1 = time.time()
        geos_tr = Parallel(n_jobs=16, prefer='threads')(
            delayed(prepare_geometry_oai_v6)(
                tr_cases[i]['img'], tr_cases[i]['sp'],
                tr_cases[i]['bm_v2'], tr_cases[i]['bm_v1'], tr_cases[i]['bm_raw'],
                c_name, ap_loo[i], dts_pre=tr_cases[i]['dts'], landmarks_pre=tr_cases[i]['lm']
            )
            for i in range(n_tr)
        )
        for i in range(n_tr):
            tr_cases[i]['u_sh'] |= geos_tr[i]['shell']

        if p1_folds_saved is not None and c_name in p1_folds_saved:
            fold_clfs = p1_folds_saved[c_name]
            p1_folds[c_name] = fold_clfs
            print(f"  [{c_name.upper()} P1] Reusing saved 4 fold HGBs", flush=True)
        else:
            def sample_p1_one(i):
                c = tr_cases[i]
                geo = geos_tr[i]
                lbl = c['lbl']
                sh = geo['shell']
                z0, z1, y0, y1, x0, x1 = geo['bbox']
                ds = geo['d_self_crop']
                rng = np.random.RandomState(600 + i)

                y_loc = np.zeros_like(lbl, dtype=np.uint8)
                y_loc[lbl == b_lbl] = 1
                y_loc[lbl == c_lbl] = 2
                for ol in opp_c_lbls:
                    y_loc[lbl == ol] = 3

                inds_sh = np.argwhere(sh)
                y_sh = y_loc[sh]
                ds_sh = ds[inds_sh[:, 0] - z0, inds_sh[:, 1] - y0, inds_sh[:, 2] - x0]

                pos_c = np.where(y_sh == 2)[0]
                pos_oc = np.where(y_sh == 3)[0]
                pos_bn_near = np.where((y_sh == 1) & (ds_sh >= -1.0))[0]
                pos_bn_far = np.where((y_sh == 1) & (ds_sh < -1.0))[0]
                neg_bg_near = np.where((y_sh == 0) & (ds_sh <= 2.8))[0]
                neg_bg_far = np.where((y_sh == 0) & (ds_sh > 2.8))[0]

                n_c_cap = 950 if c_name == 'femur' else (750 if c_name == 'tibia' else 650)
                chosen = []
                for pool, n_take in [
                    (pos_c, n_c_cap),
                    (pos_oc, 250),
                    (pos_bn_near, 380),
                    (pos_bn_far, 140),
                    (neg_bg_near, 460),
                    (neg_bg_far, 220),
                ]:
                    if len(pool) > 0:
                        chosen.append(rng.choice(pool, min(len(pool), n_take), replace=False))
                sel = inds_sh[np.concatenate(chosen)]
                smask = np.zeros_like(sh, dtype=bool)
                smask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                X1 = extract_features_oai_v6(geo, smask, rng_jit=rng)
                return X1, y_loc[smask]

            s_p1 = Parallel(n_jobs=16, prefer='threads')(delayed(sample_p1_one)(i) for i in range(n_tr))
            X1_all = np.vstack([s[0] for s in s_p1])
            y1_all = np.concatenate([s[1] for s in s_p1])
            b1 = [0]
            for s in s_p1:
                b1.append(b1[-1] + len(s[1]))
            del s_p1
            gc.collect()
            print(f"  [{c_name.upper()} P1] Sampled {X1_all.shape} in {time.time() - t_p1:.1f}s", flush=True)

            _MP_STATE['X'] = X1_all
            _MP_STATE['y'] = y1_all
            _MP_STATE['b'] = b1
            _MP_STATE['f_sz'] = f_sz
            with mp.get_context('fork').Pool(4) as pool:
                fold_clfs = pool.map(_fit_p1_fold, range(4))
            _MP_STATE.clear()
            del X1_all, y1_all
            gc.collect()
            p1_folds[c_name] = fold_clfs
            print(f"  [{c_name.upper()} P1] Fit 4 fold HGBs in {time.time() - t_p1:.1f}s", flush=True)

        _MP_STATE['geos'] = geos_tr
        _MP_STATE['fold_clfs'] = fold_clfs
        _MP_STATE['f_sz'] = f_sz
        with mp.get_context('fork').Pool(16) as pool:
            pms_tr = pool.map(_p1_tr_worker, range(n_tr))
        _MP_STATE.clear()
        for i in range(n_tr):
            tr_cases[i][f'pm1_{c_name}'] = pms_tr[i]
        del pms_tr, geos_tr
        gc.collect()

        geos_ev = Parallel(n_jobs=16, prefer='threads')(
            delayed(prepare_geometry_oai_v6)(
                ev_cases[i]['img'], ev_cases[i]['sp'],
                ev_cases[i]['bm_v2'], ev_cases[i]['bm_v1'], ev_cases[i]['bm_raw'],
                c_name, ap_full, dts_pre=ev_cases[i]['dts'], landmarks_pre=ev_cases[i]['lm']
            )
            for i in range(n_ev)
        )
        for i in range(n_ev):
            ev_cases[i]['u_sh'] |= geos_ev[i]['shell']

        _MP_STATE['geos'] = geos_ev
        _MP_STATE['fold_clfs'] = fold_clfs
        with mp.get_context('fork').Pool(16) as pool:
            pms_ev = pool.map(_p1_ev_worker, range(n_ev))
        _MP_STATE.clear()
        for i in range(n_ev):
            ev_cases[i][f'pm1_{c_name}'] = pms_ev[i]
        del pms_ev, geos_ev
        gc.collect()

        dsc_p1_ev = [
            float(2.0 * np.sum((ev_cases[i]['lbl'] == c_lbl) & (ev_cases[i][f'pm1_{c_name}'][..., 2] >= 0.48))
                  / max(np.sum(ev_cases[i]['lbl'] == c_lbl) + np.sum(ev_cases[i][f'pm1_{c_name}'][..., 2] >= 0.48), 1))
            for i in range(n_ev)
        ]
        print(f"  [{c_name.upper()} P1 31-Eval Raw Th=0.48] DSC = {np.mean(dsc_p1_ev):.4f} ± {np.std(dsc_p1_ev):.4f} ({time.time() - t_p1:.1f}s)", flush=True)

    # Save cartilage_rf_p1.joblib if not already saved
    if p1_folds_saved is None:
        joblib.dump({
            'format_version': 'oai_cartilage_v6_p1',
            'p1_folds': p1_folds,
            'atlas_priors': ap_full,
        }, p1_ckpt_path, compress=3)
        print("Saved cartilage_rf_p1.joblib!", flush=True)

    # ------------------------------------------------------------------
    # STEP 3: Train Pass-2 Auto-Context Specialist Ensembles (femur, tibia, patella)
    # ------------------------------------------------------------------
    p2_models = {}
    for c_name, b_lbl, c_lbl, opp_c_lbls in spec_specs:
        t_p2 = time.time()
        geos_tr = Parallel(n_jobs=16, prefer='threads')(
            delayed(prepare_geometry_oai_v6)(
                tr_cases[i]['img'], tr_cases[i]['sp'],
                tr_cases[i]['bm_v2'], tr_cases[i]['bm_v1'], tr_cases[i]['bm_raw'],
                c_name, ap_loo[i], dts_pre=tr_cases[i]['dts'], landmarks_pre=tr_cases[i]['lm']
            )
            for i in range(n_tr)
        )

        def get_pm_other(case_dict):
            if c_name == 'tibia':
                return case_dict['pm1_femur']
            elif c_name == 'patella':
                return case_dict['pm1_femur']
            else:
                return combine_tp_for_femur(case_dict['pm1_tibia'], case_dict['pm1_patella'], case_dict['dts'][1], case_dict['dts'][2])

        def sample_p2_one(i):
            c = tr_cases[i]
            geo = geos_tr[i]
            lbl = c['lbl']
            sh = geo['shell']
            pm_s = c[f'pm1_{c_name}']
            pm_o = get_pm_other(c)
            rng = np.random.RandomState(700 + i)

            y_loc = np.zeros_like(lbl, dtype=np.uint8)
            y_loc[lbl == b_lbl] = 1
            y_loc[lbl == c_lbl] = 2
            for ol in opp_c_lbls:
                y_loc[lbl == ol] = 3

            act = sh & ((pm_s[..., 2] >= 0.010) | (pm_s[..., 3] >= 0.010) | (pm_o[..., 2] >= 0.015) | (y_loc >= 2))
            inds_act = np.argwhere(act)
            y_act = y_loc[act]
            p1_c = pm_s[act, 2]

            pos_c = np.where(y_act == 2)[0]
            pos_oc = np.where(y_act == 3)[0]
            hard_neg_bn = np.where((y_act == 1) & (p1_c >= 0.05))[0]
            easy_neg_bn = np.where((y_act == 1) & (p1_c < 0.05))[0]
            hard_neg_bg = np.where((y_act == 0) & (p1_c >= 0.05))[0]
            easy_neg_bg = np.where((y_act == 0) & (p1_c < 0.05))[0]

            n_c_cap = 950 if c_name == 'femur' else (750 if c_name == 'tibia' else 650)
            chosen = []
            for pool, n_take in [
                (pos_c, n_c_cap),
                (pos_oc, 250),
                (hard_neg_bn, 380),
                (easy_neg_bn, 140),
                (hard_neg_bg, 460),
                (easy_neg_bg, 200),
            ]:
                if len(pool) > 0:
                    chosen.append(rng.choice(pool, min(len(pool), n_take), replace=False))
            sel = inds_act[np.concatenate(chosen)]
            smask = np.zeros_like(sh, dtype=bool)
            smask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
            X2 = extract_features_oai_v6(geo, smask, pm_self=pm_s, pm_other=pm_o, rng_jit=rng)
            return X2, y_loc[smask]

        s_p2 = Parallel(n_jobs=16, prefer='threads')(delayed(sample_p2_one)(i) for i in range(n_tr))
        X2_all = np.vstack([s[0] for s in s_p2])
        y2_all = np.concatenate([s[1] for s in s_p2])
        b2 = [0]
        for s in s_p2:
            b2.append(b2[-1] + len(s[1]))
        del s_p2
        gc.collect()
        print(f"  [{c_name.upper()} P2] Sampled {X2_all.shape} in {time.time() - t_p2:.1f}s", flush=True)

        _MP_STATE['X'] = X2_all
        _MP_STATE['y'] = y2_all
        _MP_STATE['b'] = b2
        _MP_STATE['f_sz'] = f_sz
        with mp.get_context('fork').Pool(6) as pool:
            p2_hgb_all = pool.map(_fit_p2_model, range(6))
        _MP_STATE.clear()
        hgb_1, hgb_2 = p2_hgb_all[0], p2_hgb_all[1]
        oof_p2_folds = [[p2_hgb_all[2 + fk]] for fk in range(4)]

        rng_et = np.random.RandomState(204)
        sub_et = rng_et.choice(len(y2_all), min(len(y2_all), 220000), replace=False)
        et_p2 = ExtraTreesClassifier(
            n_estimators=80, max_depth=24, min_samples_leaf=6, n_jobs=16, random_state=204
        ).fit(X2_all[sub_et], y2_all[sub_et])
        et_p2.n_jobs = 1
        del X2_all, y2_all
        gc.collect()
        print(f"  [{c_name.upper()} P2] Fit HGB+ET+OOF models in {time.time() - t_p2:.1f}s", flush=True)

        _MP_STATE['geos'] = geos_tr
        _MP_STATE['cases'] = tr_cases
        _MP_STATE['c_name'] = c_name
        _MP_STATE['f_sz'] = f_sz
        _MP_STATE['oof_folds'] = oof_p2_folds
        with mp.get_context('fork').Pool(16) as pool:
            pms2_tr = pool.map(_p2_tr_worker, range(n_tr))
        _MP_STATE.clear()
        for i in range(n_tr):
            tr_cases[i][f'pm2_{c_name}'] = pms2_tr[i]
        del pms2_tr, geos_tr
        gc.collect()

        geos_ev = Parallel(n_jobs=16, prefer='threads')(
            delayed(prepare_geometry_oai_v6)(
                ev_cases[i]['img'], ev_cases[i]['sp'],
                ev_cases[i]['bm_v2'], ev_cases[i]['bm_v1'], ev_cases[i]['bm_raw'],
                c_name, ap_full, dts_pre=ev_cases[i]['dts'], landmarks_pre=ev_cases[i]['lm']
            )
            for i in range(n_ev)
        )

        _MP_STATE['geos'] = geos_ev
        _MP_STATE['cases'] = ev_cases
        _MP_STATE['c_name'] = c_name
        _MP_STATE['hgb'] = [hgb_1, hgb_2]
        _MP_STATE['et'] = et_p2
        with mp.get_context('fork').Pool(16) as pool:
            pms2_ev = pool.map(_p2_ev_worker, range(n_ev))
        _MP_STATE.clear()
        for i in range(n_ev):
            ev_cases[i][f'pm2_{c_name}'] = pms2_ev[i]
        del pms2_ev, geos_ev
        gc.collect()

        p2_models[c_name] = {
            'hgb': [hgb_1, hgb_2],
            'et': et_p2,
            'oof_folds': oof_p2_folds,
        }
        dsc_p2_ev = [
            float(2.0 * np.sum((ev_cases[i]['lbl'] == c_lbl) & (ev_cases[i][f'pm2_{c_name}'][..., 2] >= 0.48))
                  / max(np.sum(ev_cases[i]['lbl'] == c_lbl) + np.sum(ev_cases[i][f'pm2_{c_name}'][..., 2] >= 0.48), 1))
            for i in range(n_ev)
        ]
        print(f"  [{c_name.upper()} P2 31-Eval Raw Th=0.48] DSC = {np.mean(dsc_p2_ev):.4f} ± {np.std(dsc_p2_ev):.4f} ({time.time() - t_p2:.1f}s)", flush=True)

    # Compute ROI slices from union shells
    for c in tr_cases + ev_cases:
        inds_u = np.argwhere(c.pop('u_sh'))
        uz0, uy0, ux0 = np.maximum(inds_u.min(axis=0) - 4, 0)
        uz1, uy1, ux1 = np.minimum(inds_u.max(axis=0) + 5, c['img'].shape)
        c['sl'] = (slice(uz0, uz1), slice(uy0, uy1), slice(ux0, ux1))
    gc.collect()

    # Precompute ROI priors for all 128 train and 31 eval scans
    for i in range(n_tr):
        c = tr_cases[i]
        c['priors_roi'] = compute_roi_priors_oai(
            c['sl'], c['sp'], c['dts'][0], c['dts'][1], c['dts'][2],
            c['lm'][0], c['lm'][3], c['lm'][4], c['lm'][6], ap_loo[i]
        )
    for i in range(n_ev):
        c = ev_cases[i]
        c['priors_roi'] = compute_roi_priors_oai(
            c['sl'], c['sp'], c['dts'][0], c['dts'][1], c['dts'][2],
            c['lm'][0], c['lm'][3], c['lm'][4], c['lm'][6], ap_full
        )

    # ------------------------------------------------------------------
    # STEP 4: Train Stage-3 Boundary & Subchondral Refiner per Compartment
    # ------------------------------------------------------------------
    s3_models = {}
    for c_name, b_lbl, c_lbl, opp_c_lbls in spec_specs:
        t_s3 = time.time()

        def extract_case_s3(c):
            sl = c['sl']
            df2, dt2, dp2, df1, dt1, dp1, df0, dt0, dp0 = [d[sl] for d in c['dts']]
            if c_name == 'femur':
                ds2, ds1, ds0, do2 = df2, df1, df0, np.minimum(dt2, dp2)
                pm_opp_ct = np.maximum(c['pm2_tibia'][sl][..., 2], c['pm2_patella'][sl][..., 2])
                a_new, a_old, edt_e = c['priors_roi']['a_new_fc'], c['priors_roi']['a_old_fc'], c['priors_roi']['edf_exp']
            elif c_name == 'tibia':
                ds2, ds1, ds0, do2 = dt2, dt1, dt0, df2
                pm_opp_ct = c['pm2_femur'][sl][..., 2]
                a_new, a_old, edt_e = c['priors_roi']['a_new_tc'], c['priors_roi']['a_old_tc'], c['priors_roi']['edt_exp']
            else:
                ds2, ds1, ds0, do2 = dp2, dp1, dp0, df2
                pm_opp_ct = c['pm2_femur'][sl][..., 2]
                a_new, a_old, edt_e = c['priors_roi']['a_new_pc'], c['priors_roi']['a_old_pc'], c['priors_roi']['edp_exp']

            cand, sc, sc_part, X_s3 = extract_s3_compartment_features(
                c_name, c['img'][sl], c['sp'], ds2, ds1, ds0, do2,
                df2, dt2, dp2, c[f'pm1_{c_name}'][sl], c[f'pm2_{c_name}'][sl],
                pm_opp_ct, a_new, a_old, edt_e
            )
            y_s3 = (c['lbl'][sl][cand] == c_lbl).astype(np.uint8)
            return cand, X_s3, y_s3

        tr_s3_list = Parallel(n_jobs=16, prefer='threads')(delayed(extract_case_s3)(c) for c in tr_cases)
        ev_s3_list = Parallel(n_jobs=16, prefer='threads')(delayed(extract_case_s3)(c) for c in ev_cases)

        X_s3_sub_list = []
        y_s3_sub_list = []
        b_s3 = [0]
        for i_c, (_, X_i, y_i) in enumerate(tr_s3_list):
            if len(y_i) > 2200:
                rng_s3 = np.random.RandomState(800 + i_c)
                sel_s3 = rng_s3.choice(len(y_i), 2200, replace=False)
                X_s3_sub_list.append(X_i[sel_s3])
                y_s3_sub_list.append(y_i[sel_s3])
                b_s3.append(b_s3[-1] + 2200)
            else:
                X_s3_sub_list.append(X_i)
                y_s3_sub_list.append(y_i)
                b_s3.append(b_s3[-1] + len(y_i))

        X_s3_tr = np.vstack(X_s3_sub_list)
        y_s3_tr = np.concatenate(y_s3_sub_list)
        del X_s3_sub_list, y_s3_sub_list
        print(f"  [{c_name.upper()} S3] Extracted {X_s3_tr.shape} in {time.time() - t_s3:.1f}s", flush=True)

        _MP_STATE['X'] = X_s3_tr
        _MP_STATE['y'] = y_s3_tr
        _MP_STATE['b'] = b_s3
        _MP_STATE['f_sz'] = f_sz
        with mp.get_context('fork').Pool(6) as pool:
            s3_hgb_all = pool.map(_fit_s3_model, range(6))
        _MP_STATE.clear()
        hgb_s3_1, hgb_s3_2 = s3_hgb_all[0], s3_hgb_all[1]
        oof_s3_folds = s3_hgb_all[2:6]

        rng_et3 = np.random.RandomState(503)
        sub_et3 = rng_et3.choice(len(y_s3_tr), min(len(y_s3_tr), 200000), replace=False)
        et_s3 = ExtraTreesClassifier(
            n_estimators=80, max_depth=22, min_samples_leaf=8, n_jobs=16, random_state=503
        ).fit(X_s3_tr[sub_et3], y_s3_tr[sub_et3])
        et_s3.n_jobs = 1
        del X_s3_tr, y_s3_tr
        gc.collect()

        # Keep OMP_NUM_THREADS=1 in the parent process so libgomp never spawns a team before the next fork()!
        with threadpool_limits(limits=1, user_api='openmp'):
            for i in range(n_tr):
                fk = i // f_sz
                cand, X_i, _ = tr_s3_list[i]
                p_oof = oof_s3_folds[fk].predict_proba(X_i)[:, 1].astype(np.float32) if len(X_i) > 0 else np.zeros(0, dtype=np.float32)
                tr_cases[i].setdefault('s3_scores', {})[c_name] = (cand, p_oof)

            for i in range(n_ev):
                cand, X_i, _ = ev_s3_list[i]
                if len(X_i) > 0:
                    p_ev = (
                        0.42 * hgb_s3_1.predict_proba(X_i)[:, 1]
                        + 0.38 * hgb_s3_2.predict_proba(X_i)[:, 1]
                        + 0.20 * et_s3.predict_proba(X_i)[:, 1]
                    ).astype(np.float32)
                else:
                    p_ev = np.zeros(0, dtype=np.float32)
                ev_cases[i].setdefault('s3_scores', {})[c_name] = (cand, p_ev)

        s3_models[c_name] = [hgb_s3_1, hgb_s3_2, et_s3]
        del tr_s3_list, ev_s3_list
        gc.collect()
        print(f"  [{c_name.upper()} S3] Fit & scored in {time.time() - t_s3:.1f}s", flush=True)

    # Optimize global postprocessing thresholds on 128-Train OOF ONLY (Zero eval leakage!)
    best_params = {'w_s3_f': 0.42, 'w_s3_t': 0.44, 'w_s3_p': 0.44, 'th_f': 0.49, 'th_t': 0.48, 'th_p': 0.48}
    for c_key, th_key, c_lbl in [('femur', 'th_f', 2), ('tibia', 'th_t', 4), ('patella', 'th_p', 6)]:
        best_th, best_score = best_params[th_key], -1.0
        for th_cand in np.linspace(0.42, 0.56, 15):
            test_p = dict(best_params)
            test_p[th_key] = float(th_cand)

            def _eval_tr_th(c):
                sl = c['sl']
                y_pr, _ = resolve_3cartilage_masks(
                    c['img'].shape, sl, c['sp'], c['dts'][0][sl], c['dts'][1][sl], c['dts'][2][sl],
                    c['pm1_femur'][sl], c['pm1_tibia'][sl], c['pm1_patella'][sl],
                    c['pm2_femur'][sl], c['pm2_tibia'][sl], c['pm2_patella'][sl],
                    c['priors_roi'], s3_scores=c['s3_scores'], params=test_p
                )
                gt = (c['lbl'] == c_lbl)
                pr = (y_pr == c_lbl)
                return float(2.0 * np.sum(gt & pr) / max(np.sum(gt) + np.sum(pr), 1))

            scores = Parallel(n_jobs=16, prefer='threads')(delayed(_eval_tr_th)(c) for c in tr_cases)
            m_sc = float(np.mean(scores))
            if m_sc > best_score:
                best_score = m_sc
                best_th = float(th_cand)
        best_params[th_key] = best_th
        print(f"  [128-Train OOF Calibrated {c_key}] Best {th_key} = {best_th:.4f} (OOF DSC = {best_score:.4f})", flush=True)

    # Evaluate on 31 Held-Out Eval Scans!
    dsc_eval = {2: [], 4: [], 6: []}
    for c in ev_cases:
        sl = c['sl']
        y_pr, _ = resolve_3cartilage_masks(
            c['img'].shape, sl, c['sp'], c['dts'][0][sl], c['dts'][1][sl], c['dts'][2][sl],
            c['pm1_femur'][sl], c['pm1_tibia'][sl], c['pm1_patella'][sl],
            c['pm2_femur'][sl], c['pm2_tibia'][sl], c['pm2_patella'][sl],
            c['priors_roi'], s3_scores=c['s3_scores'], params=best_params
        )
        for c_lbl in [2, 4, 6]:
            gt = (c['lbl'] == c_lbl)
            pr = (y_pr == c_lbl)
            dsc_eval[c_lbl].append(float(2.0 * np.sum(gt & pr) / max(np.sum(gt) + np.sum(pr), 1)))

    print("\n================ OAI 31-EVAL 3-CARTILAGE FINAL SUMMARY ================", flush=True)
    for c_lbl, name, old_b in [(2, 'Femoral Cartilage', 0.5171), (4, 'Tibial Cartilage', 0.4134), (6, 'Patellar Cartilage', 0.3633)]:
        arr = dsc_eval[c_lbl]
        print(f"  {name:20s}: DSC = {np.mean(arr):.4f} ± {np.std(arr):.4f} (min={min(arr):.4f}) | Baseline = {old_b:.4f} (Δ = +{np.mean(arr) - old_b:.4f})", flush=True)

    # Save cartilage_rf_p2.joblib
    joblib.dump({
        'format_version': 'oai_cartilage_v6_p2',
        'm_v6': {
            'atlas_priors': ap_full,
            'p1_folds': p1_folds,
            'p2_femur_hgb': p2_models['femur']['hgb'],
            'p2_femur_et': p2_models['femur']['et'],
            'p2_tibia_hgb': p2_models['tibia']['hgb'],
            'p2_tibia_et': p2_models['tibia']['et'],
            'p2_patella_hgb': p2_models['patella']['hgb'],
            'p2_patella_et': p2_models['patella']['et'],
        },
        's3_models': s3_models,
        'params': best_params,
    }, os.path.join(model_dir, 'cartilage_rf_p2.joblib'), compress=3)
    print(f"Saved cartilage_rf_p2.joblib in {time.time() - t_start:.1f}s!", flush=True)
