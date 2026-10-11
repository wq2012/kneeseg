import gc, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_EXP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)
from kneeseg.io import load_volume
from kneeseg.features import fast_signed_distance_transform
from train_cartilage_v6_ultimate import (
    fast_keep_top_k, compute_atlas_bins_v6,
    compute_case_atlas_hist_v6, make_atlas_priors_v6,
    prepare_geometry_v6, extract_features_v6, predict_p1_fast_strided_v6,
    d_grid, h_grid, dy_steps, r_steps
)

with open(os.path.join(_ROOT_DIR, 'kneeseg', 'data', 'ski10_full_split.json')) as f:
    split = json.load(f)
data_dir = os.environ.get('SKI10_DATA_DIR', '/path/to/SKI10/TrainingData')
lbl_dir = os.environ.get('SKI10_LABEL_DIR', data_dir + '_labels')
cache_dir = os.path.join(_EXP_DIR, 'cache')
model_dir = os.path.join(_EXP_DIR, 'models')


def prepare_geometry_tibia_v7(img, sp, bm_ref, bm_alt, bm_raw, atlas_p):
    geo = prepare_geometry_v6(img, sp, bm_ref, bm_alt, bm_raw, 'tibia', atlas_p)
    z0, z1, y0, y1, x0, x1 = geo['bbox']
    img_f32 = geo['img_f32']
    d_self_crop = geo['d_self_crop']
    d_other_crop = geo['d_other_crop']
    i_low, i_high = geo['i_low'], geo['i_high']

    # 1. Sharp 0.35mm and 0.65mm isotropic images + sharp subchondral z_bone
    img_f0_raw  = gaussian_filter(img_f32, sigma=[0.35 / s for s in sp])
    img_s06_raw = gaussian_filter(img_f32, sigma=[0.65 / s for s in sp])
    sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
    w_bn = ((d_self_crop <= -1.4) & (d_self_crop >= -5.5)).astype(np.float32) + 1e-5
    w_bn_s = gaussian_filter(w_bn, sigma=sig_bn)
    i_bn_mean = gaussian_filter(img_f0_raw * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_var  = gaussian_filter(((img_f0_raw - i_bn_mean) ** 2) * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_std  = np.maximum(np.sqrt(np.maximum(i_bn_var, 0.0)), max(0.05 * (i_high - i_low), 2.5))
    z_bone_sharp = ((img_f0_raw - i_bn_mean) / i_bn_std).astype(np.float32)

    m_low = (d_self_crop <= -1.5) & (d_self_crop >= -5.5)
    m_high = (d_self_crop >= 0.4) & (d_self_crop <= 1.9) & (d_other_crop <= 9.5)
    zb_med_glob = max(float(np.median(z_bone_sharp[m_high])), 1.0) if np.any(m_high) else 2.5

    # 2. Compartment-Local (Medial vs Lateral) Intensity & Z-Bone Calibration
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
            ih, im, iq, zm = i_high, float(np.median(img_f32[m_high])) if np.any(m_high) else i_high, 25.0, zb_med_glob
        i_low_c.append(il); i_high_c.append(ih); i_med_c.append(im); iqr_c.append(iq); zb_med_c.append(zm)

    geo['z_bone_sharp'] = z_bone_sharp
    geo['zb_med_glob'] = zb_med_glob
    geo['w_high_z'] = w_high_z
    geo['comp_masks'] = comp_masks
    geo['comp_stats'] = (i_low_c, i_high_c, i_med_c, iqr_c, zb_med_c)
    return geo


def extract_features_tibia_v7(geo, target_mask, pm_self=None, pm_other=None, rng_jit=None):
    z0, z1, y0, y1, x0, x1 = geo['bbox']
    sp = geo['sp']
    inds = np.argwhere(target_mask)
    zc, yc, xc = inds[:, 0] - z0, inds[:, 1] - y0, inds[:, 2] - x0
    coords_mm = inds.astype(np.float32) * np.array(sp, dtype=np.float32)
    N = len(zc)

    d_raw_m  = geo['d_self_crop'][zc, yc, xc]
    d_alt_m  = geo['d_alt_self_crop'][zc, yc, xc]
    d_st2_m  = geo['d_raw_self_crop'][zc, yc, xc]
    d_s_m    = geo['ds_self_crop'][zc, yc, xc]
    do_raw_m = geo['d_other_crop'][zc, yc, xc]
    do_s_m   = geo['ds_other_crop'][zc, yc, xc]

    shift_ref_alt = d_raw_m - d_alt_m
    shift_ref_st2 = d_raw_m - d_st2_m

    # True De-Stepped Subchondral Jitter: unattenuated +-0.52 mm across |d_raw_m| <= 0.40 mm,
    # with identical 1.0 * delta_d across ALL self-bone distances!
    if rng_jit is not None:
        jit = rng_jit.uniform(-0.52, 0.52, size=N).astype(np.float32)
        w_jit = np.clip(1.36 - np.abs(d_raw_m) / 1.1, 0.15, 1.0)
        delta_d = w_jit * jit
        d_raw_feat = d_raw_m + delta_d
        d_alt_feat = d_alt_m + delta_d
        d_s_feat   = d_s_m + delta_d
        d_s_anchor = d_s_feat
    else:
        delta_d = None
        d_raw_feat = d_raw_m
        d_alt_feat = d_alt_m
        d_s_feat   = d_s_m
        d_s_anchor = d_s_m

    ap = geo['atlas_p']
    bins, c_dict = compute_atlas_bins_v6(coords_mm, do_s_m, d_s_anchor, geo['f_stats'], geo['t_comp'], geo['ml_info'])

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
                i_med_c[c_i]  = float(np.median(vh))
                iqr_c[c_i]    = max(float(np.percentile(vh, 75) - np.percentile(vh, 25)), 1.0)
                zb_med_c[c_i] = max(float(np.median(z_bone_sharp[mc])), 1.0)

    inorm = (img_f32 - i_low) / max(i_high - i_low, 1.0)
    z_cart = geo['z_cart']
    abs_z = gaussian_filter(np.abs(z_cart), sigma=[1.0 / sp[0], 0.35 / sp[1], 1.0 / sp[2]])

    # Compartment-calibrated dimensionless intensity & z_bone maps
    i_low_2d  = (1.0 - w_high_z) * i_low_c[0]  + w_high_z * i_low_c[1]
    i_high_2d = (1.0 - w_high_z) * i_high_c[0] + w_high_z * i_high_c[1]
    i_med_2d  = (1.0 - w_high_z) * i_med_c[0]  + w_high_z * i_med_c[1]
    iqr_2d    = (1.0 - w_high_z) * iqr_c[0]    + w_high_z * iqr_c[1]
    zb_med_2d = (1.0 - w_high_z) * zb_med_c[0] + w_high_z * zb_med_c[1]

    inorm_comp   = ((img_f32 - i_low_2d) / np.maximum(i_high_2d - i_low_2d, 1.0)).astype(np.float32)
    z_cart_comp  = ((img_f32 - i_med_2d) / np.maximum(iqr_2d, 1.0)).astype(np.float32)
    zb_norm_glob = (z_bone_sharp / zb_med_glob).astype(np.float32)
    zb_norm_comp = (z_bone_sharp / np.maximum(zb_med_2d, 1.0)).astype(np.float32)

    img_f0   = gaussian_filter(inorm, sigma=[0.35 / s for s in sp])
    img_s06  = gaussian_filter(inorm, sigma=[0.65 / s for s in sp])
    img_lat  = gaussian_filter(inorm, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
    img_s1   = gaussian_filter(inorm, sigma=[1.0 / s for s in sp])
    img_s2   = gaussian_filter(inorm, sigma=[2.0 / s for s in sp])
    img_s4   = gaussian_filter(inorm, sigma=[4.0 / s for s in sp])
    img_s8   = gaussian_filter(inorm, sigma=[8.0 / s for s in sp])
    icomp_f0 = gaussian_filter(inorm_comp, sigma=[0.35 / s for s in sp])
    icomp_s06 = gaussian_filter(inorm_comp, sigma=[0.65 / s for s in sp])

    gz0, gy0, gx0 = np.gradient(img_f0, *sp)
    gmag0 = np.sqrt(gz0**2 + gy0**2 + gx0**2)
    gz1, gy1, gx1 = np.gradient(img_s1, *sp)
    gmag1 = np.sqrt(gz1**2 + gy1**2 + gx1**2)
    gzb, gyb, gxb = np.gradient(zb_norm_comp, *sp)
    gmag_zb = np.sqrt(gzb**2 + gyb**2 + gxb**2)

    nz, ny, nx = geo['normals_self']
    nzo, nyo, nxo = geo['normals_other']
    nz_m, ny_m, nx_m = nz[zc, yc, xc], ny[zc, yc, xc], nx[zc, yc, xc]
    nzo_m, nyo_m, nxo_m = nzo[zc, yc, xc], nyo[zc, yc, xc], nxo[zc, yc, xc]
    cz_max, cy_max, cx_max = img_f32.shape

    K = len(d_grid)
    rv0    = np.empty((N, K), dtype=np.float32)
    rv_s06 = np.empty((N, K), dtype=np.float32)
    rv_lat = np.empty((N, K), dtype=np.float32)
    rv1    = np.empty((N, K), dtype=np.float32)
    rv_zb  = np.empty((N, K), dtype=np.float32)
    rv_zbn = np.empty((N, K), dtype=np.float32)
    for k, d_val in enumerate(d_grid):
        sz = np.clip(np.round(zc + d_val * nz_m / sp[0]).astype(int), 0, cz_max - 1)
        sy = np.clip(np.round(yc + d_val * ny_m / sp[1]).astype(int), 0, cy_max - 1)
        sx = np.clip(np.round(xc + d_val * nx_m / sp[2]).astype(int), 0, cx_max - 1)
        rv0[:, k]    = img_f0[sz, sy, sx]
        rv_s06[:, k] = icomp_s06[sz, sy, sx]
        rv_lat[:, k] = img_lat[sz, sy, sx]
        rv1[:, k]    = img_s1[sz, sy, sx]
        rv_zb[:, k]  = z_bone[sz, sy, sx]
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
    rg_zb = np.diff(rv_zb, axis=1)
    rg_zbn = np.diff(rv_zbn, axis=1)

    idx_rise = np.argmax(rg_lat[:, 2:11], axis=1).astype(np.float32) - 4.0
    max_rise = np.max(rg_lat[:, 2:11], axis=1)
    idx_drop = np.argmin(rg_lat[:, 5:14], axis=1).astype(np.float32) - 2.0
    min_drop = np.min(rg_lat[:, 5:14], axis=1)

    # Sharp 0.35mm & Compartment-Calibrated Z-Bone Edge Detectors (unblurred on sloped walls & subchondral step!)
    idx_rise_0   = np.argmax(rg0[:, 2:11], axis=1).astype(np.float32) - 4.0
    max_rise_0   = np.max(rg0[:, 2:11], axis=1)
    idx_drop_0   = np.argmin(rg0[:, 5:14], axis=1).astype(np.float32) - 2.0
    min_drop_0   = np.min(rg0[:, 5:14], axis=1)
    idx_cort_min = np.argmin(rv_zbn[:, 3:10], axis=1).astype(np.float32) - 4.0
    val_cort_min = np.min(rv_zbn[:, 3:10], axis=1)
    idx_rise_zb  = np.argmax(rg_zbn[:, 2:11], axis=1).astype(np.float32) - 4.0
    max_rise_zb  = np.max(rg_zbn[:, 2:11], axis=1)
    idx_drop_zb  = np.argmin(rg_zbn[:, 5:14], axis=1).astype(np.float32) - 2.0
    min_drop_zb  = np.min(rg_zbn[:, 5:14], axis=1)

    # 9-point Vertical (Y-axis) Column Profile: BOTH sharp (0.35mm) and lateral-smoothed
    vy_0   = np.empty((N, len(dy_steps)), dtype=np.float32)
    vy_lat = np.empty((N, len(dy_steps)), dtype=np.float32)
    vy_zb  = np.empty((N, len(dy_steps)), dtype=np.float32)
    vy_zbn = np.empty((N, len(dy_steps)), dtype=np.float32)
    for k, dy_px in enumerate(dy_steps):
        sy = np.clip(yc + dy_px, 0, cy_max - 1)
        vy_0[:, k]   = icomp_f0[zc, sy, xc]
        vy_lat[:, k] = img_lat[zc, sy, xc]
        vy_zb[:, k]  = z_bone[zc, sy, xc]
        vy_zbn[:, k] = zb_norm_comp[zc, sy, xc]
    vyn_lat = (vy_lat - r_in) / r_span
    vyg_0   = np.diff(vy_0, axis=1)
    vyg_lat = np.diff(vy_lat, axis=1)
    vyg_zbn = np.diff(vy_zbn, axis=1)

    # Surface-Anchored Height Profile (anchored at d_s_anchor)
    vh_0   = np.empty((N, len(h_grid)), dtype=np.float32)
    vh_lat = np.empty((N, len(h_grid)), dtype=np.float32)
    vh_zbn = np.empty((N, len(h_grid)), dtype=np.float32)
    for k, h_val in enumerate(h_grid):
        st = h_val - d_s_anchor
        sz = np.clip(np.round(zc + st * nz_m / sp[0]).astype(int), 0, cz_max - 1)
        sy = np.clip(np.round(yc + st * ny_m / sp[1]).astype(int), 0, cy_max - 1)
        sx = np.clip(np.round(xc + st * nx_m / sp[2]).astype(int), 0, cx_max - 1)
        vh_0[:, k]   = icomp_f0[sz, sy, sx]
        vh_lat[:, k] = img_lat[sz, sy, sx]
        vh_zbn[:, k] = zb_norm_comp[sz, sy, sx]
    gh_lat = np.diff(vh_lat, axis=1)
    gh_0   = np.diff(vh_0, axis=1)

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
    rad_vlat = np.empty((N, len(r_steps)), dtype=np.float32)
    rad_zbn  = np.empty((N, len(r_steps)), dtype=np.float32)
    for k, r_mm in enumerate(r_steps):
        sz = np.clip(np.round(zc + r_mm * u_rz / sp[0]).astype(int), 0, cz_max - 1)
        sx = np.clip(np.round(xc + r_mm * u_rx / sp[2]).astype(int), 0, cx_max - 1)
        rad_vlat[:, k] = img_lat[sz, yc, sx]
        rad_zbn[:, k]  = zb_norm_comp[sz, yc, sx]
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

        # Shift Stage-1 coordinate lookup in lockstep with delta_d during training so Stage-2
        # cannot use Stage-1's step function at dt2==0 to bypass MRI intensity gradients!
        if delta_d is not None:
            zc_p = np.clip(np.round(zc + delta_d * nz_m / sp[0]).astype(int), 0, cz_max - 1)
            yc_p = np.clip(np.round(yc + delta_d * ny_m / sp[1]).astype(int), 0, cy_max - 1)
            xc_p = np.clip(np.round(xc + delta_d * nx_m / sp[2]).astype(int), 0, cx_max - 1)
        else:
            zc_p, yc_p, xc_p = zc, yc, xc

        # Spatially-Varying Local Sheet-Calibrated Intensity (evaluated at true physical voxel zc, yc, xc!)
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

        # 1) Vertical 2D Sheet Cumulative Cartilage Thickness + Smooth Column Centroid & Bounds
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

        # Smooth 2D Column Centroid (y_p1_mean_s2) and 2D Horizontal Signed Distance to Pass-1 Sheet
        w_col = np.maximum(p_ct_c * valid_c - 0.12, 0.0)
        w_sum_2d = np.sum(w_col, axis=1) + 1e-4
        yy_col = np.arange(cy_max, dtype=np.float32)[None, :, None]
        y_p1_mean = np.sum(w_col * yy_col, axis=1) / w_sum_2d
        y_p1_var = np.sum(w_col * ((yy_col - y_p1_mean[:, None, :]) ** 2), axis=1) / w_sum_2d
        y_p1_std = np.sqrt(np.maximum(y_p1_var, 0.25))
        w_sh = (pmax_col_2d[:, 0, :] >= 0.30).astype(np.float32) + 1e-4
        w_sh_s2 = gaussian_filter(w_sh, sigma=[2.2 / sp[0], 2.2 / sp[2]])
        y_p1_mean_s2 = gaussian_filter(y_p1_mean * w_sh, sigma=[2.2 / sp[0], 2.2 / sp[2]]) / w_sh_s2
        y_p1_std_s2  = gaussian_filter(y_p1_std * w_sh, sigma=[2.2 / sp[0], 2.2 / sp[2]]) / w_sh_s2

        m_t1_40 = pmax_col_2d[:, 0, :] >= 0.40
        m_t1_28 = pmax_col_2d[:, 0, :] >= 0.28
        sd_t1_40 = fast_signed_distance_transform(m_t1_40, (sp[0], sp[2])) if (np.any(m_t1_40) and not np.all(m_t1_40)) else np.zeros_like(y_p1_mean)
        sd_t1_28 = fast_signed_distance_transform(m_t1_28, (sp[0], sp[2])) if (np.any(m_t1_28) and not np.all(m_t1_28)) else np.zeros_like(y_p1_mean)

        # 2) Overhead Femoral Cartilage 2D Column Footprint (across full Y of volume!)
        if pm_other is not None:
            pf_full_zx = pm_other[z0:z1, :, x0:x1, 2].astype(np.float32)
            pf_col_max = np.max(pf_full_zx, axis=1)
            pf_col_sum = np.sum(pf_full_zx, axis=1) * sp[1]
            pf_col_s2  = gaussian_filter(pf_col_max, sigma=[2.0 / sp[0], 2.0 / sp[2]])
            pf_col_s4  = gaussian_filter(pf_col_max, sigma=[4.0 / sp[0], 4.0 / sp[2]])
            pf_sum_s2  = gaussian_filter(pf_col_sum, sigma=[2.0 / sp[0], 2.0 / sp[2]])
            has_fc_col = pf_col_max >= 0.30
            y_from_bot = np.argmax(pf_full_zx[:, ::-1, :] >= 0.30, axis=1)
            y_fc_bot_2d = np.where(has_fc_col, (pf_full_zx.shape[1] - 1) - y_from_bot, -20).astype(np.float32)
            m_fc_foot = pf_col_s2 >= 0.30
            sd_fc_foot = fast_signed_distance_transform(m_fc_foot, (sp[0], sp[2])) if (np.any(m_fc_foot) and not np.all(m_fc_foot)) else np.zeros_like(pf_col_s2)
        else:
            pf_col_max = np.zeros((cz_max, cx_max), dtype=np.float32)
            pf_col_sum = np.zeros((cz_max, cx_max), dtype=np.float32)
            pf_col_s2  = np.zeros((cz_max, cx_max), dtype=np.float32)
            pf_col_s4  = np.zeros((cz_max, cx_max), dtype=np.float32)
            pf_sum_s2  = np.zeros((cz_max, cx_max), dtype=np.float32)
            has_fc_col = np.zeros((cz_max, cx_max), dtype=bool)
            y_fc_bot_2d = np.full((cz_max, cx_max), -20.0, dtype=np.float32)
            sd_fc_foot = np.zeros((cz_max, cx_max), dtype=np.float32)

        y_glob_p = yc_p + y0
        dy_to_fc_mm = np.where(has_fc_col[zc_p, xc_p], (y_glob_p - y_fc_bot_2d[zc_p, xc_p]) * sp[1], 22.0).astype(np.float32)

        # 3) Normal ray profile of Pass-1 probabilities (BOTH sharp 0.45mm and lateral-smoothed!)
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
            # Image intensity profile stays anchored at true physical voxel (zc, yc, xc)!
            sz_i = np.clip(np.round(zc + st_val * nz_m / sp[0]).astype(int), 0, cz_max - 1)
            sy_i = np.clip(np.round(yc + st_val * ny_m / sp[1]).astype(int), 0, cy_max - 1)
            sx_i = np.clip(np.round(xc + st_val * nx_m / sp[2]).astype(int), 0, cx_max - 1)
            pr_inloc[:, k] = in_loc_lat[sz_i, sy_i, sx_i]

        c_in_ray  = np.sum(pr_ct[:, 0:5], axis=1) * 0.40
        c_out_ray = np.sum(pr_ct[:, 5:10], axis=1) * 0.42
        t_ray_tot = c_in_ray + c_out_ray
        f_in_ray  = c_in_ray / (t_ray_tot + 0.20)

        c_in_ray_sh  = np.sum(pr_ct_sh[:, 0:5], axis=1) * 0.40
        c_out_ray_sh = np.sum(pr_ct_sh[:, 5:10], axis=1) * 0.42
        t_ray_tot_sh = c_in_ray_sh + c_out_ray_sh
        f_in_ray_sh  = c_in_ray_sh / (t_ray_tot_sh + 0.20)

        # 4) Vertical (Y) 9-point Pass-1 probability column profile (in shifted Stage-1 frame yc_p)
        py_ct = np.empty((N, len(dy_steps)), dtype=np.float32)
        py_bn = np.empty((N, len(dy_steps)), dtype=np.float32)
        for k, dy_px in enumerate(dy_steps):
            sy_p = np.clip(yc_p + dy_px, 0, cy_max - 1)
            py_ct[:, k] = p_ct_lat[zc_p, sy_p, xc_p]
            py_bn[:, k] = p_bn_c[zc_p, sy_p, xc_p]

        # 5) Surface-anchored column profile of Pass-1 cartilage probability (consistent with zc_p, yc_p, xc_p!)
        ph_ct = np.empty((N, len(h_grid)), dtype=np.float32)
        ph_ot = np.empty((N, len(h_grid)), dtype=np.float32)
        for k, h_val in enumerate(h_grid):
            st = h_val - d_s_m
            sz_p = np.clip(np.round(zc + st * nz_m / sp[0]).astype(int), 0, cz_max - 1)
            sy_p = np.clip(np.round(yc + st * ny_m / sp[1]).astype(int), 0, cy_max - 1)
            sx_p = np.clip(np.round(xc + st * nx_m / sp[2]).astype(int), 0, cx_max - 1)
            ph_ct[:, k] = p_ct_lat[sz_p, sy_p, sx_p]
            ph_ot[:, k] = p_ot_s1[sz_p, sy_p, sx_p]
        col_psum = np.sum(ph_ct[:, 2:], axis=1) * 0.45
        col_pmax = np.max(ph_ct[:, 2:], axis=1)
        col_otsum = np.sum(ph_ot[:, 2:], axis=1) * 0.45

        # 6) 8-point Tangential sheet probability profile
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
        blocks.extend([p_scalar_cols, pr_ct, pr_ct_sh, pr_bn, pr_bg, pr_ot, pr_inloc, py_ct, py_bn, ph_ct, ph_ot, ptan_ct])

    return np.hstack(blocks).astype(np.float32, copy=False)


def make_stratified_sample_tibia_v7(lbl, geo, rng):
    """Preserves exact class ratios (5600 : 3200 : 4000 : 900 scaled by 1.2x = 6720 : 3840 : 4800 : 1080)
    while de-stepping the subchondral boundary (p_sub_c & p_surf_b_out) and boosting Z56 sloped rim/wall."""
    shell = geo['shell']
    z0, z1, y0, y1, x0, x1 = geo['bbox']
    sp = geo['sp']

    dc = fast_signed_distance_transform(lbl == 4, sp)
    d_self = np.full(lbl.shape, 100.0, dtype=np.float32)
    d_self[z0:z1, y0:y1, x0:x1] = geo['d_self_crop']
    ny_full = np.full(lbl.shape, -1.0, dtype=np.float32)
    ny_full[z0:z1, y0:y1, x0:x1] = geo['normals_self'][1]
    is_sloped = np.abs(ny_full) < 0.72

    y4 = np.zeros_like(lbl, dtype=np.uint8)
    y4[lbl == 3] = 1
    y4[lbl == 4] = 2
    y4[lbl == 2] = 3

    sm = np.zeros_like(shell, dtype=bool)
    def _take(pool, n_want):
        if len(pool) == 0 or n_want <= 0:
            return 0
        n_got = min(len(pool), n_want)
        chosen = pool[rng.choice(len(pool), n_got, replace=False)]
        sm[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
        return n_got

    # 1. Cartilage (Total = 6,720: 2,400 base + 4,320 core)
    p_sub_c  = np.argwhere(shell & (y4 == 2) & (d_self <= 0.15))
    p_base_c = np.argwhere(shell & (y4 == 2) & (d_self > 0.15) & (d_self <= 0.65))
    n_sub = _take(p_sub_c, 950)
    _take(p_base_c, 2400 - n_sub)

    p_rim_c  = np.argwhere(shell & (y4 == 2) & (d_self > 0.65) & is_sloped)
    p_core_c = np.argwhere(shell & (y4 == 2) & (d_self > 0.65) & (~is_sloped))
    n_rim_c = _take(p_rim_c, 1500)
    _take(p_core_c, 4320 - n_rim_c)

    # 2. Bone (Total = 3,840: 2,400 surface + 1,440 deep)
    p_surf_b_out = np.argwhere(shell & (y4 == 1) & (d_self > 0.05) & (dc <= 2.4))
    p_surf_b_in  = np.argwhere(shell & (y4 == 1) & (d_self >= -0.20) & (d_self <= 0.05) & (dc <= 2.2))
    p_deep_b     = np.argwhere(shell & (y4 == 1) & (d_self < -0.20) & (dc <= 2.2))
    n_b_out = _take(p_surf_b_out, 850)
    _take(p_surf_b_in, 2400 - n_b_out)
    _take(p_deep_b, 1440)

    # 3. Background (Total = 4,800: 3,480 hard + 1,320 outer)
    p_rim_bg   = np.argwhere(shell & (y4 == 0) & (dc <= 1.8) & is_sloped)
    p_hard_bg  = np.argwhere(shell & (y4 == 0) & (dc <= 1.8) & (~is_sloped))
    p_outer_bg = np.argwhere(shell & (y4 == 0) & (dc > 1.8))
    n_rim_bg = _take(p_rim_bg, 1400)
    _take(p_hard_bg, 3480 - n_rim_bg)
    _take(p_outer_bg, 1320)

    # 4. Other Cartilage (Total = 1,080)
    p_other_c = np.argwhere(shell & (y4 == 3) & (d_self <= 3.5))
    _take(p_other_c, 1080)

    return sm, y4[sm]


def predict_p1_fast_strided_tibia_v7(geo, clf, sm_exact=None, p_exact=None):
    sh = geo['shell']
    sub_mask = np.zeros_like(sh, dtype=bool)
    sub_mask[:, :, ::2] = sh[:, :, ::2]
    n_x = sh.shape[2]
    even_lim = n_x - 1
    odd_only = np.zeros_like(sh, dtype=bool)
    odd_only[:, :, 1::2] = sh[:, :, 1::2] & (~sh[:, :, 0:even_lim:2])
    sub_mask |= odd_only

    X_sub = extract_features_tibia_v7(geo, sub_mask, rng_jit=None)
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


if __name__ == '__main__':
    t0 = time.time()
    train_v1  = joblib.load(os.path.join(cache_dir, 'train80_ref_bones.joblib'))
    eval_v1   = joblib.load(os.path.join(cache_dir, 'eval_ref_bones_80.joblib'))
    train_v2  = joblib.load(os.path.join(cache_dir, 'train80_ref_bones_v2.joblib'))
    eval_v2   = joblib.load(os.path.join(cache_dir, 'eval_ref_bones_v2.joblib'))
    train_raw = joblib.load(os.path.join(cache_dir, 'train80_pred_bones.joblib'))
    eval_raw  = joblib.load(os.path.join(cache_dir, 'eval_pred_bones.joblib'))

    m_v6 = joblib.load(os.path.join(model_dir, 'cart_v6_refv2_models.joblib'))
    ev_v6 = joblib.load(os.path.join(cache_dir, 'ev_v6_refv2_rois.joblib'))
    ev_v5 = joblib.load(os.path.join(cache_dir, 'ev_v5_refv1_rois.joblib'))
    atlas_priors = m_v6['atlas_priors']
    p1_folds_v6 = m_v6['p1_folds']

    def load_case_all(fname, d_v1, d_v2, d_raw):
        img, sp = load_volume(os.path.join(data_dir, fname), return_spacing=True)
        lbl = load_volume(os.path.join(lbl_dir, fname.replace('image-', 'labels-')), return_spacing=False).astype(np.uint8)
        lbl[lbl == 5] = 1
        lbl[lbl == 6] = 3
        return fname, img, lbl, sp, d_v1[fname], d_v2[fname], d_raw[fname]

    tr_cases = Parallel(n_jobs=16, prefer='threads')(delayed(load_case_all)(f, train_v1, train_v2, train_raw) for f in split['train'])
    ev_cases = Parallel(n_jobs=16, prefer='threads')(delayed(load_case_all)(f, eval_v1, eval_v2, eval_raw) for f in split['eval'])
    print(f"Loaded 80 train + 20 eval cases + cached v6 models in {time.time()-t0:.1f}s", flush=True)

    # 1. Build 80-case Leave-One-Out (LOO) Multi-Atlases on bm_v2
    t1 = time.time()
    case_hists = Parallel(n_jobs=16, prefer='threads')(
        delayed(compute_case_atlas_hist_v6)(img, bm_v2, sp, lbl)
        for fname, img, lbl, sp, bm_v1, bm_v2, bm_raw in tr_cases
    )
    n_h = len(case_hists[0])
    tot_hists = [sum(h[k] for h in case_hists) for k in range(n_h)]
    loo_atlases = [make_atlas_priors_v6([tot_hists[k] - h[k] for k in range(n_h)]) for h in case_hists]
    del case_hists, tot_hists; gc.collect()
    print(f"[Tibia V7] Built 80-case LOO Multi-Atlases in {time.time()-t1:.1f}s", flush=True)

    # 2. Step 1: Sample Tibia V7 Pass-1 features and fit 4-Fold Out-Of-Fold P1 HGB
    t_b = time.time()
    def sample_p1_v7(i):
        fname, img, lbl, sp, bm_v1, bm_v2, bm_raw = tr_cases[i]
        geo = prepare_geometry_tibia_v7(img, sp, bm_v2, bm_v1, bm_raw, loo_atlases[i])
        sm, y_s = make_stratified_sample_tibia_v7(lbl, geo, np.random.RandomState(42 + i))
        X_s = extract_features_tibia_v7(geo, sm, rng_jit=np.random.RandomState(142 + i))
        return X_s, y_s

    res_p1 = Parallel(n_jobs=16, prefer='threads')(delayed(sample_p1_v7)(i) for i in range(80))
    p1_folds_v7 = []
    for f_k in range(4):
        s_f, e_f = f_k * 20, (f_k + 1) * 20
        tr_idx = [i for i in range(80) if not (s_f <= i < e_f)]
        X_f = np.vstack([res_p1[i][0] for i in tr_idx])
        y_f = np.concatenate([res_p1[i][1] for i in tr_idx])
        clf_f = HistGradientBoostingClassifier(
            max_iter=250, max_leaf_nodes=63, min_samples_leaf=25,
            learning_rate=0.055, max_features=0.70, l2_regularization=1.0,
            early_stopping=False, random_state=42 + f_k
        ).fit(X_f, y_f)
        p1_folds_v7.append(clf_f)
        del X_f, y_f; gc.collect()
    del res_p1; gc.collect()
    print(f"[Tibia V7] Sampled & Fit 4-Fold P1 HGB in {time.time()-t_b:.1f}s", flush=True)

    # 3. Step 2: Generate OOF P1 probability maps (blending v6 + v7 P1 for Tibia, plus v6 P1 for Femur) & extract P2 training set
    t_s2 = time.time()
    Xt2_list, yt2_list = [], []
    is_lat_list, is_rim_list = [], []

    for start in range(0, 80, 16):
        idxs = list(range(start, min(start + 16, 80)))
        def _proc_tr_p2_v7(i):
            fname, img, lbl, sp, bm_v1, bm_v2, bm_raw = tr_cases[i]
            geo_t7 = prepare_geometry_tibia_v7(img, sp, bm_v2, bm_v1, bm_raw, loo_atlases[i])
            geo_f6 = prepare_geometry_v6(img, sp, bm_v2, bm_v1, bm_raw, 'femur', loo_atlases[i])

            clf_t6 = p1_folds_v6['tibia'][i // 20]
            clf_t7 = p1_folds_v7[i // 20]
            clf_f6 = p1_folds_v6['femur'][i // 20]

            sm_t, y_t = make_stratified_sample_tibia_v7(lbl, geo_t7, np.random.RandomState(542 + i))

            # Exact at sm_t + fast strided elsewhere for both v6 and v7 Tibia P1!
            Xt6_ex = extract_features_v6(geo_t7, sm_t, rng_jit=None)
            pt6_ex = clf_t6.predict_proba(Xt6_ex).astype(np.float32)
            del Xt6_ex
            pm_t6 = predict_p1_fast_strided_v6(geo_t7, clf_t6, sm_t, pt6_ex)

            Xt7_ex = extract_features_tibia_v7(geo_t7, sm_t, rng_jit=None)
            pt7_ex = clf_t7.predict_proba(Xt7_ex).astype(np.float32)
            del Xt7_ex
            pm_t7 = predict_p1_fast_strided_tibia_v7(geo_t7, clf_t7, sm_t, pt7_ex)

            pm_t = 0.45 * pm_t6 + 0.55 * pm_t7
            del pm_t6, pm_t7

            pm_f = predict_p1_fast_strided_v6(geo_f6, clf_f6)
            Xt_2 = extract_features_tibia_v7(geo_t7, sm_t, pm_self=pm_t, pm_other=pm_f, rng_jit=np.random.RandomState(642 + i))

            # Track compartment (Medial vs Lateral) and Rim/Subchondral indicators for specialized ensemble members
            z0, z1, y0, y1, x0, x1 = geo_t7['bbox']
            inds_s = np.argwhere(sm_t)
            zc_s, yc_s, xc_s = inds_s[:, 0] - z0, inds_s[:, 1] - y0, inds_s[:, 2] - x0
            med_low_z = geo_t7['ml_info'][0]
            z_med_split = geo_t7['ml_info'][2][4]
            in_low = (inds_s[:, 0] * sp[0]) < z_med_split
            is_lat_s = (~in_low) if med_low_z else in_low
            ny_s = geo_t7['normals_self'][1][zc_s, yc_s, xc_s]
            dt_s = geo_t7['d_self_crop'][zc_s, yc_s, xc_s]
            is_rim_s = (np.abs(ny_s) < 0.75) | (np.abs(dt_s) <= 0.65)
            return Xt_2, y_t, is_lat_s, is_rim_s

        batch = Parallel(n_jobs=len(idxs), prefer='threads')(delayed(_proc_tr_p2_v7)(i) for i in idxs)
        for xt, yt, ilat, irim in batch:
            Xt2_list.append(xt); yt2_list.append(yt)
            is_lat_list.append(ilat); is_rim_list.append(irim)
        del batch; gc.collect()
        print(f"  [Tibia V7] Extracted P2 OOF train cases {min(start+16, 80)}/80 in {time.time()-t_s2:.1f}s", flush=True)

    # 4. Step 3: Fit 4-Fold OOF Pass-2 HGBs + Full-80 Pass-2 Multi-Family + Compartment/Rim-Specialized Ensemble for Tibia
    t_oof2 = time.time()
    p2_tibia_oof_folds = []
    for f_k in range(4):
        s_f, e_f = f_k * 20, (f_k + 1) * 20
        tr_idx = [i for i in range(80) if not (s_f <= i < e_f)]
        X_f = np.vstack([Xt2_list[i] for i in tr_idx])
        y_f = np.concatenate([yt2_list[i] for i in tr_idx])
        ilat_f = np.concatenate([is_lat_list[i] for i in tr_idx])
        irim_f = np.concatenate([is_rim_list[i] for i in tr_idx])
        if len(y_f) > 420000:
            sub_idx = np.random.RandomState(400 + f_k).choice(len(y_f), 420000, replace=False)
            X_f, y_f, ilat_f, irim_f = X_f[sub_idx], y_f[sub_idx], ilat_f[sub_idx], irim_f[sub_idx]

        h1_f = HistGradientBoostingClassifier(
            max_iter=240, max_leaf_nodes=63, min_samples_leaf=30,
            learning_rate=0.05, max_features=0.65, l2_regularization=1.0,
            early_stopping=False, random_state=310 + f_k
        ).fit(X_f, y_f)
        hm_f = HistGradientBoostingClassifier(
            max_iter=230, max_leaf_nodes=71, min_samples_leaf=30,
            learning_rate=0.05, max_features=0.65, l2_regularization=1.2,
            early_stopping=False, random_state=330 + f_k
        ).fit(X_f, y_f, sample_weight=np.where(~ilat_f, 2.5, 0.8).astype(np.float32))
        hl_f = HistGradientBoostingClassifier(
            max_iter=230, max_leaf_nodes=71, min_samples_leaf=30,
            learning_rate=0.05, max_features=0.65, l2_regularization=1.2,
            early_stopping=False, random_state=340 + f_k
        ).fit(X_f, y_f, sample_weight=np.where(ilat_f, 2.5, 0.8).astype(np.float32))
        hr_f = HistGradientBoostingClassifier(
            max_iter=240, max_leaf_nodes=79, min_samples_leaf=30,
            learning_rate=0.05, max_features=0.65, l2_regularization=1.2,
            early_stopping=False, random_state=350 + f_k
        ).fit(X_f, y_f, sample_weight=np.where(irim_f, 2.2, 0.8).astype(np.float32))
        p2_tibia_oof_folds.append((h1_f, h1_f, h1_f, hm_f, hl_f, hr_f))
        del X_f, y_f, ilat_f, irim_f; gc.collect()
    print(f"[Tibia V7] Fit 4-Fold OOF P2 HGB ensemble in {time.time()-t_oof2:.1f}s", flush=True)

    t_fit2 = time.time()
    X2_tr = np.vstack(Xt2_list)
    y2_tr = np.concatenate(yt2_list)
    is_lat_tr = np.concatenate(is_lat_list)
    is_rim_tr = np.concatenate(is_rim_list)
    Xt2_list.clear(); yt2_list.clear(); is_lat_list.clear(); is_rim_list.clear(); gc.collect()
    print(f"[Tibia V7] Training matrix X2_tr: {X2_tr.shape} ({X2_tr.nbytes / 1e9:.2f} GB)", flush=True)

    hgb2_1 = HistGradientBoostingClassifier(
        max_iter=270, max_leaf_nodes=63, min_samples_leaf=30,
        learning_rate=0.05, max_features=0.55, l2_regularization=1.0,
        early_stopping=False, random_state=301
    ).fit(X2_tr, y2_tr)
    hgb2_2 = HistGradientBoostingClassifier(
        max_iter=270, max_leaf_nodes=79, min_samples_leaf=35,
        learning_rate=0.05, max_features=0.70, l2_regularization=1.0,
        early_stopping=False, random_state=302
    ).fit(X2_tr, y2_tr)
    hgb2_3 = HistGradientBoostingClassifier(
        max_iter=270, max_leaf_nodes=95, min_samples_leaf=40,
        learning_rate=0.045, max_features=0.80, l2_regularization=1.5,
        early_stopping=False, random_state=303
    ).fit(X2_tr, y2_tr)

    # Medial-Specialized, Lateral-Specialized, and Rim/Subchondral-Specialized HGBs!
    w_med = np.where(~is_lat_tr, 2.5, 0.8).astype(np.float32)
    hgb2_med = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=71, min_samples_leaf=30,
        learning_rate=0.05, max_features=0.65, l2_regularization=1.2,
        early_stopping=False, random_state=304
    ).fit(X2_tr, y2_tr, sample_weight=w_med)

    w_lat = np.where(is_lat_tr, 2.5, 0.8).astype(np.float32)
    hgb2_lat = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=71, min_samples_leaf=30,
        learning_rate=0.05, max_features=0.65, l2_regularization=1.2,
        early_stopping=False, random_state=305
    ).fit(X2_tr, y2_tr, sample_weight=w_lat)

    w_rim = np.where(is_rim_tr, 2.2, 0.8).astype(np.float32)
    hgb2_rim = HistGradientBoostingClassifier(
        max_iter=260, max_leaf_nodes=79, min_samples_leaf=30,
        learning_rate=0.05, max_features=0.65, l2_regularization=1.2,
        early_stopping=False, random_state=306
    ).fit(X2_tr, y2_tr, sample_weight=w_rim)

    rng_et = np.random.RandomState(377)
    et_idx = rng_et.choice(len(y2_tr), min(len(y2_tr), 500000), replace=False)
    et2 = ExtraTreesClassifier(
        n_estimators=200, max_depth=26, min_samples_leaf=10,
        max_features='sqrt', n_jobs=16, random_state=307
    ).fit(X2_tr[et_idx], y2_tr[et_idx])
    del X2_tr, y2_tr; gc.collect()
    print(f"[Tibia V7] Fit Full-80 6xHGB + ExtraTrees P2 ensemble in {time.time()-t_fit2:.1f}s", flush=True)

    joblib.dump({
        'p1_folds_v7': tuple(p1_folds_v7),
        'p2_tibia_hgb': (hgb2_1, hgb2_2, hgb2_3, hgb2_med, hgb2_lat, hgb2_rim),
        'p2_tibia_et': et2,
        'p2_tibia_oof_folds': tuple(p2_tibia_oof_folds),
    }, os.path.join(model_dir, 'cart_tibia_v7_models.joblib'), compress=3)

    # 5. Step 4: Full-Resolution Evaluation on all 20 held-out eval cases (saving compact cropped ROIs!)
    t_ev = time.time()
    def pred_eval_tibia_v7(i):
        fname, img, lbl, sp, bm_v1, bm_v2, bm_raw = ev_cases[i]
        geo_t7 = prepare_geometry_tibia_v7(img, sp, bm_v2, bm_v1, bm_raw, atlas_priors)

        # Unpack cached v6 P1 for Tibia and Femur (0s!)
        ev_d6 = ev_v6[i]
        uz0, uz1, uy0, uy1, ux0, ux1 = ev_d6['bbox']
        pm1_t6 = np.zeros((*img.shape, 4), dtype=np.float32); pm1_t6[..., 0] = 1.0
        pm1_t6[uz0:uz1, uy0:uy1, ux0:ux1] = ev_d6['t_p1'].astype(np.float32)
        pm1_f6 = np.zeros((*img.shape, 4), dtype=np.float32); pm1_f6[..., 0] = 1.0
        pm1_f6[uz0:uz1, uy0:uy1, ux0:ux1] = ev_d6['f_p1'].astype(np.float32)

        # Full-resolution V7 P1 on Tibia shell
        X1_t7 = extract_features_tibia_v7(geo_t7, geo_t7['shell'], rng_jit=None)
        p1_t7 = 0.25 * sum(m.predict_proba(X1_t7) for m in p1_folds_v7).astype(np.float32)
        del X1_t7
        pm1_t7 = np.zeros((*img.shape, 4), dtype=np.float32); pm1_t7[..., 0] = 1.0
        pm1_t7[geo_t7['shell']] = p1_t7

        pm1_t = 0.45 * pm1_t6 + 0.55 * pm1_t7

        # Full-resolution V7 P2 on Tibia shell
        X2_t7 = extract_features_tibia_v7(geo_t7, geo_t7['shell'], pm_self=pm1_t, pm_other=pm1_f6, rng_jit=None)
        p_g = (hgb2_1.predict_proba(X2_t7) + hgb2_2.predict_proba(X2_t7) + hgb2_3.predict_proba(X2_t7)) / 3.0
        p_m = hgb2_med.predict_proba(X2_t7)
        p_l = hgb2_lat.predict_proba(X2_t7)
        p_r = hgb2_rim.predict_proba(X2_t7)
        p_et = et2.predict_proba(X2_t7).astype(np.float32)

        # Compartment-routed HGB prediction
        inds_sh = np.argwhere(geo_t7['shell'])
        med_low_z = geo_t7['ml_info'][0]
        z_med_split = geo_t7['ml_info'][2][4]
        w_hz = np.clip(0.5 + ((inds_sh[:, 0] * sp[0]) - z_med_split) / 4.0, 0.0, 1.0)[:, None].astype(np.float32)
        w_lat_sh = w_hz if med_low_z else (1.0 - w_hz)
        p_comp = (1.0 - w_lat_sh) * p_m + w_lat_sh * p_l

        p2_hgb_all = (0.55 * p_g + 0.25 * p_comp + 0.20 * p_r).astype(np.float32)
        del X2_t7

        sl = (slice(uz0, uz1), slice(uy0, uy1), slice(ux0, ux1))
        def _pack(sh, arr):
            vol = np.zeros((*img.shape, 4), dtype=np.float32); vol[..., 0] = 1.0
            vol[sh] = arr
            return vol[sl].astype(np.float16)

        return i, {
            'fname': fname,
            'bbox': (uz0, uz1, uy0, uy1, ux0, ux1),
            'shape': img.shape,
            't_p1_v7': _pack(geo_t7['shell'], p1_t7),
            't_p2_hgb': _pack(geo_t7['shell'], p2_hgb_all),
            't_p2_g': _pack(geo_t7['shell'], p_g.astype(np.float32)),
            't_p2_comp': _pack(geo_t7['shell'], p_comp.astype(np.float32)),
            't_p2_rim': _pack(geo_t7['shell'], p_r.astype(np.float32)),
            't_p2_et': _pack(geo_t7['shell'], p_et),
        }

    ev_v7 = {}
    for start in range(0, 20, 10):
        idxs = list(range(start, min(start + 10, 20)))
        batch = Parallel(n_jobs=len(idxs), prefer='threads')(delayed(pred_eval_tibia_v7)(i) for i in idxs)
        for i, d in batch:
            ev_v7[i] = d
    joblib.dump(ev_v7, os.path.join(cache_dir, 'ev_tibia_v7_rois.joblib'), compress=3)
    print(f"[Tibia V7] Predicted 20 eval cases and saved ROIs in {time.time()-t_ev:.1f}s (Total: {time.time()-t0:.1f}s)", flush=True)

    # 6. Evaluate V7 Standalone and V7 + V6 + V5 Multi-Bone Ensemble on all 20 held-out eval cases!
    prep = joblib.load(os.path.join(cache_dir, 'eval20_prep_uncompressed.joblib'), mmap_mode='r')
    def get_t_smoothed(ev_d, key_p1, key_hgb, key_et, w_et, sp, sl, shape):
        pm_t = 0.10 * ev_d[key_p1].astype(np.float32) + (0.90 * (1.0 - w_et)) * ev_d[key_hgb].astype(np.float32) + (0.90 * w_et) * ev_d[key_et].astype(np.float32)
        uz0, uz1, uy0, uy1, ux0, ux1 = ev_d['bbox']
        sig_t = [0.35 / sp[0], 0.14 / sp[1], 0.25 / sp[2]]
        pt_u  = gaussian_filter(pm_t[..., 2], sigma=sig_t)
        ptb_u = gaussian_filter(pm_t[..., 1], sigma=sig_t)
        out = {}
        for name, arr in [('pt', pt_u), ('ptb', ptb_u)]:
            vol = np.zeros(shape, dtype=np.float32)
            vol[uz0:uz1, uy0:uy1, ux0:ux1] = arr
            out[name] = vol[sl]
        return out

    ev_d7_list = []
    for i in range(20):
        c = prep[i]
        fname, img, lbl, sp, bm_v1, bm_v2, bm_raw = ev_cases[i]
        z0, z1, y0, y1, x0, x1 = c['sl_bbox']
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        ev_d7_list.append(get_t_smoothed(ev_v7[i], 't_p1_v7', 't_p2_hgb', 't_p2_et', 0.18, sp, sl, img.shape))

    print("\n================ 20-EVALUATION-CASE TIBIA V7 RESULTS ================", flush=True)
    w7, w6, w5, b_pen, th_t = 0.45, 0.25, 0.30, 0.20, 0.50
    fcs, tcs = [], []
    for i, c in enumerate(prep):
        d6, d5, d7 = c['d6'], c['d5'], ev_d7_list[i]
        lbl_c, df_c, dt_c = c['lbl_c'], c['df_c'], c['dt_c']
        ngf, ngt = c['ngf'], c['ngt']
        a_old_fc, a_new_fc, a_old_tc, a_new_tc = c['a_old_fc'], c['a_new_fc'], c['a_old_tc'], c['a_new_tc']

        pf  = 0.55 * d6['pf']  + 0.45 * d5['pf']
        pfb = 0.55 * d6['pfb'] + 0.45 * d5['pfb']
        pt  = w7 * d7['pt']  + w6 * d6['pt']  + w5 * d5['pt']
        ptb = w7 * d7['ptb'] + w6 * d6['ptb'] + w5 * d5['ptb']

        del_f = a_new_fc - a_old_fc
        del_t = a_new_tc - a_old_tc
        sf_part = pf - 0.24 * pfb + 0.20 * del_f
        st_part = pt - b_pen * ptb + 0.35 * del_t
        sf = sf_part + 0.28 * np.minimum(pf, pt)
        st = st_part + 0.28 * np.minimum(pf, pt)

        pf_50 = float(np.sum((pf - 0.24 * pfb) >= 0.50 + 0.05 * np.clip(df_c - 0.8, 0.0, 2.5)))
        pt_52 = float(np.sum((pt - b_pen * ptb) >= th_t + 0.05 * np.clip(dt_c - 0.8, 0.0, 2.5)))
        af_sum = float(np.sum(a_new_fc))
        at_sum = float(np.sum(a_new_tc))
        pf_sum = float(np.sum(pf[pf >= 0.05]))
        pt_sum = float(np.sum(pt[pt >= 0.05]))

        ind_f = 0.5 * ((pf_50 / pf_sum - 0.815) / 0.035) + 0.5 * ((pf_50 / af_sum - 0.98) / 0.10)
        ind_t = 0.5 * ((pt_52 / pt_sum - 0.765) / 0.040) + 0.5 * ((pt_52 / at_sum - 0.96) / 0.12)
        dth_f = np.clip(-0.04 * 0.60 * ind_f, -0.09, 0.09)
        dth_t = np.clip(-0.04 * 0.45 * ind_t, -0.09, 0.09)

        pred_fc = fast_keep_top_k((sf >= 0.50 + dth_f) & (sf_part >= st_part), k=1, min_size=300)
        pred_tc = fast_keep_top_k((st >= th_t + dth_t) & (st_part > sf_part), k=2, min_size=150)
        fcs.append(2.0 * np.sum((lbl_c == 2) & pred_fc) / (ngf + np.sum(pred_fc)))
        tcs.append(2.0 * np.sum((lbl_c == 4) & pred_tc) / (ngt + np.sum(pred_tc)))
    print(f"V7+V6+V5 Ensemble -> FC={np.mean(fcs):.5f}, TC={np.mean(tcs):.5f}", flush=True)
