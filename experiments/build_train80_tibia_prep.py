import gc
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
from threadpoolctl import threadpool_limits

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_EXP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)
from kneeseg.io import load_volume
from kneeseg.features import fast_signed_distance_transform
from kneeseg.rf_seg import _compute_articular_landmarks_and_stats
import train_cartilage_v6_ultimate as v6m
import train_tibia_v7_fast as v7m

# Exact fast partition replacement for fixed-column percentiles in v6m and v7m
_orig_pct = np.percentile
def _fast_pct(a, q, axis=None, keepdims=False):
    if axis == 1 and keepdims and a.ndim == 2:
        if q == 20 and a.shape[1] == 6:
            return np.partition(a, 1, axis=1)[:, 1:2]
        if q == 85 and a.shape[1] == 8:
            s = np.partition(a, (5, 6), axis=1)
            return 0.05 * s[:, 5:6] + 0.95 * s[:, 6:7]
    return _orig_pct(a, q, axis=axis, keepdims=keepdims)

v6m.np.percentile = _fast_pct
v7m.np.percentile = _fast_pct


def make_geo(img, sp, bm_ref, df, dt, df_alt, dt_alt, df_raw, dt_raw, bone_name, atlas_p, f_stats, f_comp, t_stats, t_comp, ml_info):
    if bone_name == 'tibia':
        shell = (dt >= -0.95) & (dt <= 4.6) & (df >= -1.0) & (df <= 25.0)
    else:
        shell = (df >= -0.95) & (df <= 4.7) & (dt >= -1.0) & (dt <= 9.0)
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
    img_lat_raw = gaussian_filter(img_f32, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
    sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
    w_bn = ((d_self_crop <= -1.4) & (d_self_crop >= -5.5)).astype(np.float32) + 1e-5
    w_bn_s = gaussian_filter(w_bn, sigma=sig_bn)
    i_bn_mean = gaussian_filter(img_lat_raw * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_var = gaussian_filter(((img_lat_raw - i_bn_mean) ** 2) * w_bn, sigma=sig_bn) / w_bn_s
    i_bn_std = np.maximum(np.sqrt(np.maximum(i_bn_var, 0.0)), max(0.05 * (i_high - i_low), 2.5))
    z_bone_crop = ((img_lat_raw - i_bn_mean) / i_bn_std).astype(np.float32)
    return {
        'bone_name': bone_name, 'shell': shell, 'bbox': (z0, z1, y0, y1, x0, x1), 'sp': sp,
        'img_f32': img_f32, 'inorm': inorm, 'z_cart': z_cart, 'z_bone_crop': z_bone_crop,
        'i_low': i_low, 'i_high': i_high, 'df_crop': df_crop, 'dt_crop': dt_crop,
        'dfs_crop': dfs_crop, 'dts_crop': dts_crop, 'd_self_crop': d_self_crop,
        'd_other_crop': d_other_crop, 'd_alt_self_crop': d_alt_self_crop,
        'd_raw_self_crop': d_raw_self_crop, 'ds_self_crop': ds_self_crop, 'ds_other_crop': ds_other_crop,
        'normals_self': (nz, ny, nx), 'normals_other': (nzo, nyo, nxo),
        'f_stats': f_stats, 'f_comp': f_comp, 't_stats': t_stats, 't_comp': t_comp,
        'ml_info': ml_info, 'atlas_p': atlas_p,
    }


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
            ih, im, iq, zm = i_high, float(np.median(img_f32[m_high])) if np.any(m_high) else i_high, 25.0, zb_med_glob
        i_low_c.append(il); i_high_c.append(ih); i_med_c.append(im); iqr_c.append(iq); zb_med_c.append(zm)
    geo['z_bone_sharp'] = z_bone_sharp
    geo['zb_med_glob'] = zb_med_glob
    geo['w_high_z'] = w_high_z
    geo['comp_masks'] = comp_masks
    geo['comp_stats'] = (i_low_c, i_high_c, i_med_c, iqr_c, zb_med_c)
    return geo


def predict_p2_hybrid(geo, pm1_self, pm1_other, clfs, is_v7=False):
    sh = geo['shell']
    p1c = pm1_self[..., 2]
    m_base = sh & ((p1c >= 0.14) | (np.roll(p1c, 1, axis=1) >= 0.24) | (np.roll(p1c, -1, axis=1) >= 0.24))
    m_even = np.zeros_like(m_base)
    m_even[:, :, ::2] = m_base[:, :, ::2]
    m_bnd = m_base & ((p1c >= 0.22) & (p1c <= 0.78))
    m_hyb = m_even | m_bnd

    if not is_v7:
        X2 = v6m.extract_features_v6(geo, m_hyb, pm_self=pm1_self, pm_other=pm1_other)
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
        X2 = v7m.extract_features_tibia_v7(geo, m_hyb, pm_self=pm1_self, pm_other=pm1_other)
        h1, h2, h3, hm, hl, hr = clfs
        p_g = (h1.predict_proba(X2) + h2.predict_proba(X2) + h3.predict_proba(X2)) / 3.0
        p_m = hm.predict_proba(X2)
        p_l = hl.predict_proba(X2)
        p_r = hr.predict_proba(X2)
        del X2
        inds_sh = np.argwhere(m_hyb)
        med_low_z = geo['ml_info'][0]
        z_med_split = geo['ml_info'][2][4]
        sp = geo['sp']
        w_hz = np.clip(0.5 + ((inds_sh[:, 0] * sp[0]) - z_med_split) / 4.0, 0.0, 1.0)[:, None].astype(np.float32)
        w_lat_sh = w_hz if med_low_z else (1.0 - w_hz)
        p_comp = (1.0 - w_lat_sh) * p_m + w_lat_sh * p_l
        p_all = (0.55 * p_g + 0.25 * p_comp + 0.20 * p_r).astype(np.float32)

        odd_fill = m_base & (~m_hyb)
        zs, ys, xs = np.where(odd_fill)
        xm1 = np.maximum(xs - 1, 0)
        xp1 = np.minimum(xs + 1, pm1_self.shape[2] - 1)
        outs = []
        for arr_sub in (p_all, p_comp.astype(np.float32), p_r.astype(np.float32)):
            vol = pm1_self.copy()
            vol[m_hyb] = arr_sub
            vol[zs, ys, xs] = 0.5 * (vol[zs, ys, xm1] + vol[zs, ys, xp1])
            outs.append(vol)
        return outs[0], outs[1], outs[2]


if __name__ == '__main__':
    t0 = time.time()
    with open(os.path.join(_ROOT_DIR, 'kneeseg', 'data', 'ski10_full_split.json')) as f:
        split = json.load(f)
    data_dir = os.environ.get('SKI10_DATA_DIR', '/path/to/SKI10/TrainingData')
    lbl_dir = os.environ.get('SKI10_LABEL_DIR', data_dir + '_labels')
    cache_dir = os.path.join(_EXP_DIR, 'cache')
    model_dir = os.path.join(_EXP_DIR, 'models')

    train_v2 = joblib.load(os.path.join(cache_dir, 'train80_ref_bones_v2.joblib'))
    train_v1 = joblib.load(os.path.join(cache_dir, 'train80_ref_bones.joblib'))
    train_raw = joblib.load(os.path.join(cache_dir, 'train80_pred_bones.joblib'))
    m_v6 = joblib.load(os.path.join(model_dir, 'cart_v6_refv2_models.joblib'))
    m_v5 = joblib.load(os.path.join(model_dir, 'cart_v5_refv1_models.joblib'))
    m_v7 = joblib.load(os.path.join(model_dir, 'cart_tibia_v7_models.joblib'))

    # 1. Build 80-case Leave-One-Out (LOO) Multi-Atlases on bm_v2 and bm_v1
    def _load_for_atlas(fname):
        img, sp = load_volume(os.path.join(data_dir, fname), return_spacing=True)
        lbl = load_volume(os.path.join(lbl_dir, fname.replace('image-', 'labels-')), return_spacing=False).astype(np.uint8)
        lbl[lbl == 5] = 1
        lbl[lbl == 6] = 3
        h2 = v6m.compute_case_atlas_hist_v6(img, train_v2[fname], sp, lbl)
        h1 = v6m.compute_case_atlas_hist_v6(img, train_v1[fname], sp, lbl)
        return h2, h1

    from joblib import Parallel, delayed
    hists = Parallel(n_jobs=16, prefer='threads')(delayed(_load_for_atlas)(f) for f in split['train'])
    h2_list = [x[0] for x in hists]
    h1_list = [x[1] for x in hists]
    n_h = len(h2_list[0])
    tot_h2 = [sum(h[k] for h in h2_list) for k in range(n_h)]
    tot_h1 = [sum(h[k] for h in h1_list) for k in range(n_h)]
    loo_ap6 = [v6m.make_atlas_priors_v6([tot_h2[k] - h[k] for k in range(n_h)]) for h in h2_list]
    loo_ap5 = [v6m.make_atlas_priors_v6([tot_h1[k] - h[k] for k in range(n_h)]) for h in h1_list]
    del hists, h2_list, h1_list, tot_h2, tot_h1
    gc.collect()
    print(f"Built 80-case LOO Multi-Atlases (v2 & v1) in {time.time()-t0:.1f}s", flush=True)

    def process_train_case(i):
        with threadpool_limits(limits=4):
            fname = split['train'][i]
            fold = i // 20
            img, sp = load_volume(os.path.join(data_dir, fname), return_spacing=True)
            lbl = load_volume(os.path.join(lbl_dir, fname.replace('image-', 'labels-')), return_spacing=False).astype(np.uint8)
            lbl[lbl == 5] = 1
            lbl[lbl == 6] = 3
            p99 = float(_orig_pct(img, 99))
            fov_y = float(img.shape[1] * sp[1])
            fov_z = float(img.shape[0] * sp[0])
            ngf = int(np.sum(lbl == 2))
            ngt = int(np.sum(lbl == 4))

            bm2 = train_v2[fname]
            bm1 = train_v1[fname]
            bm0 = train_raw[fname]
            u_mask = (bm2['femur'] > 0) | (bm2['tibia'] > 0) | (bm1['femur'] > 0) | (bm1['tibia'] > 0) | (bm0['tibia'] > 0)
            uz, uy, ux = np.where(u_mask)
            sh = img.shape
            bsl = (
                slice(max(0, int(uz.min()) - 10), min(sh[0], int(uz.max()) + 11)),
                slice(max(0, int(uy.min()) - 22), min(sh[1], int(uy.max()) + 23)),
                slice(max(0, int(ux.min()) - 22), min(sh[2], int(ux.max()) + 23)),
            )
            img_b = img[bsl]
            lbl_b = lbl[bsl]
            bm2_b = {k: v[bsl] for k, v in bm2.items()}
            bm1_b = {k: v[bsl] for k, v in bm1.items()}
            bm0_b = {k: v[bsl] for k, v in bm0.items()}

            df2 = fast_signed_distance_transform(bm2_b['femur'] > 0, sp)
            dt2 = fast_signed_distance_transform(bm2_b['tibia'] > 0, sp)
            df1 = fast_signed_distance_transform(bm1_b['femur'] > 0, sp)
            dt1 = fast_signed_distance_transform(bm1_b['tibia'] > 0, sp)
            df0 = fast_signed_distance_transform(bm0_b['femur'] > 0, sp)
            dt0 = fast_signed_distance_transform(bm0_b['tibia'] > 0, sp)

            f_st2, _, f_cp2 = _compute_articular_landmarks_and_stats(bm2_b['femur'], df2, dt2, 45.0, sp)
            t_st2, _, t_cp2 = _compute_articular_landmarks_and_stats(bm2_b['tibia'], dt2, df2, 25.0, sp)
            ml2 = v6m.compute_ml_frames(bm2_b, df2, dt2, sp)

            f_st1, _, f_cp1 = _compute_articular_landmarks_and_stats(bm1_b['femur'], df1, dt1, 45.0, sp)
            t_st1, _, t_cp1 = _compute_articular_landmarks_and_stats(bm1_b['tibia'], dt1, df1, 25.0, sp)
            ml1 = v6m.compute_ml_frames(bm1_b, df1, dt1, sp)

            ap6_i = loo_ap6[i]
            ap5_i = loo_ap5[i]

            geo_t6 = make_geo(img_b, sp, bm2_b, df2, dt2, df1, dt1, df0, dt0, 'tibia', ap6_i, f_st2, f_cp2, t_st2, t_cp2, ml2)
            geo_f6 = make_geo(img_b, sp, bm2_b, df2, dt2, df1, dt1, df0, dt0, 'femur', ap6_i, f_st2, f_cp2, t_st2, t_cp2, ml2)
            geo_t5 = make_geo(img_b, sp, bm1_b, df1, dt1, df2, dt2, df0, dt0, 'tibia', ap5_i, f_st1, f_cp1, t_st1, t_cp1, ml1)
            geo_f5 = make_geo(img_b, sp, bm1_b, df1, dt1, df2, dt2, df0, dt0, 'femur', ap5_i, f_st1, f_cp1, t_st1, t_cp1, ml1)
            geo_t7 = add_v7_fields(dict(geo_t6))

            pm1_t6 = v6m.predict_p1_fast_strided_v6(geo_t6, m_v6['p1_folds']['tibia'][fold])
            pm1_f6 = v6m.predict_p1_fast_strided_v6(geo_f6, m_v6['p1_folds']['femur'][fold])
            pm1_t5 = v6m.predict_p1_fast_strided_v6(geo_t5, m_v5['p1_folds']['tibia'][fold])
            pm1_f5 = v6m.predict_p1_fast_strided_v6(geo_f5, m_v5['p1_folds']['femur'][fold])
            pm1_t7 = v7m.predict_p1_fast_strided_tibia_v7(geo_t7, m_v7['p1_folds_v7'][fold])
            pm1_t67 = 0.45 * pm1_t6 + 0.55 * pm1_t7

            pm2_t6 = predict_p2_hybrid(geo_t6, pm1_t6, pm1_f6, m_v6['p2_tibia_oof_folds'][fold], is_v7=False)
            pm2_f6 = predict_p2_hybrid(geo_f6, pm1_f6, pm1_t6, m_v6['p2_femur_oof_folds'][fold], is_v7=False)
            pm2_t5 = predict_p2_hybrid(geo_t5, pm1_t5, pm1_f5, m_v5['p2_tibia_oof_folds'][fold], is_v7=False)
            pm2_f5 = predict_p2_hybrid(geo_f5, pm1_f5, pm1_t5, m_v5['p2_femur_oof_folds'][fold], is_v7=False)
            pm2_t7, pm2c_t7, pm2r_t7 = predict_p2_hybrid(geo_t7, pm1_t67, pm1_f6, m_v7['p2_tibia_oof_folds'][fold], is_v7=True)

            # Use tibia ROI crop sl = geo_t6['bbox'] (expanded slightly to include any lbl_b == 4)
            z0, z1, y0, y1, x0, x1 = geo_t6['bbox']
            if np.any(lbl_b == 4):
                tz, ty, tx = np.where(lbl_b == 4)
                z0 = min(z0, max(0, int(tz.min()) - 3))
                z1 = max(z1, min(img_b.shape[0], int(tz.max()) + 4))
                y0 = min(y0, max(0, int(ty.min()) - 3))
                y1 = max(y1, min(img_b.shape[1], int(ty.max()) + 4))
                x0 = min(x0, max(0, int(tx.min()) - 3))
                x1 = max(x1, min(img_b.shape[2], int(tx.max()) + 4))
            sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

            lbl_c = lbl_b[sl]
            img_c = img_b[sl].astype(np.float32)
            df2_c = df2[sl]; dt2_c = dt2[sl]; dt1_c = dt1[sl]; dt0_c = dt0[sl]

            dfs_c = gaussian_filter(df2_c, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
            dts_c = gaussian_filter(dt2_c, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
            _, nyt_c, _ = np.gradient(dts_c, *sp)
            zz, yy, xx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
            coords_c = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(np.float32) * np.array(sp, dtype=np.float32)
            bins_c, _ = v6m.compute_atlas_bins_v6(coords_c, dfs_c.ravel(), dts_c.ravel(), f_st2, t_cp2, ml2)
            sh_c = df2_c.shape
            a_old_tc = ap6_i['pt_old_tc'][bins_c['t_old']].reshape(sh_c)
            a_new_tc = (
                0.32 * ap6_i['pt_ml_tc'][bins_c['t_ml']] +
                0.30 * ap6_i['pt_sep_tc'][bins_c['t_sep']] +
                0.20 * ap6_i['pt_mly_tc'][bins_c['t_mly']] +
                0.18 * ap6_i['pt_wide_tc'][bins_c['t_wide']]
            ).reshape(sh_c)
            edt_ens = (
                0.40 * ap6_i['edt_ml_2d'][bins_c['t_ml'][0], bins_c['t_ml'][1]] +
                0.35 * ap6_i['edt_sep_2d'][bins_c['t_sep'][0], bins_c['t_sep'][1], bins_c['t_sep'][2]] +
                0.25 * ap6_i['edt_wide_2d'][bins_c['t_wide'][0], bins_c['t_wide'][1]]
            ).reshape(sh_c)

            img_f0 = gaussian_filter(img_c, sigma=[0.35 / s for s in sp])
            img_lat = gaussian_filter(img_c, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
            sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
            w_bn = ((dt2_c <= -1.4) & (dt2_c >= -5.5)).astype(np.float32) + 1e-5
            w_bn_s = gaussian_filter(w_bn, sigma=sig_bn)
            i_bn_mean = gaussian_filter(img_lat * w_bn, sigma=sig_bn) / w_bn_s
            i_bn_var = gaussian_filter(((img_lat - i_bn_mean) ** 2) * w_bn, sigma=sig_bn) / w_bn_s
            i_bn_std = np.maximum(np.sqrt(np.maximum(i_bn_var, 0.0)), 3.0)
            zb = ((img_lat - i_bn_mean) / i_bn_std).astype(np.float32)

            pm_t6 = 0.12 * pm1_t6[sl] + 0.88 * pm2_t6[sl]
            pm_f6 = 0.12 * pm1_f6[sl] + 0.88 * pm2_f6[sl]
            pm_t5 = 0.12 * pm1_t5[sl] + 0.88 * pm2_t5[sl]
            pm_f5 = 0.12 * pm1_f5[sl] + 0.88 * pm2_f5[sl]

            sig_f = [0.25 / sp[0], 0.10 / sp[1], 0.18 / sp[2]]
            sig_t = [0.35 / sp[0], 0.14 / sp[1], 0.25 / sp[2]]
            pm_t = 0.70 * pm_t6 + 0.30 * pm_t5
            pm_f = 0.70 * pm_f6 + 0.30 * pm_f5

            pt_bg = gaussian_filter(pm_t[..., 0], sigma=sig_t)
            ptb   = gaussian_filter(pm_t[..., 1], sigma=sig_t)
            pt    = gaussian_filter(pm_t[..., 2], sigma=sig_t)
            pt_ot = gaussian_filter(pm_t[..., 3], sigma=sig_t)

            pf_bg = gaussian_filter(pm_f[..., 0], sigma=sig_f)
            pfb   = gaussian_filter(pm_f[..., 1], sigma=sig_f)
            pf    = gaussian_filter(pm_f[..., 2], sigma=sig_f)
            pf_ot = gaussian_filter(pm_f[..., 3], sigma=sig_f)

            pt6  = gaussian_filter(pm_t6[..., 2], sigma=sig_t)
            ptb6 = gaussian_filter(pm_t6[..., 1], sigma=sig_t)
            pt5  = gaussian_filter(pm_t5[..., 2], sigma=sig_t)
            ptb5 = gaussian_filter(pm_t5[..., 1], sigma=sig_t)

            pm7 = 0.12 * pm1_t7[sl] + 0.88 * pm2_t7[sl]
            pt7  = gaussian_filter(pm7[..., 2], sigma=sig_t)
            ptb7 = gaussian_filter(pm7[..., 1], sigma=sig_t)

            b_pen = 0.20
            del_t = a_new_tc - a_old_tc
            sf_part = pf - b_pen * pfb
            st_part = pt - b_pen * ptb + 0.25 * del_t
            sf = sf_part + 0.28 * np.minimum(pf, pt)
            st = st_part + 0.28 * np.minimum(pf, pt)

            pt_52 = float(np.sum((pt - b_pen * ptb) >= 0.50 + 0.05 * np.clip(dt2_c - 0.8, 0.0, 2.5)))
            at_sum = float(np.sum(a_new_tc))
            pt_sum = float(np.sum(pt[pt >= 0.05]))
            ind_t = 0.5 * ((pt_52 / pt_sum - 0.765) / 0.040) + 0.5 * ((pt_52 / at_sum - 0.96) / 0.12)
            dth_t = float(np.clip(-0.04 * 0.45 * ind_t, -0.09, 0.09))

            pred_tc = v6m.fast_keep_top_k((st >= 0.50 + dth_t) & (st_part > sf_part), k=2, min_size=150)
            dsc0 = float(2.0 * np.sum(pred_tc & (lbl_c == 4)) / (ngt + np.sum(pred_tc)))

            return {
                'fname': fname, 'sp': sp, 'p99': p99, 'fov_y': fov_y, 'fov_z': fov_z,
                'lbl_c': lbl_c.astype(np.uint8),
                'img_f0': img_f0.astype(np.float16),
                'img_lat': img_lat.astype(np.float16),
                'zb': zb.astype(np.float16),
                'df2': df2_c.astype(np.float16), 'dt2': dt2_c.astype(np.float16),
                'dt1': dt1_c.astype(np.float16), 'dt0': dt0_c.astype(np.float16),
                'nyt': nyt_c.astype(np.float16),
                'ngf': ngf, 'ngt': ngt,
                'a_old_tc': a_old_tc.astype(np.float16), 'a_new_tc': a_new_tc.astype(np.float16),
                'edt_ens': edt_ens.astype(np.float16),
                'pt_bg': pt_bg.astype(np.float16), 'ptb': ptb.astype(np.float16),
                'pt': pt.astype(np.float16), 'pt_ot': pt_ot.astype(np.float16),
                'pf_bg': pf_bg.astype(np.float16), 'pfb': pfb.astype(np.float16),
                'pf': pf.astype(np.float16), 'pf_ot': pf_ot.astype(np.float16),
                'pt6': pt6.astype(np.float16), 'ptb6': ptb6.astype(np.float16),
                'pt5': pt5.astype(np.float16), 'ptb5': ptb5.astype(np.float16),
                'pt7': pt7.astype(np.float16), 'ptb7': ptb7.astype(np.float16),
                'p1_tc6': pm1_t6[sl][..., 2].astype(np.float16),
                'p1_tb6': pm1_t6[sl][..., 1].astype(np.float16),
                'p2h_tc6': pm2_t6[sl][..., 2].astype(np.float16),
                'p2h_tb6': pm2_t6[sl][..., 1].astype(np.float16),
                'p1_tc7': pm1_t7[sl][..., 2].astype(np.float16),
                'p2h_tc7': pm2_t7[sl][..., 2].astype(np.float16),
                'p2h_tb7': pm2_t7[sl][..., 1].astype(np.float16),
                'p2c_tc7': pm2c_t7[sl][..., 2].astype(np.float16),
                'p2r_tc7': pm2r_t7[sl][..., 2].astype(np.float16),
                'st': st.astype(np.float16), 'sf': sf.astype(np.float16),
                'st_part': st_part.astype(np.float16), 'sf_part': sf_part.astype(np.float16),
                'dth_t': dth_t, 'pred_tc': pred_tc, 'dsc0': dsc0,
            }

    all_prep = []
    n_workers = min(16, os.cpu_count() or 4)
    if 'fork' in mp.get_all_start_methods():
        with mp.get_context('fork').Pool(n_workers) as pool:
            for b_idx, start in enumerate(range(0, 80, 16)):
                tb = time.time()
                batch_idxs = list(range(start, start + 16))
                batch_res = pool.map(process_train_case, batch_idxs)
                all_prep.extend(batch_res)
                mean_b = np.mean([r['dsc0'] for r in batch_res])
                mean_cum = np.mean([r['dsc0'] for r in all_prep])
                print(f"Batch {b_idx+1}/5 ({len(all_prep)}/80) done in {time.time()-tb:.1f}s | Batch TC DSC={mean_b:.5f} | Cum TC DSC={mean_cum:.5f}", flush=True)
    else:
        for b_idx, start in enumerate(range(0, 80, 16)):
            tb = time.time()
            batch_idxs = list(range(start, start + 16))
            batch_res = Parallel(n_jobs=n_workers, prefer='threads')(delayed(process_train_case)(i) for i in batch_idxs)
            all_prep.extend(batch_res)
            mean_b = np.mean([r['dsc0'] for r in batch_res])
            mean_cum = np.mean([r['dsc0'] for r in all_prep])
            print(f"Batch {b_idx+1}/5 ({len(all_prep)}/80) done in {time.time()-tb:.1f}s | Batch TC DSC={mean_b:.5f} | Cum TC DSC={mean_cum:.5f}", flush=True)

    out_path = os.path.join(cache_dir, 'train80_tibia_prep.joblib')
    joblib.dump(all_prep, out_path, compress=0)
    print(f"Saved {out_path} ({len(all_prep)} cases, Mean OOF TC DSC={np.mean([r['dsc0'] for r in all_prep]):.5f}) in {time.time()-t0:.1f}s!", flush=True)
    del all_prep; gc.collect()

    # 2. Also build eval20_tibia_prep_v2.joblib on the 20 eval cases using ev_v6, ev_v5, ev_v7 ROIs
    t_ev = time.time()
    eval_v2 = joblib.load(os.path.join(cache_dir, 'eval_ref_bones_v2.joblib'))
    eval_v1 = joblib.load(os.path.join(cache_dir, 'eval_ref_bones_80.joblib'))
    eval_raw = joblib.load(os.path.join(cache_dir, 'eval_pred_bones.joblib'))
    ev_v6 = joblib.load(os.path.join(cache_dir, 'ev_v6_refv2_rois.joblib'))
    ev_v5 = joblib.load(os.path.join(cache_dir, 'ev_v5_refv1_rois.joblib'))
    ev_v7 = joblib.load(os.path.join(cache_dir, 'ev_tibia_v7_rois.joblib'))
    ap6_full = m_v6['atlas_priors']

    def process_eval_case(i):
        fname = split['eval'][i]
        img, sp = load_volume(os.path.join(data_dir, fname), return_spacing=True)
        lbl = load_volume(os.path.join(lbl_dir, fname.replace('image-', 'labels-')), return_spacing=False).astype(np.uint8)
        lbl[lbl == 5] = 1
        lbl[lbl == 6] = 3
        p99 = float(_orig_pct(img, 99))
        fov_y = float(img.shape[1] * sp[1])
        fov_z = float(img.shape[0] * sp[0])
        ngf = int(np.sum(lbl == 2))
        ngt = int(np.sum(lbl == 4))

        bm2 = eval_v2[fname]
        bm1 = eval_v1[fname]
        bm0 = eval_raw[fname]
        df2 = fast_signed_distance_transform(bm2['femur'] > 0, sp)
        dt2 = fast_signed_distance_transform(bm2['tibia'] > 0, sp)
        dt1 = fast_signed_distance_transform(bm1['tibia'] > 0, sp)
        dt0 = fast_signed_distance_transform(bm0['tibia'] > 0, sp)

        f_st2, _, f_cp2 = _compute_articular_landmarks_and_stats(bm2['femur'], df2, dt2, 45.0, sp)
        t_st2, _, t_cp2 = _compute_articular_landmarks_and_stats(bm2['tibia'], dt2, df2, 25.0, sp)
        ml2 = v6m.compute_ml_frames(bm2, df2, dt2, sp)

        d6 = ev_v6[i]
        d5 = ev_v5[i]
        d7 = ev_v7[i]
        uz0, uz1, uy0, uy1, ux0, ux1 = [int(x) for x in d6['bbox']]
        u_bbox = (uz0, uz1, uy0, uy1, ux0, ux1)

        shell_t = (dt2 >= -0.95) & (dt2 <= 4.6) & (df2 >= -1.0) & (df2 <= 25.0)
        inds_t = np.argwhere(shell_t)
        margin = [int(np.ceil(6.0 / s)) for s in sp]
        z0 = max(uz0, int(inds_t[:, 0].min()) - margin[0])
        z1 = min(uz1, int(inds_t[:, 0].max()) + 5)
        y0 = max(uy0, int(inds_t[:, 1].min()) - margin[1])
        y1 = min(uy1, int(inds_t[:, 1].max()) + 5)
        x0 = max(ux0, int(inds_t[:, 2].min()) - margin[2])
        x1 = min(ux1, int(inds_t[:, 2].max()) + 5)
        sub_bbox = (z0 - uz0, z1 - uz0, y0 - uy0, y1 - uy0, x0 - ux0, x1 - ux0)
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        sub_sl = (slice(z0 - uz0, z1 - uz0), slice(y0 - uy0, y1 - uy0), slice(x0 - ux0, x1 - ux0))

        lbl_c = lbl[sl]
        img_c = img[sl].astype(np.float32)
        df2_c = df2[sl]; dt2_c = dt2[sl]; dt1_c = dt1[sl]; dt0_c = dt0[sl]

        dfs_c = gaussian_filter(df2_c, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
        dts_c = gaussian_filter(dt2_c, sigma=[2.0 / sp[0], 0.7 / sp[1], 2.0 / sp[2]])
        _, nyt_c, _ = np.gradient(dts_c, *sp)
        zz, yy, xx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
        coords_c = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(np.float32) * np.array(sp, dtype=np.float32)
        bins_c, _ = v6m.compute_atlas_bins_v6(coords_c, dfs_c.ravel(), dts_c.ravel(), f_st2, t_cp2, ml2)
        sh_c = df2_c.shape
        a_old_tc = ap6_full['pt_old_tc'][bins_c['t_old']].reshape(sh_c)
        a_new_tc = (
            0.32 * ap6_full['pt_ml_tc'][bins_c['t_ml']] +
            0.30 * ap6_full['pt_sep_tc'][bins_c['t_sep']] +
            0.20 * ap6_full['pt_mly_tc'][bins_c['t_mly']] +
            0.18 * ap6_full['pt_wide_tc'][bins_c['t_wide']]
        ).reshape(sh_c)
        edt_ens = (
            0.40 * ap6_full['edt_ml_2d'][bins_c['t_ml'][0], bins_c['t_ml'][1]] +
            0.35 * ap6_full['edt_sep_2d'][bins_c['t_sep'][0], bins_c['t_sep'][1], bins_c['t_sep'][2]] +
            0.25 * ap6_full['edt_wide_2d'][bins_c['t_wide'][0], bins_c['t_wide'][1]]
        ).reshape(sh_c)

        img_f0 = gaussian_filter(img_c, sigma=[0.35 / s for s in sp])
        img_lat = gaussian_filter(img_c, sigma=[1.2 / sp[0], 0.30 / sp[1], 1.2 / sp[2]])
        sig_bn = [4.5 / sp[0], 2.2 / sp[1], 4.5 / sp[2]]
        w_bn = ((dt2_c <= -1.4) & (dt2_c >= -5.5)).astype(np.float32) + 1e-5
        w_bn_s = gaussian_filter(w_bn, sigma=sig_bn)
        i_bn_mean = gaussian_filter(img_lat * w_bn, sigma=sig_bn) / w_bn_s
        i_bn_var = gaussian_filter(((img_lat - i_bn_mean) ** 2) * w_bn, sigma=sig_bn) / w_bn_s
        i_bn_std = np.maximum(np.sqrt(np.maximum(i_bn_var, 0.0)), 3.0)
        zb = ((img_lat - i_bn_mean) / i_bn_std).astype(np.float32)

        def _unpack_ev(ev_d, key):
            bz0, bz1, by0, by1, bx0, bx1 = ev_d['bbox']
            vol = np.zeros((*img.shape, 4), dtype=np.float32)
            vol[..., 0] = 1.0
            vol[bz0:bz1, by0:by1, bx0:bx1] = ev_d[key].astype(np.float32)
            return vol[sl]

        w_et = 0.18
        pm_t6 = 0.10 * _unpack_ev(d6, 't_p1') + (0.90 * (1.0 - w_et)) * _unpack_ev(d6, 't_p2_hgb') + (0.90 * w_et) * _unpack_ev(d6, 't_p2_et')
        pm_f6 = 0.10 * _unpack_ev(d6, 'f_p1') + (0.90 * (1.0 - w_et)) * _unpack_ev(d6, 'f_p2_hgb') + (0.90 * w_et) * _unpack_ev(d6, 'f_p2_et')
        pm_t5 = 0.10 * _unpack_ev(d5, 't_p1') + (0.90 * (1.0 - w_et)) * _unpack_ev(d5, 't_p2_hgb') + (0.90 * w_et) * _unpack_ev(d5, 't_p2_et')
        pm_f5 = 0.10 * _unpack_ev(d5, 'f_p1') + (0.90 * (1.0 - w_et)) * _unpack_ev(d5, 'f_p2_hgb') + (0.90 * w_et) * _unpack_ev(d5, 'f_p2_et')
        pm7 = 0.10 * _unpack_ev(d7, 't_p1_v7') + (0.90 * (1.0 - w_et)) * _unpack_ev(d7, 't_p2_hgb') + (0.90 * w_et) * _unpack_ev(d7, 't_p2_et')

        sig_f = [0.25 / sp[0], 0.10 / sp[1], 0.18 / sp[2]]
        sig_t = [0.35 / sp[0], 0.14 / sp[1], 0.25 / sp[2]]
        pm_t = 0.70 * pm_t6 + 0.30 * pm_t5
        pm_f = 0.70 * pm_f6 + 0.30 * pm_f5

        pt_bg = gaussian_filter(pm_t[..., 0], sigma=sig_t)
        ptb   = gaussian_filter(pm_t[..., 1], sigma=sig_t)
        pt    = gaussian_filter(pm_t[..., 2], sigma=sig_t)
        pt_ot = gaussian_filter(pm_t[..., 3], sigma=sig_t)

        pf_bg = gaussian_filter(pm_f[..., 0], sigma=sig_f)
        pfb   = gaussian_filter(pm_f[..., 1], sigma=sig_f)
        pf    = gaussian_filter(pm_f[..., 2], sigma=sig_f)
        pf_ot = gaussian_filter(pm_f[..., 3], sigma=sig_f)

        pt6  = gaussian_filter(pm_t6[..., 2], sigma=sig_t)
        ptb6 = gaussian_filter(pm_t6[..., 1], sigma=sig_t)
        pt5  = gaussian_filter(pm_t5[..., 2], sigma=sig_t)
        ptb5 = gaussian_filter(pm_t5[..., 1], sigma=sig_t)
        pt7  = gaussian_filter(pm7[..., 2], sigma=sig_t)
        ptb7 = gaussian_filter(pm7[..., 1], sigma=sig_t)

        b_pen = 0.20
        del_t = a_new_tc - a_old_tc
        sf_part = pf - b_pen * pfb
        st_part = pt - b_pen * ptb + 0.25 * del_t
        sf = sf_part + 0.28 * np.minimum(pf, pt)
        st = st_part + 0.28 * np.minimum(pf, pt)

        pt_52 = float(np.sum((pt - b_pen * ptb) >= 0.50 + 0.05 * np.clip(dt2_c - 0.8, 0.0, 2.5)))
        at_sum = float(np.sum(a_new_tc))
        pt_sum = float(np.sum(pt[pt >= 0.05]))
        ind_t = 0.5 * ((pt_52 / pt_sum - 0.765) / 0.040) + 0.5 * ((pt_52 / at_sum - 0.96) / 0.12)
        dth_t = float(np.clip(-0.04 * 0.45 * ind_t, -0.09, 0.09))

        pred_tc = v6m.fast_keep_top_k((st >= 0.50 + dth_t) & (st_part > sf_part), k=2, min_size=150)
        dsc0 = float(2.0 * np.sum(pred_tc & (lbl_c == 4)) / (ngt + np.sum(pred_tc)))

        return {
            'fname': fname, 'sp': sp, 'p99': p99, 'fov_y': fov_y, 'fov_z': fov_z,
            'orig_shape': img.shape, 'u_bbox': u_bbox, 'sub_bbox': sub_bbox,
            'lbl_c': lbl_c.astype(np.uint8),
            'img_f0': img_f0.astype(np.float16),
            'img_lat': img_lat.astype(np.float16),
            'zb': zb.astype(np.float16),
            'df2': df2_c.astype(np.float16), 'dt2': dt2_c.astype(np.float16),
            'dt1': dt1_c.astype(np.float16), 'dt0': dt0_c.astype(np.float16),
            'nyt': nyt_c.astype(np.float16),
            'ngf': ngf, 'ngt': ngt,
            'a_old_tc': a_old_tc.astype(np.float16), 'a_new_tc': a_new_tc.astype(np.float16),
            'edt_ens': edt_ens.astype(np.float16),
            'pt_bg': pt_bg.astype(np.float16), 'ptb': ptb.astype(np.float16),
            'pt': pt.astype(np.float16), 'pt_ot': pt_ot.astype(np.float16),
            'pf_bg': pf_bg.astype(np.float16), 'pfb': pfb.astype(np.float16),
            'pf': pf.astype(np.float16), 'pf_ot': pf_ot.astype(np.float16),
            'pt6': pt6.astype(np.float16), 'ptb6': ptb6.astype(np.float16),
            'pt5': pt5.astype(np.float16), 'ptb5': ptb5.astype(np.float16),
            'pt7': pt7.astype(np.float16), 'ptb7': ptb7.astype(np.float16),
            'p1_tc6': d6['t_p1'][sub_sl][..., 2].astype(np.float16),
            'p1_tb6': d6['t_p1'][sub_sl][..., 1].astype(np.float16),
            'p2h_tc6': d6['t_p2_hgb'][sub_sl][..., 2].astype(np.float16),
            'p2h_tb6': d6['t_p2_hgb'][sub_sl][..., 1].astype(np.float16),
            'p1_tc7': d7['t_p1_v7'][sub_sl][..., 2].astype(np.float16),
            'p2h_tc7': d7['t_p2_hgb'][sub_sl][..., 2].astype(np.float16),
            'p2h_tb7': d7['t_p2_hgb'][sub_sl][..., 1].astype(np.float16),
            'p2c_tc7': d7['t_p2_comp'][sub_sl][..., 2].astype(np.float16),
            'p2r_tc7': d7['t_p2_rim'][sub_sl][..., 2].astype(np.float16),
            'st': st.astype(np.float16), 'sf': sf.astype(np.float16),
            'st_part': st_part.astype(np.float16), 'sf_part': sf_part.astype(np.float16),
            'dth_t': dth_t, 'pred_tc': pred_tc, 'dsc0': dsc0,
        }

    eval_prep = Parallel(n_jobs=16, prefer='threads')(delayed(process_eval_case)(i) for i in range(20))
    ev_out_path = os.path.join(cache_dir, 'eval20_tibia_prep_v2.joblib')
    joblib.dump(eval_prep, ev_out_path, compress=0)
    print(f"Saved {ev_out_path} ({len(eval_prep)} cases, Mean Eval TC DSC0={np.mean([r['dsc0'] for r in eval_prep]):.5f}) in {time.time()-t_ev:.1f}s!", flush=True)

