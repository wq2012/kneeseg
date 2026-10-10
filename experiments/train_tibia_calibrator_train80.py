"""Train and save the 80-train-only Tibia Cartilage (th, bp) calibrator using matched OOF Stage-1/2/3 & subchondral MRI features."""
import os
import sys
import time
import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_EXP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)
from train_cartilage_v6_ultimate import fast_keep_top_k

SIG_T = [0.35 / 1.5, 0.14 / 0.39, 0.25 / 0.39]
PTB_0 = 0.12


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
    p3_c_s = gaussian_filter(p3_c, sigma=SIG_T)[cand]

    p3f_1d = 0.65 * f3_g + 0.35 * f3_c
    p3_f = np.zeros(lbl_sh, dtype=np.float32)
    p3_f[cand] = p3f_1d + 0.12
    p3_f_s = gaussian_filter(p3_f, sigma=SIG_T)[cand]

    p3_p_c = np.zeros(lbl_sh, dtype=np.float32)
    p3_p_c[cand] = p3_pure_1d - (float(np.mean(p3_pure_1d)) - float(np.mean(p3c_1d))) + 0.12
    p3_p_c_s = gaussian_filter(p3_p_c, sigma=SIG_T)[cand]

    p3_p_r = np.zeros(lbl_sh, dtype=np.float32)
    p3_p_r[cand] = p3_pure_1d + 0.12
    p3_p_r_s = gaussian_filter(p3_p_r, sigma=SIG_T)[cand]

    del_t = a_new - a_old
    base6_s = gaussian_filter(
        0.5 * pt6 + 0.5 * pt + 0.35 * del_t + 0.28 * np.minimum(pf, pt),
        sigma=SIG_T,
    )[cand]
    pt7_s = gaussian_filter(pt7, sigma=SIG_T)[cand]
    ptb_s = gaussian_filter(ptb, sigma=SIG_T)[cand]
    a_new_s = gaussian_filter(a_new, sigma=SIG_T)[cand]

    pt_ym1 = np.pad(pt, ((0, 0), (1, 0), (0, 0)), mode='edge')[:, :-1, :]
    pt_ym2 = np.pad(pt, ((0, 0), (2, 0), (0, 0)), mode='edge')[:, :-2, :]
    m_bot = (dt2 >= -0.45) & (dt2 <= 0.55) & (np.maximum(pt_ym1, pt_ym2) >= 0.50)
    pr_st = (st >= 0.51) & part
    core80 = (st >= 0.80) & part
    ic80 = float(np.median(img[core80])) if np.any(core80) else 100.0
    ib_deep = float(np.median(img[(dt2 <= -2.0) & (dt2 >= -4.5)]))
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
    if 'inorm_c' in cc:
        inorm_c = cc['inorm_c']
        zbn_c = cc['zbn_c']
    else:
        ic_c = float(np.median(img[core78])) if np.any(core78) else 100.0
        ib_c = float(np.median(img[dt2 <= -0.6])) if np.any(dt2 <= -0.6) else 50.0
        inorm_c = (img - ib_c) / max(ic_c - ib_c, 10.0)
        zb_med78 = float(np.median(zb[core78])) if np.any(core78) else 15.0
        zbn_c = zb / max(zb_med78, 3.0)
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

    return np.array(
        [
            float(np.mean((p2h_6 - p1_6)[sub])),  # 0
            float(np.mean((p2hb_6 - p1b_6)[sub])),  # 1
            float(np.mean((p2h_6 - p1_6)[bnd])),  # 2
            float(np.mean((p2h_6 - p1_6)[sup])),  # 3
            float(
                np.sum((p1_6 >= 0.50) & part)
                / max(np.sum((p2h_6 >= 0.50) & part), 1.0)
            ),  # 4
            float(
                np.sum((p2r_7 >= 0.50) & part)
                / max(np.sum((p2h_6 >= 0.50) & part), 1.0)
            ),  # 5
            float(np.mean((p2r_7 - p2h_6)[bnd])),  # 6
            float(np.mean((p2r_7 - p2h_6)[sub])),  # 7
            float(np.mean(s_bot[sub_s])),  # 8
            float(np.mean(ptb[sub_s])),  # 9
            float(np.mean(dt2[b_sh])),  # 10
            float(contrast),  # 11
            float(p['dth_t']),  # 12
            float(np.mean((dt2 - dt0)[sub_s])),  # 13
            float(np.mean(dt2[pr] <= 0.45)),  # 14
            float(np.mean(dt2[pr] <= 0.60)),  # 15
            float(np.mean((dt2 / np.maximum(2.0 * edt, 0.8))[pr])),  # 16
            float(np.mean((dt2 / np.maximum(2.0 * edt, 0.8))[l_th])),  # 17
            float(np.mean(zbn[part_sub & pr])),  # 18
            float(np.log10(max(float(p['p99']), 10.0))),  # 19
            float(np.mean(inorm_full[m_in1])),  # 20
            float(np.mean(inorm_full[m_out1])),  # 21
            dt2_at_grad,  # 22
            float(np.mean((a_new_full - a_old_full)[m_out1])),  # 23
            (float(np.sum(pr_sup)) - float(np.sum(p3c_raw[part_sup]))) / n_pr,  # 24
            (float(np.sum(pr_sup)) - float(np.sum(p3f_sm_c[part_sup]))) / n_pr,  # 25
            float(np.mean((s - pt7)[pr_sup]) - np.mean((s - pt7)[pr])),  # 26
            float(np.sum(pr)) / max(float(np.sum(an[part])), 1.0),  # 27
            float(np.mean((p2h_6 - p1_6)[sub]) - np.mean((p2h_6 - p1_6)[core])),  # 28
            float(
                np.mean((p2hb_6 - p1b_6)[sub]) - np.mean((p2hb_6 - p1b_6)[core])
            ),  # 29
            float(np.mean((p2h_6 - p1_6)[bnd]) - np.mean((p2h_6 - p1_6)[core])),  # 30
            float(np.mean((p3f - p3c)[b_sh])) if np.any(b_sh) else 0.0,  # 31
            0.0,  # 32
            float(np.mean((p3f - p3c)[sub_sm])) if np.any(sub_sm) else 0.0,  # 33
            0.0,  # 34
            0.0,  # 35
            float(np.mean(inorm_c[sub_lo]) - np.mean(inorm_c[sub_hi]))
            if (np.any(sub_lo) and np.any(sub_hi))
            else 0.0,  # 36
            float(np.mean(inorm0[sub_lo]) - np.mean(inorm0[sub_hi]))
            if (np.any(sub_lo) and np.any(sub_hi))
            else 0.0,  # 37
            float(np.mean(zbn_c[sub_lo]) - np.mean(zbn_c[sub_hi]))
            if (np.any(sub_lo) and np.any(sub_hi))
            else 0.0,  # 38
            float(np.mean(inorm_c[sub_sm])) if np.any(sub_sm) else 0.5,  # 39
            float(np.mean(zbn_c[sub_sm])) if np.any(sub_sm) else 0.5,  # 40
            float(np.mean(sc_base[b_sh])) if np.any(b_sh) else 0.5,  # 41
            float(np.mean(sc_base[sub_sm])) if np.any(sub_sm) else 0.5,  # 42
            float(np.mean(sc_base[sup])) if np.any(sup) else 0.5,  # 43
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
    sc = sc_base - float(bp_val) * (cc['ptb'] - PTB_0)
    pr_1d = cc['part'] & (sc >= float(th_val))
    pr_3d = np.zeros(cand.shape, dtype=bool)
    pr_3d[cand] = pr_1d
    return fast_keep_top_k(pr_3d, k=4, min_size=50)


if __name__ == '__main__':
    t0 = time.time()
    prep80 = joblib.load('experiments/cache/train80_tibia_prep.joblib', mmap_mode='r')
    prep20 = joblib.load('experiments/cache/eval20_tibia_prep_v2.joblib', mmap_mode='r')
    s3_tr = joblib.load('experiments/cache/train80_s3_clean_oof.joblib', mmap_mode='r')
    s3_ev = joblib.load(
        'experiments/cache/eval20_s3_clean_from_train80.joblib', mmap_mode='r'
    )
    f3 = joblib.load('experiments/cache/full_oof_s3_preds.joblib', mmap_mode='r')
    pure = joblib.load('experiments/cache/pure_oof_s3_preds.joblib', mmap_mode='r')
    ev_m = joblib.load('experiments/cache/eval20_matched_p1p2.joblib', mmap_mode='r')
    C_tr = [
        compute_case_smoothed_components(
            prep80[i],
            s3_tr['glob_declamp'][i],
            s3_tr['ctx_declamp'][i],
            s3_tr['cand'][i],
            f3['oof_g'][i],
            f3['oof_c'][i],
            pure['oof_pure_tr'][i],
        )
        for i in range(len(prep80))
    ]
    C_ev = [
        compute_case_smoothed_components(
            prep20[i],
            s3_ev['glob_declamp'][i],
            s3_ev['ctx_declamp'][i],
            s3_ev['cand'][i],
            f3['ev_g'][i],
            f3['ev_c'][i],
            pure['ev_pure_matched'][i],
        )
        for i in range(len(prep20))
    ]
    joblib.dump(
        {'C_tr': C_tr, 'C_ev': C_ev},
        'experiments/cache/smoothed_comps_v1.joblib',
        compress=3,
    )
    surfs = joblib.load('experiments/cache/dsc_surfaces_v2.joblib')
    D_tr_old = surfs['D_tr']

    bps = np.arange(-0.35, 0.651, 0.04)
    ths = np.arange(0.38, 0.661, 0.01)
    TH_grid, BP_grid = np.meshgrid(ths, bps)
    D_tr_orth = np.zeros_like(D_tr_old)
    for bi, bp in enumerate(bps):
        for ti, th in enumerate(ths):
            ti_eff = int(np.argmin(np.abs(ths - (th - PTB_0 * bp))))
            D_tr_orth[:, bi, ti] = D_tr_old[:, bi, ti_eff]

    th_t = np.zeros(80, dtype=np.float64)
    bp_t = np.zeros(80, dtype=np.float64)
    for i in range(80):
        s = D_tr_orth[i]
        m = np.max(s)
        w = np.exp((s - m) / 0.0015)
        w /= np.sum(w)
        th_t[i] = float(np.sum(w * TH_grid))
        bp_t[i] = float(np.sum(w * BP_grid))

    F_tr = extract_clean_calib_feats(
        prep80,
        s3_tr,
        f3['oof_g'],
        f3['oof_c'],
        pure['oof_pure_tr'],
        None,
        C_tr,
    )
    F_ev = extract_clean_calib_feats(
        prep20,
        s3_ev,
        f3['ev_g'],
        f3['ev_c'],
        pure['ev_pure_matched'],
        ev_m,
        C_ev,
    )

    th_cols = [2, 4, 7, 12, 14, 17, 18, 22, 38, 39]
    bp_cols = [1, 4, 7, 20, 21, 24, 25, 28, 29]

    sc_th = StandardScaler().fit(F_tr[:, th_cols])
    sc_bp = StandardScaler().fit(F_tr[:, bp_cols])
    m_th = Ridge(alpha=20.0).fit(
        sc_th.transform(F_tr[:, th_cols]), th_t - np.mean(th_t)
    )
    m_bp = Ridge(alpha=10.0).fit(
        sc_bp.transform(F_tr[:, bp_cols]), bp_t - np.mean(bp_t)
    )

    calib_model = {
        'th_cols': th_cols,
        'bp_cols': bp_cols,
        'sc_th': sc_th,
        'sc_bp': sc_bp,
        'm_th': m_th,
        'm_bp': m_bp,
        'th_base': 0.508,
        's_th': 1.65,
        's_bp': 1.60,
        'ptb_0': PTB_0,
    }
    c2_path = os.path.join(_EXP_DIR, 'models', 'cartilage_rf_p2.joblib')
    if os.path.exists(c2_path):
        c2_data = joblib.load(c2_path)
        c2_data['calib_model'] = calib_model
        joblib.dump(c2_data, c2_path, compress=3)

    ev_th = 0.508 + 1.65 * m_th.predict(sc_th.transform(F_ev[:, th_cols]))
    ev_bp = 1.60 * m_bp.predict(sc_bp.transform(F_ev[:, bp_cols]))

    ev_dscs = []
    for i, p in enumerate(prep20):
        lbl = np.array(p['lbl_c'], dtype=np.uint8)
        ngt = int(p['ngt'])
        pr_3d = predict_tibia_roi_from_components(
            C_ev[i], s3_ev['cand'][i], ev_th[i], ev_bp[i]
        )
        ev_dscs.append(2.0 * np.sum((lbl == 4) & pr_3d) / (ngt + np.sum(pr_3d)))

    print(
        f'Updated calib_model in cartilage_rf_p2.joblib | 3D Eval-20 TC DSC = {np.mean(ev_dscs):.5f} ± {np.std(ev_dscs):.5f} in {time.time()-t0:.1f}s'
    )
