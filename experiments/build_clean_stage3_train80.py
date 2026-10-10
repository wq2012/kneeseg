"""Train clean 80-train-only Stage-3 Tibia models (ZERO P1, ZERO P2-P1, ZERO V7, ZERO dt1, ZERO a_old contamination) and save OOF + Eval-20 caches."""
import os
import time
import gc
import joblib
from joblib import Parallel, delayed
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from sklearn.ensemble import HistGradientBoostingClassifier

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
        pt_ot = np.array(p['pt_ot'], dtype=np.float32)
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
        y_std_col = np.sqrt(np.maximum(np.sum(w_c * ((yy - y_mean_col) ** 2), axis=1, keepdims=True) / w_c_sum, 0.25))

        thick_soft_mm = np.sum(pt, axis=1, keepdims=True) * sp[1]
        thick_hard_mm = np.sum(pred0, axis=1, keepdims=True).astype(np.float32) * sp[1]
        pt_col_max = np.max(pt, axis=1, keepdims=True)

        pt_row_max = np.max(pt, axis=2, keepdims=True)
        pt_row_max_s = gaussian_filter(pt_row_max[:, :, 0], sigma=[1.2 / sp[0], 1.2 / sp[1]])[:, :, None]
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
        y_top_smooth = (gaussian_filter(np.where(pr_col[:, 0, :], y_top_raw[:, 0, :], 0.0) * w_col2d, sigma=sig_2d)[:, None, :]) / w_col_s
        y_bot_smooth = (gaussian_filter(np.where(pr_col[:, 0, :], y_bot_raw[:, 0, :], 0.0) * w_col2d, sigma=sig_2d)[:, None, :]) / w_col_s

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

        # Case-level unsupervised physical context vector (ZERO dt1, ZERO a_old!)
        dv20_all = float(np.mean((dt2 - dt0)[m_bot_vol])) if np.any(m_bot_vol) else 0.0
        rcb_bot_all = float(np.mean(s_bot[m_bot_vol])) if np.any(m_bot_vol) else 0.5
        m_top_pr = m_top_vol & (st >= 0.42)
        st_top_mean = float(np.mean(st[m_top_pr])) if np.any(m_top_pr) else 0.55
        m_wb = (dt2 >= 0.0) & (dt2 <= 2.5) & (a_new >= 0.35)
        js_med = float(np.median((df2 + dt2)[m_wb])) if np.any(m_wb) else 4.0
        m_sup = (dt2 >= 1.1) & (dt2 <= 2.2) & (pt >= 0.50)
        m_mid = (dt2 >= 0.1) & (dt2 <= 0.9) & (pt >= 0.50)
        isup_imid = float(np.mean(inorm_f0[m_sup]) - np.mean(inorm_f0[m_mid])) if (np.any(m_sup) and np.any(m_mid)) else 0.0
        m_cor = (dt2 >= 0.4) & (dt2 <= 1.4) & (a_new >= 0.45)
        st_cor = float(np.mean(st[m_cor])) if np.any(m_cor) else 0.85
        log_p99 = float(np.log10(max(p99, 10.0)))
        low_p99 = float(p99 < 250.0)

        case_ctx = np.array([
            log_p99, low_p99, (fov_y - 136.0) / 10.0, (fov_z - 110.0) / 8.0,
            dv20_all * 10.0, rcb_bot_all,
            st_top_mean - st_cor, (js_med - 4.0) / 1.5, isup_imid * 5.0, st_cor
        ], dtype=np.float32)

        cand = ((st >= 0.16) | (st_adj >= 0.16) | (d_out_px <= 2.01)) & (dt2 >= -1.2) & (dt2 <= 5.5)
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

        # ZERO p1_tc6/tb6/tc7, ZERO p2h_tc6/tb6, ZERO dt1, ZERO a_old, ZERO pf_col_max!
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


if __name__ == '__main__':
    print("Loading train80_tibia_prep.joblib & eval20_tibia_prep_v2.joblib (mmap)...", flush=True)
    t0 = time.time()
    prep80 = joblib.load('experiments/cache/train80_tibia_prep.joblib', mmap_mode='r')
    prep20 = joblib.load('experiments/cache/eval20_tibia_prep_v2.joblib', mmap_mode='r')
    print(f"Loaded in {time.time()-t0:.2f}s", flush=True)
    t1 = time.time()
    data_tr = Parallel(n_jobs=16, prefer='threads')(delayed(extract_clean_s3_case)(prep80[i]) for i in range(80))
    data_ev = Parallel(n_jobs=16, prefer='threads')(delayed(extract_clean_s3_case)(prep20[i]) for i in range(20))
    print(f"Extracted clean Stage-3 features ({data_tr[0][0].shape[1]} local + {len(data_tr[0][4])} ctx) in {time.time()-t1:.1f}s", flush=True)

    oof80_glob_declamp = [None] * 80
    oof80_ctx_declamp = [None] * 80

    ev20_glob_declamp = [np.zeros(len(data_ev[i][1]), dtype=np.float32) for i in range(20)]
    ev20_ctx_declamp = [np.zeros(len(data_ev[i][1]), dtype=np.float32) for i in range(20)]

    s3_models_folds = []

    for fold in range(4):
        tf = time.time()
        val_idx = list(range(fold * 20, (fold + 1) * 20))
        tr_idx = [i for i in range(80) if i not in val_idx]
        rng = np.random.default_rng(42 + fold)

        Xs, Xs_ctx, ys = [], [], []
        for i in tr_idx:
            Xi, yi, _, _, ctx_i = data_tr[i]
            idx = rng.choice(len(yi), size=min(len(yi), 10000), replace=False)
            Xi_s = Xi[idx, 5:]
            ctx_mat = np.broadcast_to(ctx_i[None, :], (len(idx), len(ctx_i)))
            Xs.append(Xi_s)
            Xs_ctx.append(np.hstack([Xi_s, ctx_mat]))
            ys.append(yi[idx])
        X_tr = np.vstack(Xs); X_tr_ctx = np.vstack(Xs_ctx); y_tr = np.concatenate(ys)
        del Xs, Xs_ctx, ys; gc.collect()

        clf_d1 = HistGradientBoostingClassifier(max_iter=240, max_leaf_nodes=55, learning_rate=0.05, min_samples_leaf=60, l2_regularization=2.5, max_features=0.60, random_state=242 + fold).fit(X_tr, y_tr)
        clf_cd1 = HistGradientBoostingClassifier(max_iter=240, max_leaf_nodes=55, learning_rate=0.05, min_samples_leaf=60, l2_regularization=2.5, max_features=0.60, random_state=1142 + fold).fit(X_tr_ctx, y_tr)
        del X_tr, X_tr_ctx, y_tr; gc.collect()

        s3_models_folds.append({
            'd1': clf_d1,
            'cd1': clf_cd1,
        })

        def _pred_case(Xi, yi, ctx_i):
            Xi_d = Xi[:, 5:]
            ctx_mat = np.broadcast_to(ctx_i[None, :], (len(yi), len(ctx_i)))
            Xi_ctx = np.hstack([Xi_d, ctx_mat])
            pd = clf_d1.predict_proba(Xi_d)[:, 1].astype(np.float32)
            pcd = clf_cd1.predict_proba(Xi_ctx)[:, 1].astype(np.float32)
            return pd, pcd

        for i in val_idx:
            Xi, yi, _, _, ctx_i = data_tr[i]
            pd, pcd = _pred_case(Xi, yi, ctx_i)
            oof80_glob_declamp[i] = pd
            oof80_ctx_declamp[i] = pcd

        for i in range(20):
            Xi, yi, _, _, ctx_i = data_ev[i]
            pd, pcd = _pred_case(Xi, yi, ctx_i)
            ev20_glob_declamp[i] += 0.25 * pd
            ev20_ctx_declamp[i] += 0.25 * pcd

        print(f"  Fold {fold} trained and evaluated in {time.time()-tf:.1f}s", flush=True)

    for name, tr_l, ev_l in [
        ('glob_declamp', oof80_glob_declamp, ev20_glob_declamp),
        ('ctx_declamp', oof80_ctx_declamp, ev20_ctx_declamp),
        ('blend_65_35', [0.65*oof80_glob_declamp[i]+0.35*oof80_ctx_declamp[i] for i in range(80)],
                        [0.65*ev20_glob_declamp[i]+0.35*ev20_ctx_declamp[i] for i in range(20)]),
    ]:
        d_tr, d_ev = [], []
        for i, p in enumerate(prep80):
            cand = data_tr[i][3]
            gt = (np.array(p['lbl_c'])[cand] == 4)
            part = (np.array(p['st_part'])[cand] > np.array(p['sf_part'])[cand])
            pr = (tr_l[i] >= 0.38) & part
            d_tr.append(2.0 * np.sum(gt & pr) / (int(p['ngt']) + np.sum(pr)))
        for i, p in enumerate(prep20):
            cand = data_ev[i][3]
            gt = (np.array(p['lbl_c'])[cand] == 4)
            part = (np.array(p['st_part'])[cand] > np.array(p['sf_part'])[cand])
            pr = (ev_l[i] >= 0.38) & part
            d_ev.append(2.0 * np.sum(gt & pr) / (int(p['ngt']) + np.sum(pr)))
        print(f"[NO-P1 CLEAN S3] {name:15s}: Train-80 = {np.mean(d_tr):.5f} | Eval-20 = {np.mean(d_ev):.5f} | Gap = {np.mean(d_tr)-np.mean(d_ev):+.5f}", flush=True)

    os.makedirs('experiments/models', exist_ok=True)
    joblib.dump({
        's3_models_folds': s3_models_folds,
    }, 'experiments/models/stage3_clean_train80_models.joblib', compress=3)

    joblib.dump({
        'glob_declamp': oof80_glob_declamp,
        'ctx_declamp': oof80_ctx_declamp,
        'cand': [d[3] for d in data_tr],
        'case_ctx': [d[4] for d in data_tr],
    }, 'experiments/cache/train80_s3_clean_oof.joblib', compress=0)

    joblib.dump({
        'glob_declamp': ev20_glob_declamp,
        'ctx_declamp': ev20_ctx_declamp,
        'cand': [d[3] for d in data_ev],
        'case_ctx': [d[4] for d in data_ev],
    }, 'experiments/cache/eval20_s3_clean_from_train80.joblib', compress=0)

    print("Saved stage3_clean_train80_models.joblib, train80_s3_clean_oof.joblib, and eval20_s3_clean_from_train80.joblib!", flush=True)
