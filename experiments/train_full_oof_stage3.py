"""Train 4-fold Out-Of-Fold Full Stage-3 Tibia Cartilage models on Train-80 using matched OOF Pass-1/2/TC7 features."""
import gc
import os
import sys
import time
import joblib
from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_EXP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)
from build_clean_stage3_train80 import extract_clean_s3_case, shift_ax


def extract_full_oof_s3_case(p, m_override=None):
    """Extract 151 local + 16 context Stage-3 features including matched OOF Pass-1/2/TC7 features."""
    p_eff = dict(p)
    if m_override is not None:
        for k, v in m_override.items():
            p_eff[k] = v
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


def predict_full_oof_s3_case(p, f3_models, m_override=None):
    """Run 4-fold ensemble inference of full_oof_s3_models on a single prepared Tibia case dict."""
    X_g, _, m, ctx_full = extract_full_oof_s3_case(p, m_override=m_override)
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


if __name__ == '__main__':
    t0 = time.time()
    prep80 = joblib.load('experiments/cache/train80_tibia_prep.joblib', mmap_mode='r')
    prep20 = joblib.load('experiments/cache/eval20_tibia_prep_v2.joblib', mmap_mode='r')
    ev_m = joblib.load('experiments/cache/eval20_matched_p1p2.joblib', mmap_mode='r')

    data_tr = Parallel(n_jobs=16, prefer='threads')(
        delayed(extract_full_oof_s3_case)(prep80[i]) for i in range(80)
    )
    data_ev = Parallel(n_jobs=16, prefer='threads')(
        delayed(extract_full_oof_s3_case)(prep20[i], ev_m[i]) for i in range(20)
    )
    print(
        f"Extracted full OOF Stage-3 features ({data_tr[0][0].shape[1]} local + {len(data_tr[0][3])} ctx) in {time.time()-t0:.1f}s",
        flush=True,
    )

    oof_g = [None] * 80
    oof_c = [None] * 80
    ev_g = [np.zeros(len(data_ev[i][1]), dtype=np.float32) for i in range(20)]
    ev_c = [np.zeros(len(data_ev[i][1]), dtype=np.float32) for i in range(20)]
    models_g, models_c = [], []

    for fold in range(4):
        tf = time.time()
        val_idx = list(range(fold * 20, (fold + 1) * 20))
        tr_idx = [i for i in range(80) if i not in val_idx]
        rng = np.random.default_rng(342 + fold)

        Xs_g, Xs_c, ys = [], [], []
        for i in tr_idx:
            Xi_g, yi, _, ctx_i = data_tr[i]
            idx = rng.choice(len(yi), size=min(len(yi), 10000), replace=False)
            Xi_s = Xi_g[idx]
            ctx_mat = np.broadcast_to(ctx_i[None, :], (len(idx), len(ctx_i)))
            Xs_g.append(Xi_s)
            Xs_c.append(np.hstack([Xi_s, ctx_mat]))
            ys.append(yi[idx])
        X_tr_g = np.vstack(Xs_g)
        X_tr_c = np.vstack(Xs_c)
        y_tr = np.concatenate(ys)
        del Xs_g, Xs_c, ys
        gc.collect()

        mg = HistGradientBoostingClassifier(
            max_iter=240, max_leaf_nodes=55, learning_rate=0.05,
            min_samples_leaf=60, l2_regularization=2.5, max_features=0.60,
            random_state=442 + fold,
        ).fit(X_tr_g, y_tr)
        mc = HistGradientBoostingClassifier(
            max_iter=240, max_leaf_nodes=55, learning_rate=0.05,
            min_samples_leaf=60, l2_regularization=2.5, max_features=0.60,
            random_state=1442 + fold,
        ).fit(X_tr_c, y_tr)
        del X_tr_g, X_tr_c, y_tr
        gc.collect()

        models_g.append(mg)
        models_c.append(mc)

        for i in val_idx:
            Xi_g, yi, _, ctx_i = data_tr[i]
            ctx_mat = np.broadcast_to(ctx_i[None, :], (len(yi), len(ctx_i)))
            oof_g[i] = mg.predict_proba(Xi_g)[:, 1].astype(np.float32)
            oof_c[i] = mc.predict_proba(np.hstack([Xi_g, ctx_mat]))[:, 1].astype(np.float32)
        for i in range(20):
            Xi_g, yi, _, ctx_i = data_ev[i]
            ctx_mat = np.broadcast_to(ctx_i[None, :], (len(yi), len(ctx_i)))
            ev_g[i] += 0.25 * mg.predict_proba(Xi_g)[:, 1].astype(np.float32)
            ev_c[i] += 0.25 * mc.predict_proba(np.hstack([Xi_g, ctx_mat]))[:, 1].astype(np.float32)
        print(f"  Full-OOF Stage-3 Fold {fold} done in {time.time()-tf:.1f}s", flush=True)

    joblib.dump(
        {'models_g': models_g, 'models_c': models_c},
        'experiments/models/full_oof_s3_models.joblib',
        compress=3,
    )
    joblib.dump(
        {'oof_g': oof_g, 'oof_c': oof_c, 'ev_g': ev_g, 'ev_c': ev_c},
        'experiments/cache/full_oof_s3_preds.joblib',
        compress=0,
    )
    print("Saved full_oof_s3_models.joblib and full_oof_s3_preds.joblib!", flush=True)
