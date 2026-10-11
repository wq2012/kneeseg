
import unittest
import numpy as np
from kneeseg.features import extract_features
from kneeseg.rf_seg import CartilageClassifier
from kneeseg.bone_rf import BoneClassifier

class TestOAICompatibility(unittest.TestCase):
    def test_feature_extraction_legacy(self):
        # Mock inputs
        img = np.zeros((10, 10, 10), dtype=np.float32)
        dts = {
            'femur': np.ones_like(img),
            'tibia': np.ones_like(img)
            # patella missing
        }
        
        # Legacy call (no target_bones) -> uses sorted(dts.keys()) -> ['femur', 'tibia']
        feats_legacy = extract_features(img, dts)
        print(f"Legacy Features Shape: {feats_legacy.shape}")
        
        # Should have N features. 
        # Intensity(1) + Gauss(1) + Grad(1) + DTs(2) + Arith(2) + RSID(30) + ...
        # Just checking it doesn't crash and has some shape
        self.assertTrue(feats_legacy.shape[1] > 0)
        
    def test_feature_extraction_oai_strict(self):
        img = np.zeros((10, 10, 10), dtype=np.float32)
        dts = {
            'femur': np.ones_like(img),
            'tibia': np.ones_like(img)
            # patella missing in dts, but requested
        }
        target_bones = ['femur', 'patella', 'tibia'] # Intentionally mixed order
        
        feats = extract_features(img, dts, sorted_bones_override=target_bones)
        print(f"OAI Features Shape: {feats.shape}")
        
        # Should be larger than legacy because of extra bone feature
        # And should not crash despite missing 'patella' in dts
        
        feats_legacy = extract_features(img, dts)
        # Difference should be exactly 1 feature (the missing DT)? 
        # Wait, if I add patella, loop runs 3 times instead of 2. So +1 feature.
        self.assertEqual(feats.shape[1], feats_legacy.shape[1] + 1)
        
    def test_cartilage_classifier_init(self):
        clf = CartilageClassifier(target_bones=['femur', 'patella', 'tibia'])
        self.assertEqual(clf.target_bones, ['femur', 'patella', 'tibia'])
        
    def test_bone_classifier_mapping(self):
        # We can't easily test internal mapping without mocking train, 
        # but we can check if it runs without error on 6-label input
        clf = BoneClassifier(n_estimators=1)
        img = np.zeros((10, 10, 10), dtype=np.float32)
        lbl = np.zeros((10, 10, 10), dtype=np.uint8)
        lbl[0,0,0] = 5 # Patella
        lbl[0,0,1] = 6 # PatCart
        
        # This will fail if I didn't update mapping logic to handle 5/6
        # But wait, subsampling might miss these few pixels if random.
        # Let's fill more
        lbl[:,:,:] = 5
        
        try:
            clf.train([img], [lbl], subsample=100)
            success = True
        except Exception as e:
            print(f"Bone Train Error: {e}")
            success = False
            
        self.assertTrue(success)

    def test_bone_classifier_pass1_and_pass2_no_name_error(self):
        clf_p1 = BoneClassifier(n_estimators=2, max_depth=4, n_jobs=1)
        img = np.zeros((32, 32, 32), dtype=np.float32)
        img[8:16, 8:24, 8:24] = 100.0
        img[18:26, 8:24, 8:24] = 120.0
        lbl = np.zeros((32, 32, 32), dtype=np.uint8)
        lbl[8:16, 8:24, 8:24] = 1
        lbl[16:18, 10:22, 10:22] = 2
        lbl[18:26, 8:24, 8:24] = 3
        clf_p1.train([img, img + 1.0], [lbl, lbl], spacings=[(1.5, 1.5, 1.5), (1.5, 1.5, 1.5)])
        _, pm1 = clf_p1.predict(img, spacing=(1.5, 1.5, 1.5))
        _, pm2 = clf_p1.predict(img + 1.0, spacing=(1.5, 1.5, 1.5))
        clf_p2 = BoneClassifier(n_estimators=2, max_depth=4, n_jobs=1)
        clf_p2.train([img, img + 1.0], [lbl, lbl], prob_maps=[pm1, pm2], spacings=[(1.5, 1.5, 1.5), (1.5, 1.5, 1.5)])
        pred_p2, prob_p2 = clf_p2.predict(img, prob_map=pm1, spacing=(1.5, 1.5, 1.5))
        self.assertEqual(pred_p2.shape, img.shape)
        self.assertEqual(prob_p2.shape[:3], img.shape)

    def test_v2_bone_and_v6_cartilage_uncached_volume(self):
        import os
        from kneeseg.bone_rf import _BONE_OOF_PROB_CACHE, _BONE_V2_SESSION_CACHE
        from kneeseg.cartilage_v6 import _CART_P1_SESSION_CACHE

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        b1_path = os.path.join(base_dir, 'experiments', 'models', 'bone_rf_p1.joblib')
        b2_path = os.path.join(base_dir, 'experiments', 'models', 'bone_rf_p2.joblib')
        c1_path = os.path.join(base_dir, 'experiments', 'models', 'cartilage_rf_p1.joblib')
        c2_path = os.path.join(base_dir, 'experiments', 'models', 'cartilage_rf_p2.joblib')
        if not (os.path.exists(b1_path) and os.path.exists(b2_path) and os.path.exists(c1_path) and os.path.exists(c2_path)):
            return
        b1 = BoneClassifier()
        b1.load(b1_path)
        b2 = BoneClassifier()
        b2.load(b2_path)
        c1 = CartilageClassifier()
        c1.load(c1_path)
        c2 = CartilageClassifier()
        c2.load(c2_path)

        ref_v2 = b2._get_ref_v2()
        for bone_key in ('tibia', 'femur'):
            for et_key in ('et_v1', 'et_v2'):
                n_j = getattr(ref_v2[bone_key][et_key], 'n_jobs', 1)
                self.assertTrue(1 <= n_j <= 16, f"{bone_key}.{et_key}.n_jobs={n_j} should be bounded")

        # Synthetic uncached volume (fingerprint guaranteed not in eval cache)
        img = np.zeros((36, 40, 40), dtype=np.float32)
        img[8:18, 6:20, 10:30] = 150.0
        img[8:18, 22:36, 10:30] = 160.0
        bm_pred1, bm_prob1 = b1.predict(img, spacing=(1.5, 1.5, 1.5))
        self.assertEqual(bm_pred1.shape, img.shape)
        self.assertEqual(bm_prob1.shape, (*img.shape, 3))
        self.assertTrue(np.any(bm_prob1[..., 1:3] > 0.0))

        bm_pred, bm_prob = b2.predict(img, prob_map=bm_prob1, spacing=(1.5, 1.5, 1.5))
        self.assertEqual(bm_pred.shape, img.shape)
        self.assertEqual(bm_prob.shape[:3], img.shape)

        synth_bones = {'femur': np.zeros_like(img, dtype=np.uint8), 'tibia': np.zeros_like(img, dtype=np.uint8)}
        synth_bones['femur'][8:18, 6:20, 10:30] = 1
        synth_bones['tibia'][8:18, 22:36, 10:30] = 1
        c_prob1 = c1.predict_proba_map(img, synth_bones, spacing=(1.5, 1.5, 1.5))
        self.assertEqual(c_prob1.shape, (*img.shape, 3))
        self.assertTrue(np.any(c_prob1[..., 1:3] > 0.0))

        y_cart, p_cart = c2.predict(img, synth_bones, prob_map=c_prob1, spacing=(1.5, 1.5, 1.5))
        self.assertEqual(y_cart.shape, img.shape)
        self.assertEqual(p_cart.shape[:3], img.shape)

        # Verify repeat c2.predict produces identical output and preserves session cache
        y_cart_rep, p_cart_rep = c2.predict(img, synth_bones, prob_map=c_prob1, spacing=(1.5, 1.5, 1.5))
        np.testing.assert_array_equal(y_cart, y_cart_rep)
        np.testing.assert_allclose(p_cart, p_cart_rep)

        # Verify clearing session caches still recovers identical Pass-1 and Pass-2 predictions
        _BONE_V2_SESSION_CACHE.clear()
        _BONE_OOF_PROB_CACHE.clear()
        _CART_P1_SESSION_CACHE.clear()
        c_prob1_nocache = c1.predict_proba_map(img, synth_bones, spacing=(1.5, 1.5, 1.5))
        y_cart_nocache, p_cart_nocache = c2.predict(img, synth_bones, prob_map=c_prob1_nocache, spacing=(1.5, 1.5, 1.5))
        np.testing.assert_allclose(c_prob1, c_prob1_nocache)
        np.testing.assert_array_equal(y_cart, y_cart_nocache)
        np.testing.assert_allclose(p_cart, p_cart_nocache)

    def test_no_cache_or_sys_path_hacks_in_kneeseg_package(self):
        import os
        import kneeseg.pipeline.inference as mod_inf
        import kneeseg.pipeline.train as mod_trn
        import kneeseg.pipeline.run as mod_run
        self.assertTrue(hasattr(mod_inf, 'inference_improved'))
        self.assertTrue(hasattr(mod_trn, 'train_improved'))
        self.assertTrue(hasattr(mod_run, 'main'))

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pkg_dir = os.path.join(base_dir, 'kneeseg')
        for root, _, files in os.walk(pkg_dir):
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, base_dir)
                with open(full_path, 'r', encoding='utf-8') as f:
                    src = f.read()
                self.assertNotIn('experiments/cache', src, f"Found cache lookup in {rel_path}")
                self.assertNotIn("os.path.join(base_dir, 'experiments', 'cache')", src, f"Found cache lookup in {rel_path}")
                self.assertNotIn('sys.path.insert', src, f"Found sys.path.insert in {rel_path}")
                self.assertNotIn('/usr/local/google/home/quanw', src, f"Found hardcoded user path in {rel_path}")

    def test_legacy_v1_raw_rf_checkpoint_backward_compatibility(self):
        import os
        import tempfile
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        img = np.random.RandomState(0).rand(12, 12, 12).astype(np.float32)
        lbl_oai = np.zeros((12, 12, 12), dtype=np.uint8)
        lbl_oai[2:5, 2:5, 2:5] = 1
        lbl_oai[5:6, 2:5, 2:5] = 2
        lbl_oai[6:9, 2:5, 2:5] = 3
        lbl_oai[9:10, 2:5, 2:5] = 4
        lbl_oai[2:5, 6:9, 2:5] = 5
        lbl_oai[5:6, 6:9, 2:5] = 6

        bone_masks_oai = {
            'femur': (lbl_oai == 1).astype(np.uint8),
            'tibia': (lbl_oai == 3).astype(np.uint8),
            'patella': (lbl_oai == 5).astype(np.uint8),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train & dump raw sklearn RandomForestClassifier objects like v1 HF repos
            b1_train = BoneClassifier(n_estimators=2, max_depth=3, n_jobs=1)
            b1_train.clf = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=42)
            b1_train.train([img], [lbl_oai], subsample=200)
            b1_path = os.path.join(tmpdir, 'bone_rf_p1.joblib')
            joblib.dump(b1_train.clf, b1_path)

            c1_train = CartilageClassifier(n_estimators=2, max_depth=3, n_jobs=1, target_bones=['femur', 'tibia', 'patella'])
            c1_train.clf = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=42)
            c1_train.train([img], [bone_masks_oai], [lbl_oai])
            c1_path = os.path.join(tmpdir, 'cartilage_rf_p1.joblib')
            joblib.dump(c1_train.clf, c1_path)

            # Load without passing target_bones (as in HF README snippets)
            b1_loaded = BoneClassifier()
            b1_loaded.load(b1_path)
            self.assertEqual(b1_loaded.stage, 0)
            self.assertEqual(b1_loaded.bone_Map, {'femur': 1, 'tibia': 2, 'patella': 3})
            b_pred, b_prob = b1_loaded.predict(img)
            self.assertEqual(b_pred.shape, img.shape)
            self.assertEqual(b_prob.shape, (*img.shape, 4))

            # Verify pass1_probs alias works on BoneClassifier.predict
            b2_train = BoneClassifier(n_estimators=2, max_depth=3, n_jobs=1)
            b2_train.clf = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=42)
            b2_train.train([img], [lbl_oai], prob_maps=[b_prob], subsample=200)
            b2_path = os.path.join(tmpdir, 'bone_rf_p2.joblib')
            joblib.dump(b2_train.clf, b2_path)
            b2_loaded = BoneClassifier()
            b2_loaded.load(b2_path)
            pred_b2_a, _ = b2_loaded.predict(img, prob_map=b_prob)
            pred_b2_b, _ = b2_loaded.predict(img, pass1_probs=b_prob)
            np.testing.assert_array_equal(pred_b2_a, pred_b2_b)

            c1_loaded = CartilageClassifier()
            c1_loaded.load(c1_path)
            self.assertEqual(c1_loaded.stage, 0)
            self.assertEqual(c1_loaded.target_bones, ['femur', 'tibia', 'patella'])
            c_prob = c1_loaded.predict_proba_map(img, bone_masks_oai)
            c_pred, _ = c1_loaded.predict(img, bone_masks_oai)
            self.assertEqual(c_pred.shape, img.shape)
            self.assertEqual(c_prob.shape, (*img.shape, 7))

            # Verify pass1_probs alias works on CartilageClassifier.predict
            c2_train = CartilageClassifier(n_estimators=2, max_depth=3, n_jobs=1, target_bones=['femur', 'tibia', 'patella'])
            c2_train.clf = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=42)
            c2_train.train([img], [bone_masks_oai], [lbl_oai], prob_maps=[c_prob])
            c2_path = os.path.join(tmpdir, 'cartilage_rf_p2.joblib')
            joblib.dump(c2_train.clf, c2_path)
            c2_loaded = CartilageClassifier()
            c2_loaded.load(c2_path)
            pred_c2_a, _ = c2_loaded.predict(img, bone_masks_oai, prob_map=c_prob)
            pred_c2_b, _ = c2_loaded.predict(img, bone_masks_oai, pass1_probs=c_prob)
            np.testing.assert_array_equal(pred_c2_a, pred_c2_b)

    def test_oai_subchondral_atlas_and_unified_pipeline(self):
        import os
        from kneeseg.oai_subchondral_atlas import (
            _OAI_BONE_AUX_CACHE,
            _OAI_CART_P1_CACHE,
            detect_oai_layout,
            from_canonical_oai,
            to_canonical_oai,
        )

        # Verify canonical coordinate transforms round-trip for C-order ('oai_c'), F-order ('oai_f'), and native layouts
        zz, yy, xx = np.meshgrid(
            np.linspace(0, 1, 112, dtype=np.float32),
            np.linspace(0, 1, 140, dtype=np.float32),
            np.linspace(0, 1, 140, dtype=np.float32),
            indexing='ij',
        )
        smooth_f = (25.0 * zz + 50.0 * yy + 25.0 * xx).astype(np.float32)
        smooth_c = smooth_f.ravel(order='F').reshape((112, 140, 140))
        native_arr = np.arange(32 * 36 * 40, dtype=np.int32).reshape((32, 36, 40))

        for arr, expected_layout in [(smooth_c, 'oai_c'), (smooth_f, 'oai_f'), (native_arr, 'native')]:
            layout = detect_oai_layout(arr)
            self.assertEqual(layout, expected_layout)
            c_arr = to_canonical_oai(arr, layout=layout)
            if expected_layout != 'native':
                self.assertEqual(c_arr.shape, (112, 140, 140))
            rt_arr = from_canonical_oai(c_arr, layout=layout)
            np.testing.assert_allclose(arr, rt_arr, atol=1e-5)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        oai_dir = os.path.join(base_dir, 'experiments', 'models_oai')
        if all(os.path.exists(os.path.join(oai_dir, fn)) for fn in ('bone_rf_p1.joblib', 'bone_rf_p2.joblib', 'cartilage_rf_p1.joblib', 'cartilage_rf_p2.joblib')):
            b1 = BoneClassifier()
            b1.load(os.path.join(oai_dir, 'bone_rf_p1.joblib'))
            b2 = BoneClassifier()
            b2.load(os.path.join(oai_dir, 'bone_rf_p2.joblib'))
            c1 = CartilageClassifier()
            c1.load(os.path.join(oai_dir, 'cartilage_rf_p1.joblib'))
            c2 = CartilageClassifier()
            c2.load(os.path.join(oai_dir, 'cartilage_rf_p2.joblib'))

            img_oai = np.zeros((36, 44, 44), dtype=np.float32)
            img_oai[6:30, 6:18, 10:34] = 160.0
            img_oai[6:30, 22:38, 10:34] = 150.0
            img_oai[10:26, 8:20, 35:42] = 170.0
            bp1, bprob1 = b1.predict(img_oai, spacing=(3.0, 3.0, 3.0))
            self.assertEqual(bp1.shape, img_oai.shape)
            self.assertEqual(bprob1.shape, (*img_oai.shape, 4))
            bp2, bprob2 = b2.predict(img_oai, prob_map=bprob1, spacing=(3.0, 3.0, 3.0))
            self.assertEqual(bp2.shape, img_oai.shape)
            self.assertEqual(bprob2.shape, (*img_oai.shape, 4))

            bm_oai = {
                'femur': (img_oai == 160.0).astype(np.uint8),
                'tibia': (img_oai == 150.0).astype(np.uint8),
                'patella': (img_oai == 170.0).astype(np.uint8),
            }
            cprob1 = c1.predict_proba_map(img_oai, bm_oai, spacing=(3.0, 3.0, 3.0))
            self.assertEqual(cprob1.shape, (*img_oai.shape, 7))
            y_c2, p_c2 = c2.predict(img_oai, bm_oai, prob_map=cprob1, spacing=(3.0, 3.0, 3.0))
            self.assertEqual(y_c2.shape, img_oai.shape)
            self.assertEqual(p_c2.shape, (*img_oai.shape, 7))
            _OAI_BONE_AUX_CACHE.clear()
            _OAI_CART_P1_CACHE.clear()

        uni_dir = os.path.join(base_dir, 'experiments', 'models_unified')
        if all(os.path.exists(os.path.join(uni_dir, fn)) for fn in ('bone_rf_p1.joblib', 'bone_rf_p2.joblib', 'cartilage_rf_p1.joblib', 'cartilage_rf_p2.joblib')):
            ub1 = BoneClassifier()
            ub1.load(os.path.join(uni_dir, 'bone_rf_p1.joblib'))
            ub2 = BoneClassifier()
            ub2.load(os.path.join(uni_dir, 'bone_rf_p2.joblib'))
            uc1 = CartilageClassifier()
            uc1.load(os.path.join(uni_dir, 'cartilage_rf_p1.joblib'))
            uc2 = CartilageClassifier()
            uc2.load(os.path.join(uni_dir, 'cartilage_rf_p2.joblib'))

            img_ski = np.zeros((36, 40, 40), dtype=np.float32)
            img_ski[8:18, 6:20, 10:30] = 150.0
            img_ski[8:18, 22:36, 10:30] = 160.0
            s_bp1, s_prob1 = ub1.predict(img_ski, spacing=(0.5, 0.5, 0.5))
            self.assertEqual(s_bp1.shape, img_ski.shape)
            s_bp2, _ = ub2.predict(img_ski, prob_map=s_prob1, spacing=(0.5, 0.5, 0.5))
            self.assertEqual(s_bp2.shape, img_ski.shape)


if __name__ == '__main__':
    unittest.main()



