
import unittest
import numpy as np
import os
try:
    import ml_dtypes
    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False

from kneeseg.features import extract_features, compute_rsid_features, compute_signed_distance_transforms

class TestBFloat16(unittest.TestCase):
    @unittest.skipUnless(HAS_ML_DTYPES, "ml_dtypes not installed")
    def test_extract_features_bfloat16(self):
        # Create dummy image
        img = np.random.rand(20, 20, 20).astype(np.float32)
        dts = {'bone1': np.random.rand(20, 20, 20).astype(np.float32)}
        mask = np.ones((20, 20, 20), dtype=bool) # Full mask
        
        # Test float32 first
        feats_f32 = extract_features(img, dts, mask=mask, target_dtype='float32')
        self.assertEqual(feats_f32.dtype, np.float32)
        
        # Test bfloat16
        feats_bf16 = extract_features(img, dts, mask=mask, target_dtype='bfloat16')
        self.assertEqual(feats_bf16.dtype, ml_dtypes.bfloat16)
        
        # Check values are close
        # bfloat16 has lower precision, so tolerance needs to be appropriate
        diff = np.abs(feats_f32 - feats_bf16.astype(np.float32))
        max_diff = np.max(diff)
        print(f"Max diff between float32 and bfloat16 features: {max_diff}")
        
        # bfloat16 has ~3-4 decimal digits of precision. 
        # Feature values can be large (coordinates) or small.
        # We expect reasonably small difference relative to value magnitude.
        
        # Simple check: Mean difference shouldn't be massive
        self.assertTrue(max_diff < 0.1, f"Difference too large: {max_diff}")

    @unittest.skipUnless(HAS_ML_DTYPES, "ml_dtypes not installed")
    def test_rsid_bfloat16(self):
        img_norm = np.random.randn(20, 20, 20).astype(np.float32)
        
        rsid_f32 = compute_rsid_features(img_norm, num_shifts=5, max_shift=5, dtype=np.float32)
        self.assertEqual(rsid_f32.dtype, np.float32)
        
        rsid_bf16 = compute_rsid_features(img_norm, num_shifts=5, max_shift=5, dtype=ml_dtypes.bfloat16)
        self.assertEqual(rsid_bf16.dtype, ml_dtypes.bfloat16)
        
        # RSID involves subtraction and accumulation.
        diff = np.abs(rsid_f32 - rsid_bf16.astype(np.float32))
        self.assertTrue(np.max(diff) < 0.1)

if __name__ == '__main__':
    unittest.main()
