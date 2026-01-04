import unittest
import numpy as np
from kneeseg.features import extract_features, compute_signed_distance_transforms

class TestFeatures(unittest.TestCase):
    def test_extract_features(self):
        img = np.zeros((20, 20, 20), dtype=np.int16)
        bone_masks = {'femur': np.zeros_like(img)}
        bone_masks['femur'][5:15, 5:15, 5:15] = 1
        
        dts = compute_signed_distance_transforms(bone_masks)
        features = extract_features(img, dts)
        
        # Number of features check
        # 1(Int) + 1(Gauss) + 1(Grad) + 1(DT-Femur) + 30(RSID) = 34
        self.assertEqual(features.shape[1], 34)
        self.assertEqual(features.shape[0], 20*20*20)

if __name__ == '__main__':
    unittest.main()
