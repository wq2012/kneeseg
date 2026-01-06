
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

if __name__ == '__main__':
    unittest.main()
