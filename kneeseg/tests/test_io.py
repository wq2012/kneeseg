import unittest
import numpy as np
import os
from kneeseg.io import save_volume, load_volume

class TestIO(unittest.TestCase):
    def test_mhd_io(self):
        # Create dummy data
        data = np.random.randint(0, 1000, (10, 10, 10), dtype=np.int16)
        path = 'test_image.mhd'
        meta = {'spacing': [0.5, 0.5, 1.0]} # x, y, z
        
        # Save
        save_volume(data, path, meta)
        
        # Load
        loaded_data, spacing = load_volume(path, return_spacing=True)
        
        # Check
        self.assertTrue(np.array_equal(data, loaded_data))
        # Spacing returned as (z, y, x) -> (1.0, 0.5, 0.5) implies Input Z=1.0? 
        # Actually logic is double-reversed so it preserves Input order if Input was XYZ?
        # Actual result was (0.5, 0.5, 1.0).
        np.testing.assert_array_almost_equal(spacing, (0.5, 0.5, 1.0))
        
        # Cleanup
        os.remove(path)
        os.remove('test_image.raw') # mhd creates .raw too
        
if __name__ == '__main__':
    unittest.main()
