import unittest
import os
import sys

def run_tests():
    sys.path.insert(0, os.getcwd())
    
    suite = unittest.TestSuite()
    for test_dir in ['kneeseg/tests', 'tests']:
        if os.path.exists(test_dir):
            loader = unittest.TestLoader()
            suite.addTests(loader.discover(test_dir))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
