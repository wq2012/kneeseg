
import unittest
import os
import sys

def run_tests():
    # Add current directory to path
    sys.path.append(os.getcwd())
    
    loader = unittest.TestLoader()
    start_dir = 'delivery/tests'
    suite = loader.discover(start_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
