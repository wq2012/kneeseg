
import unittest
import os
import json
import tempfile
import shutil
from kneeseg.validate_config import validate_config
from jsonschema.exceptions import ValidationError

class TestValidateConfig(unittest.TestCase):
    def setUp(self):
        # Create a dummy valid config based on schema
        self.valid_config = {
            "data_config": {
                "image_directory": "/tmp/images",
                "label_directory": "/tmp/labels"
            },
            "training_config": {
                "bone_parameters": {
                    "n_estimators": 10,
                    "max_depth": 5
                },
                "cartilage_parameters": {
                    "n_estimators": 10,
                    "max_depth": 5
                }
            },
            "model_config": {
                 "target_bones": ["femur"],
                 "model_directory": "/tmp/models"
            },
            "output_config": {
                "prediction_directory": "/tmp/preds"
            }
        }

    def test_valid_config(self):
        # Should not raise exception
        try:
            validate_config(self.valid_config)
        except Exception as e:
            self.fail(f"validate_config raised {e} unexpectedly!")

    def test_missing_required_data_config(self):
        cfg = self.valid_config.copy()
        del cfg['data_config']
        with self.assertRaises(ValidationError):
            validate_config(cfg)

    def test_missing_required_output_config(self):
        cfg = self.valid_config.copy()
        del cfg['output_config']
        with self.assertRaises(ValidationError):
            validate_config(cfg)
            
    def test_invalid_type(self):
        cfg = self.valid_config.copy()
        cfg['training_config']['bone_parameters']['n_estimators'] = "string_value" # Should be int
        with self.assertRaises(ValidationError):
            validate_config(cfg)

if __name__ == '__main__':
    unittest.main()
