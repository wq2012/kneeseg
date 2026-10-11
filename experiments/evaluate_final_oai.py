"""End-to-end cache-free evaluation on the 31 held-out OAI test scans."""

import argparse
import json
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from kneeseg.pipeline.inference import inference_improved


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate kneeseg OAI Subchondral Atlas model on the 31 held-out OAI test scans.'
    )
    parser.add_argument(
        '--data-dir',
        default=os.environ.get('OAI_DATA_DIR', '/path/to/OAI_refactored/Data'),
        help='Directory containing OAI Data .mhd/.raw volumes.',
    )
    parser.add_argument(
        '--model-dir',
        default='./experiments/models_oai',
        help='Directory containing trained model .joblib files.',
    )
    parser.add_argument(
        '--output-dir',
        default='./experiments/predictions_oai',
        help='Directory to write predicted volumes and evaluation_report.json.',
    )
    args = parser.parse_args()

    os.chdir(base_dir)
    with open('kneeseg/configs/oai_full_inference_config.json', 'r') as f:
        config = json.load(f)
    if args.data_dir:
        config['data_config']['image_directory'] = args.data_dir
        config['data_config']['label_directory'] = os.environ.get(
            'OAI_LABEL_DIR', args.data_dir + '_labels'
        )
    config['model_config']['model_directory'] = args.model_dir
    config['output_config']['prediction_directory'] = args.output_dir
    inference_improved(config)


if __name__ == '__main__':
    main()
