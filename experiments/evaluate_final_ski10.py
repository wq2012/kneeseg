"""End-to-end cache-free evaluation on the 20 held-out SKI10 test scans."""

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
        description='Evaluate kneeseg on the 20 held-out SKI10 test scans.'
    )
    parser.add_argument(
        '--data-dir',
        default=os.environ.get('SKI10_DATA_DIR'),
        help='Directory containing SKI10 TrainingData .mhd/.raw volumes.',
    )
    args = parser.parse_args()

    os.chdir(base_dir)
    with open('kneeseg/configs/ski10_full_inference_config.json', 'r') as f:
        config = json.load(f)
    if args.data_dir:
        config['data_config']['image_directory'] = args.data_dir
        config['data_config']['label_directory'] = os.environ.get(
            'SKI10_LABEL_DIR', args.data_dir + '_labels'
        )
    inference_improved(config)


if __name__ == '__main__':
    main()

