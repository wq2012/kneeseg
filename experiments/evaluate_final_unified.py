"""End-to-end cache-free evaluation of the Unified SKI10+OAI model on both OAI (31 eval) and SKI10 (20 eval)."""

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
        description='Evaluate unified SKI10+OAI model on 31 OAI test scans and 20 SKI10 test scans.'
    )
    parser.add_argument(
        '--oai-data-dir',
        default=os.environ.get('OAI_DATA_DIR', '/path/to/OAI_refactored/Data'),
        help='Directory containing OAI Data .mhd/.raw volumes.',
    )
    parser.add_argument(
        '--ski10-data-dir',
        default=os.environ.get('SKI10_DATA_DIR', '/path/to/SKI10/TrainingData'),
        help='Directory containing SKI10 TrainingData .mhd/.raw volumes.',
    )
    parser.add_argument(
        '--dataset',
        choices=['oai', 'ski10', 'both'],
        default='both',
        help='Which dataset(s) to evaluate.',
    )
    args = parser.parse_args()

    os.chdir(base_dir)
    if args.dataset in ('oai', 'both'):
        with open('kneeseg/configs/unified_oai_inference_config.json', 'r') as f:
            cfg_oai = json.load(f)
        cfg_oai['data_config']['image_directory'] = args.oai_data_dir
        cfg_oai['data_config']['label_directory'] = os.environ.get(
            'OAI_LABEL_DIR', args.oai_data_dir + '_labels'
        )
        print("\n=== Evaluating Unified Model on 31 Held-Out OAI Scans ===", flush=True)
        inference_improved(cfg_oai)

    if args.dataset in ('ski10', 'both'):
        with open('kneeseg/configs/unified_ski10_inference_config.json', 'r') as f:
            cfg_ski10 = json.load(f)
        cfg_ski10['data_config']['image_directory'] = args.ski10_data_dir
        cfg_ski10['data_config']['label_directory'] = os.environ.get(
            'SKI10_LABEL_DIR', args.ski10_data_dir + '_labels'
        )
        print("\n=== Evaluating Unified Model on 20 Held-Out SKI10 Scans ===", flush=True)
        inference_improved(cfg_ski10)


if __name__ == '__main__':
    main()
