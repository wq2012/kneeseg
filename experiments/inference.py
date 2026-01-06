
import json
import argparse
from kneeseg.pipeline.inference import inference_improved

def main():
    parser = argparse.ArgumentParser(description="Run KneeSeg inference using configuration file.")
    parser.add_argument('config', help="Path to the JSON configuration file.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    inference_improved(config)

if __name__ == "__main__":
    main()
