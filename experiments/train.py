
import json
import argparse
from kneeseg.pipeline.train import train_improved

def main():
    parser = argparse.ArgumentParser(description="Train KneeSeg models using configuration file.")
    parser.add_argument('config', help="Path to the JSON configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train_improved(config)

if __name__ == "__main__":
    main()
