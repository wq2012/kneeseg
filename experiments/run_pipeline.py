
import os
import json
import argparse
from kneeseg.pipeline.train import train_improved
from kneeseg.pipeline.inference import inference_improved

def load_default_configs():
    # Load from package configs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, 'configs')
    
    train_cfg_path = os.path.join(config_dir, 'ski10_full_train_config.json')
    infer_cfg_path = os.path.join(config_dir, 'ski10_full_inference_config.json')
    
    with open(train_cfg_path, 'r') as f:
        train_config = json.load(f)
    with open(infer_cfg_path, 'r') as f:
        infer_config = json.load(f)
        
    return train_config, infer_config

def main():
    print("=== KneeSeg Pipeline (Semantic Context Forests) ===")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help="Override image directory from config")
    args = parser.parse_args()
    
    train_config, infer_config = load_default_configs()
    
    if args.data_dir:
        # Override data paths
        train_config['data_config']['image_directory'] = args.data_dir
        train_config['data_config']['label_directory'] = args.data_dir + "_labels"
        
        infer_config['data_config']['image_directory'] = args.data_dir
        infer_config['data_config']['label_directory'] = args.data_dir + "_labels"
        
    model_dir = train_config['model_config']['model_directory']
    eval_dir = infer_config['output_config']['prediction_directory']
    
    # 2. Check Training Status
    required_models = [
        'bone_rf_p1.joblib', 
        'bone_rf_p2.joblib', 
        'cartilage_rf_p1.joblib',
        'cartilage_rf_p2.joblib'
    ]
    all_models_exist = all(os.path.exists(os.path.join(model_dir, m)) for m in required_models)
    
    if all_models_exist:
        print(f"[SKIP] Training: All models found in {model_dir}")
    else:
        print(f"[RUN] Training: Missing models using {model_dir}...")
        try:
            train_improved(train_config)
        except Exception as e:
            print(f"Training Failed: {e}")
            raise

    # 3. Check Inference Status
    report_path = os.path.join(eval_dir, 'evaluation_report.json')
    if os.path.exists(report_path):
        print(f"[SKIP] Inference: Evaluation report found at {report_path}")
        with open(report_path, 'r') as f:
            print(json.dumps(json.load(f), indent=2))
    else:
        print(f"[RUN] Inference: Report not found. Starting inference...")
        inference_improved(infer_config)

if __name__ == "__main__":
    main()
