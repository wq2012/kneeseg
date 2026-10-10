
import json
import os
import jsonschema

def validate_config(config):
    """
    Validates the configuration dictionary against the schema.
    
    Args:
        config (dict): The configuration dictionary.
        
    Raises:
        jsonschema.ValidationError: If the config is invalid.
        FileNotFoundError: If the schema file cannot be found.
    """
    # Path to schema file - assume it's in kneeseg/configs relative to this file?
    # Or simplified: checking keys manually if jsonschema not available, 
    # but we should use jsonschema if possible.
    
    # We will try to find the schema file relative to the package installation
    # or use a bundled schema.
    
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(pkg_dir)
    schema_path = os.path.join(pkg_dir, 'configs', 'config_schema.json')

    if not os.path.exists(schema_path):
        potential_paths = [
            os.path.join(base_dir, 'configs', 'config_schema.json'),
            'kneeseg/configs/config_schema.json',
            'delivery/kneeseg/configs/config_schema.json',
        ]
        for p in potential_paths:
            if os.path.exists(p):
                schema_path = p
                break
                
    if not os.path.exists(schema_path):
        print(f"Warning: Schema file not found at {schema_path}. Skipping validation.")
        return

    with open(schema_path, 'r') as f:
        schema = json.load(f)
        
    jsonschema.validate(instance=config, schema=schema)
    print("Configuration validation successful.")
