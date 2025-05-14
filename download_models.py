import os
import argparse
import gensim.downloader as api
from gensim.models import KeyedVectors
import yaml

# Create models directory if it doesn't exist
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def list_available_models():
    """List all available models in Gensim."""
    available_models = api.info()['models'].keys()
    print("Available models in Gensim:")
    for model in sorted(available_models):
        print(f"- {model}")

def download_model(model_name, output_dir="models"):
    """Download a specific model from Gensim."""
    ensure_directory(output_dir)
    
    print(f"Downloading model: {model_name}")
    try:
        # Download the model
        model = api.load(model_name)
        
        # Save the model
        model_path = os.path.join(output_dir, f"{model_name}.model")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    except Exception as e:
        print(f"Error downloading model {model_name}: {str(e)}")
        return None

def load_local_model(model_path):
    """Load a model from a local file."""
    try:
        if model_path.endswith('.bin') or model_path.endswith('.vec'):
            model = KeyedVectors.load_word2vec_format(model_path, binary=model_path.endswith('.bin'))
        else:
            model = KeyedVectors.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None

def update_config(model_name, model_path, config_file="config.yaml"):
    """Update the config file with the new model."""
    # Load existing config if it exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    # Initialize models list if it doesn't exist
    if 'models' not in config:
        config['models'] = []
    
    # Check if model already exists in config
    for model in config['models']:
        if model.get('file') == f"{model_name}.json":
            print(f"Model {model_name} already exists in config")
            return
    
    # Add new model to config
    config['models'].append({
        'name': model_name,
        'file': f"{model_name}.json",
        'description': f"Word embeddings from {model_name} model"
    })
    
    # Save updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config file: {config_file}")

def main():
    parser = argparse.ArgumentParser(description="Download and manage word embedding models from Gensim")
    
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--download", type=str, help="Download a specific model")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--load", type=str, help="Load a local model file")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    
    if args.download:
        model_path = download_model(args.download, args.output_dir)
        if model_path:
            update_config(args.download, model_path)
    
    if args.load:
        model = load_local_model(args.load)
        if model:
            model_name = os.path.basename(args.load).split('.')[0]
            update_config(model_name, args.load)

if __name__ == "__main__":
    main()