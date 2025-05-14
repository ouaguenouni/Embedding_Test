import os
import json
import argparse
import numpy as np
from sklearn.decomposition import PCA
import gensim.downloader as api
from gensim.models import KeyedVectors

def convert_model_to_json(model, output_file, max_words=10000, num_similar=10):
    """Convert a Gensim model to a JSON format for visualization."""
    print(f"Processing model with {len(model.index_to_key)} words")
    
    # Limit to max_words most common words
    words = model.index_to_key[:max_words]
    
    # Get word vectors
    vectors = [model[word] for word in words]
    
    # Apply PCA to reduce to 2 dimensions
    print("Applying PCA to reduce dimensionality to 2D...")
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(vectors).tolist()
    
    # Find similar words for each word
    print(f"Finding {num_similar} similar words for each word...")
    similar_words = {}
    
    for i, word in enumerate(words):
        if i % 500 == 0:
            print(f"Processing similar words: {i}/{len(words)}")
        
        try:
            similar = model.most_similar(word, topn=num_similar)
            similar_words[word] = [{"word": w, "similarity": float(s)} for w, s in similar]
        except:
            similar_words[word] = []
    
    # Create JSON structure
    output = {
        "words": words,
        "coordinates": coordinates,
        "similarWords": similar_words
    }
    
    # Save as JSON
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output, f)
    
    print("Conversion complete!")
    return output_file

def load_model(model_path_or_name):
    """Load a model from a file or download it from Gensim."""
    try:
        # Check if it's a local file
        if os.path.exists(model_path_or_name):
            print(f"Loading model from {model_path_or_name}")
            if model_path_or_name.endswith('.bin') or model_path_or_name.endswith('.vec'):
                model = KeyedVectors.load_word2vec_format(model_path_or_name, binary=model_path_or_name.endswith('.bin'))
            else:
                model = KeyedVectors.load(model_path_or_name)
        else:
            # Try to download from Gensim
            print(f"Downloading model {model_path_or_name} from Gensim")
            model = api.load(model_path_or_name)
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert Gensim word embedding models to JSON for visualization")
    
    parser.add_argument("--model", type=str, required=True, help="Path to model file or name of Gensim model")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--max-words", type=int, default=10000, help="Maximum number of words to include")
    parser.add_argument("--num-similar", type=int, default=10, help="Number of similar words to find for each word")
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    if model is None:
        return
    
    # Determine output filename if not provided
    if args.output is None:
        model_name = os.path.basename(args.model).split('.')[0] if os.path.exists(args.model) else args.model
        args.output = f"{model_name}.json"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    convert_model_to_json(model, args.output, args.max_words, args.num_similar)

if __name__ == "__main__":
    main()