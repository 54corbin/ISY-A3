import argparse
import pickle
import numpy as np
import re
import nltk
import os
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

DEFAULT_MODEL_PATH = "sentiment_category_model.keras"
DEFAULT_TOKENIZER_PATH = "tokenizer.pickle"
DEFAULT_MAPPING_PATH = "category_mapping.pickle"

# --- NLTK Resource Management ---
def ensure_nltk_data_path():
    """Ensure NLTK knows where to find its data, adding common user paths if necessary."""
    # Print current NLTK data paths for debugging
    print(f"Current NLTK data paths: {nltk.data.path}")
    
    # Common NLTK data paths
    common_paths = [
        os.path.join(os.path.expanduser("~"), "nltk_data"),
        os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data"), 
        "/usr/share/nltk_data",
        "/usr/local/share/nltk_data",
        "/usr/lib/nltk_data",
        "/usr/local/lib/nltk_data"
    ]
    
    # Add current script's directory + nltk_data as a potential path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir_nltk_data = os.path.join(script_dir, "nltk_data")
    common_paths.insert(0, script_dir_nltk_data) # Prioritize local nltk_data folder
    
    # Also add a nltk_data directory in the current working directory
    cwd_nltk_data = os.path.join(os.getcwd(), "nltk_data")
    common_paths.insert(0, cwd_nltk_data)

    # Add all paths to nltk.data.path
    for path in common_paths:
        if path not in nltk.data.path:
            nltk.data.path.append(path)
            print(f"Added NLTK data path: {path}")

    # If an NLTK_DATA environment variable is set, ensure it's in the path
    nltk_data_env = os.environ.get('NLTK_DATA')
    if nltk_data_env and nltk_data_env not in nltk.data.path:
        nltk.data.path.append(nltk_data_env)
        print(f"Added NLTK_DATA environment path: {nltk_data_env}")

    # Create nltk_data directory in the script directory if it doesn't exist
    os.makedirs(script_dir_nltk_data, exist_ok=True)
    os.makedirs(cwd_nltk_data, exist_ok=True)
    
    print(f"Updated NLTK data search paths: {nltk.data.path}")

def download_nltk_resources():
    """Downloads necessary NLTK data if not already present."""
    ensure_nltk_data_path()
    resources = [('corpora/stopwords', 'stopwords'), ('tokenizers/punkt', 'punkt')]
    all_downloaded = True
    
    for path_suffix, name in resources:
        try:
            # Check if the resource can be found through NLTK's standard find mechanism
            nltk.data.find(path_suffix)
            print(f"NLTK resource '{name}' found.")
        except LookupError:
            print(f"NLTK resource '{name}' not found. Downloading...")
            try:
                # Force download to a writable location
                download_dir = os.path.join(os.getcwd(), "nltk_data")
                os.makedirs(download_dir, exist_ok=True)
                nltk.download(name, download_dir=download_dir)
                print(f"Successfully downloaded '{name}' to {download_dir}")
                
                # Add the download directory to NLTK's search path
                if download_dir not in nltk.data.path:
                    nltk.data.path.insert(0, download_dir)
                    print(f"Added download directory to NLTK path: {download_dir}")
                
                # Verify again after download
                try:
                    nltk.data.find(path_suffix)
                    print(f"Verified '{name}' is now findable.")
                except LookupError:
                    print(f"Warning: '{name}' downloaded but still not findable. Check NLTK data paths.")
                    all_downloaded = False
            except Exception as e:
                print(f"Error downloading '{name}': {e}")
                print(f"Please try running 'python -m nltk.downloader {name}' manually or ensure internet connection.")
                all_downloaded = False
    
    # Test actual functionality instead of checking for specific resource files
    try:
        print("Testing NLTK tokenization...")
        test_tokenize = word_tokenize("This is a test.")
        print(f"Tokenization test result: {test_tokenize}")
        
        print("Testing NLTK stopwords...")
        test_stopwords = set(stopwords.words('english'))
        print(f"First few stopwords: {list(test_stopwords)[:5]}")
        
        print("NLTK functionality verified.")
        return True
    except Exception as e:
        print(f"NLTK functionality test failed: {e}")
        return False

# --- Text Cleaning ---
def clean_text_for_prediction(text, stop_words_set_local):
    """Cleans input text using the same steps as training."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    try:
        words = word_tokenize(text)
        print(f"Tokenization successful: {words[:5]}...")
    except Exception as e:
        print(f"CRITICAL Error during tokenization: {e}")
        print("Current NLTK paths:", nltk.data.path)
        return None

    try:
        words = [word for word in words if word not in stop_words_set_local and len(word) > 2]
    except Exception as e:
        print(f"CRITICAL Error during stopword filtering: {e}")
        return None

    cleaned_review = " ".join(words)
    return cleaned_review

# --- Main Application Logic ---
def main(args):
    """Loads artifacts, takes input, preprocesses, predicts, and shows output."""

    print("--- Sentiment and Category Predictor ---")

    # 1. Download/Verify NLTK data
    print("Checking NLTK resources...")
    if not download_nltk_resources():
        print("NLTK verification failed. Attempting direct download and test...")
        try:
            # Direct download attempt as a last resort
            nltk.download('all')
            # nltk.download('stopwords')
            
            test_tokens = word_tokenize("Testing tokenization after direct download.")
            test_stops = set(stopwords.words('english'))
            
            if test_tokens and test_stops:
                print("Direct download successful. Proceeding with the application.")
            else:
                print("Direct download unsuccessful. Exiting.")
                return
        except Exception as e:
            print(f"Direct download failed: {e}")
            print("Exiting due to missing NLTK resources.")
            return

    try:
        stop_words_set_main = set(stopwords.words('english'))
        print(f"Successfully loaded {len(stop_words_set_main)} stopwords.")
    except Exception as e:
        print(f"Failed to load stopwords: {e}")
        return
    print("NLTK resources ready.")

    # Load Artifacts
    model_path = args.model_path if args.model_path else DEFAULT_MODEL_PATH
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else DEFAULT_TOKENIZER_PATH
    mapping_path = args.mapping_path if args.mapping_path else DEFAULT_MAPPING_PATH

    print(f"Attempting to load model from: {os.path.abspath(model_path)}")
    print(f"Attempting to load tokenizer from: {os.path.abspath(tokenizer_path)}")
    print(f"Attempting to load mapping from: {os.path.abspath(mapping_path)}")

    try:
        model = load_model(model_path)
        print("Model loaded successfully.")

        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully.")

        with open(mapping_path, 'rb') as handle:
            category_mapping = pickle.load(handle)
        print("Category mapping loaded successfully.")

        index_to_category = {v: k for k, v in category_mapping.items()}

    except FileNotFoundError as e:
        print(f"Error: One or more artifact files not found. Searched for:\n"
              f"Model: {os.path.abspath(model_path)}\n"
              f"Tokenizer: {os.path.abspath(tokenizer_path)}\n"
              f"Mapping: {os.path.abspath(mapping_path)}\n"
              f"Details: {e}\n"
              f"Please ensure the files are in the correct location or provide paths using arguments.")
        return
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return

    # Get MAX_LEN from the loaded model's input shape
    try:
        max_seq_len = model.input_shape[-1]
        if max_seq_len is None:
             raise ValueError("Could not automatically determine MAX_LEN from model input shape.")
        print(f"Determined MAX_LEN from model: {max_seq_len}")
    except Exception as e:
        print(f"Warning: Error determining MAX_LEN from model: {e}")
        try:
            max_seq_len = int(input("Could not determine MAX_LEN. Please enter the sequence length used during training (e.g., 120): "))
            if max_seq_len <= 0:
                raise ValueError
        except ValueError:
            print("Invalid input for MAX_LEN. Exiting.")
            return

    print("\nReady for input.")
    print("---------------------------------------")

    # Input Loop
    while True:
        try:
            text_input = input("Enter text for analysis (or type 'quit' to exit): ")
            if text_input.lower() == 'quit':
                break
            if not text_input.strip():
                print("Please enter some text.")
                continue

            cleaned_text = clean_text_for_prediction(text_input, stop_words_set_main)
            if cleaned_text is None:
                 continue

            if not cleaned_text:
                print("Text resulted in empty sequence after cleaning. Cannot predict.")
                continue

            sequences = tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequences, maxlen=max_seq_len, padding='post', truncating='post')

            predictions = model.predict(padded_sequence, verbose=0)

            sentiment_pred_prob = predictions[0][0][0]
            category_pred_probs = predictions[1][0]

            sentiment_label = "Positive" if sentiment_pred_prob >= 0.5 else "Negative"
            sentiment_confidence = sentiment_pred_prob * 100 if sentiment_label == "Positive" else (1 - sentiment_pred_prob) * 100

            predicted_category_index = np.argmax(category_pred_probs)
            predicted_category_label = index_to_category.get(predicted_category_index, "Unknown Category")
            category_confidence = category_pred_probs[predicted_category_index] * 100

            print("\n--- Prediction Results ---")
            print(f"Sentiment: {sentiment_label} (Confidence: {sentiment_confidence:.2f}%)")
            print(f"Category:  {predicted_category_label} (Confidence: {category_confidence:.2f}%)")
            print("---------------------------------------\n")

        except EOFError:
             print("\nExiting.")
             break
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please try again or type 'quit' to exit.")

    print("\n--- Program Finished ---")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict sentiment and category from text using a trained Keras model.\n"
                    "Loads artifacts from the current directory by default.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help=f"Path to the saved Keras model file (.keras).\nDefault: ./{DEFAULT_MODEL_PATH}")
    parser.add_argument("-t", "--tokenizer_path", type=str, default=None,
                        help=f"Path to the saved tokenizer file (.pickle).\nDefault: ./{DEFAULT_TOKENIZER_PATH}")
    parser.add_argument("-c", "--mapping_path", type=str, default=None,
                        help=f"Path to the saved category mapping file (.pickle).\nDefault: ./{DEFAULT_MAPPING_PATH}")

    args = parser.parse_args()
    main(args)
