#!/usr/bin/env python3
import os
import sys
import dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

dotenv.load_dotenv()

def main():
    repo_id = "vedangvatsa/vedang-turingbench-roberta-large"
    local_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'roberta_large_semantic_model')
    
    if not os.path.exists(local_dir):
        print(f"Error: Local model directory {local_dir} not found. Please run the training pipeline first.")
        return

    print(f"Loading corrected model and tokenizer from: {local_dir}")
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSequenceClassification.from_pretrained(local_dir)
    
    print(f"\nUploading to Hugging Face Hub repo: {repo_id}")
    print("Note: You must run 'huggingface-cli login' (or 'hf auth login') in your terminal first to authenticate.")
    
    try:
        # Uploading tokenizer (which has the correct 50k vocab size config)
        tokenizer.push_to_hub(repo_id)
        # Uploading model (which has the adapted Beemo weights)
        model.push_to_hub(repo_id)
        print("\nSuccessfully uploaded the corrected tokenizer and adapted model weights to Hugging Face Hub!")
    except Exception as e:
        print(f"\nUpload failed: {e}")
        print("Please verify that you have logged in via the Hugging Face CLI and have write permissions for this repository.")

if __name__ == '__main__':
    main()
