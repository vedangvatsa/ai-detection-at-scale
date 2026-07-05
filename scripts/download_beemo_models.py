#!/usr/bin/env python3
import os
import subprocess
import zipfile
import shutil

MODELS_DIR = "/Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/models"
KERNEL_ID = "vedangvatsa123/ai-detection-beemo-sota-hybrid"

def main():
    print(f"Creating models directory at {MODELS_DIR} if it does not exist...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"Downloading outputs from Kaggle kernel: {KERNEL_ID}...")
    # Run kaggle kernels output
    cmd = [
        ".venv/bin/kaggle", "kernels", "output",
        KERNEL_ID, "-p", MODELS_DIR
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error downloading outputs from Kaggle:")
        print(result.stderr)
        return
    
    print("Download complete. Checking downloaded files...")
    zip_path = os.path.join(MODELS_DIR, "beemo_semantic_model.zip")
    ensembler_path = os.path.join(MODELS_DIR, "beemo_hybrid_ensembler.joblib")
    
    if not os.path.exists(zip_path):
        print(f"Error: Expected checkpoint zip file '{zip_path}' not found.")
        return
        
    print(f"Extracting {zip_path} to models/beemo_semantic_model...")
    dest_dir = os.path.join(MODELS_DIR, "beemo_semantic_model")
    os.makedirs(dest_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
        
    print("Cleaning up temporary zip file...")
    os.remove(zip_path)
    
    print("Successfully downloaded and extracted Beemo SOTA model assets!")

if __name__ == "__main__":
    main()
