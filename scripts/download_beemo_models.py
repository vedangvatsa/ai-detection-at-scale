#!/usr/bin/env python3
import os
import subprocess
import zipfile
import shutil

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
KAGGLE_BIN = os.path.join(PROJECT_DIR, ".venv", "bin", "kaggle")
KERNEL_ID = "vedangvatsa123/ai-detection-sota-pipeline"

def main():
    print(f"Creating models directory at {MODELS_DIR} if it does not exist...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Downloading outputs from Kaggle kernel: {KERNEL_ID}...")
    cmd = [
        KAGGLE_BIN, "kernels", "output",
        KERNEL_ID, "-p", MODELS_DIR
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error downloading outputs from Kaggle:")
        print(result.stderr)
        return
    
    print("Download complete. Checking downloaded files...")
    onnx_path = os.path.join(MODELS_DIR, "deberta_onnx_quantized.onnx")
    ensemblers_path = os.path.join(MODELS_DIR, "beemo_register_ensemblers.joblib")
    
    if not os.path.exists(onnx_path):
        print(f"Error: Expected quantized ONNX model '{onnx_path}' not found.")
        return
    if not os.path.exists(ensemblers_path):
        print(f"Error: Expected ensemblers joblib '{ensemblers_path}' not found.")
        return
        
    print("Successfully downloaded SOTA model assets (ONNX model + register-specific ensemblers)!")

if __name__ == "__main__":
    main()
