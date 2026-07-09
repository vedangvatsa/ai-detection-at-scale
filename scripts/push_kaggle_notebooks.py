#!/usr/bin/env python3
import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

def load_env():
    env_path = PROJECT_DIR / '.env'
    env = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env

def main():
    env = load_env()
    
    # Extract credentials
    hf_token = env.get('HF_TOKEN')
    kaggle_user = env.get('KAGGLE_USERNAME')
    # Support both key names
    kaggle_key = env.get('KAGGLE_KEY') or env.get('KAGGLE_API_TOKEN')
    
    if not hf_token:
        print("Error: HF_TOKEN not found in .env")
        return
    if not kaggle_user or not kaggle_key:
        print("Error: Kaggle credentials not found in .env")
        return
        
    print("Found credentials in .env.")
    
    # Configure Kaggle environment variables for subprocesses
    os.environ['KAGGLE_USERNAME'] = kaggle_user
    os.environ['KAGGLE_KEY'] = kaggle_key
    
    # Activate virtual environment if it exists
    venv_bin = PROJECT_DIR / '.venv' / 'bin' / 'kaggle'
    kaggle_cmd = str(venv_bin) if venv_bin.exists() else 'kaggle'
    
    # Notebook definitions
    notebooks = [
        {
            "name": "roberta-large",
            "notebook_path": PROJECT_DIR / "notebooks" / "kaggle_turingbench_finetune.ipynb",
            "metadata_path": PROJECT_DIR / "notebooks" / "kaggle_turingbench_finetune" / "kernel-metadata.json",
            "dest_filename": "kaggle_turingbench_finetune.ipynb"
        },
        {
            "name": "deberta-v3-large",
            "notebook_path": PROJECT_DIR / "notebooks" / "kaggle_turingbench_deberta_v3_large.ipynb",
            "metadata_path": PROJECT_DIR / "notebooks" / "kernel-metadata-turingbench-deberta.json",
            "dest_filename": "kaggle_turingbench_deberta_v3_large.ipynb"
        }
    ]
    
    for run in notebooks:
        print(f"\nProcessing {run['name']}...")
        
        # 1. Read the clean notebook
        with open(run['notebook_path'], 'r') as f:
            data = json.load(f)
            
        # 2. Inject token into cell containing kaggle_secrets/HF_TOKEN setup
        injected = False
        for cell in data['cells']:
            if cell['cell_type'] == 'code' and any('kaggle_secrets' in line for line in cell['source']):
                new_source = []
                for line in cell['source']:
                    if 'Set HF_TOKEN manually' in line:
                        new_source.append(f"    print('Using fallback HF_TOKEN injected from deployment script.')\n")
                        new_source.append(f"    os.environ['HF_TOKEN'] = '{hf_token}'\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                injected = True
                break
                
        if not injected:
            print(f"Warning: Could not find token loading cell in {run['notebook_path']}")
            continue
            
        # 3. Create temporary directory to push from
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir_path = Path(tmpdir)
            
            # Copy modified notebook
            tmp_notebook_path = tmp_dir_path / run['dest_filename']
            with open(tmp_notebook_path, 'w') as f:
                json.dump(data, f, indent=1)
                
            # Copy metadata file (making sure code_file points to the correct location/name)
            with open(run['metadata_path'], 'r') as f:
                meta = json.load(f)
            meta['code_file'] = run['dest_filename']
            
            tmp_meta_path = tmp_dir_path / "kernel-metadata.json"
            with open(tmp_meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
                
            print(f"Pushed temporary setup to {tmp_dir_path}. Launching Kaggle push...")
            
            # 4. Execute kaggle kernels push
            res = subprocess.run(
                [kaggle_cmd, "kernels", "push"],
                cwd=str(tmp_dir_path),
                capture_output=True,
                text=True
            )
            
            if res.returncode == 0:
                print(f"Success: {res.stdout.strip()}")
            else:
                print(f"Error pushing kernel: {res.stderr.strip()}")
                
if __name__ == '__main__':
    main()
