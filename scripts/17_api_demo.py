#!/usr/bin/env python3
"""
Lightweight API demo script.
Starts the FastAPI server locally and tests the /health and /detect endpoints.
"""
import subprocess
import time
import requests
import signal
import sys
import os

API_URL = "http://127.0.0.1:8000"
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SAMPLE_AI = """
The rapid advancement of artificial intelligence has transformed numerous industries, enabling
automation at an unprecedented scale. Machine learning models, particularly large language models,
have demonstrated remarkable capabilities in natural language understanding, generation, and reasoning.
These systems are trained on vast corpora of text data and can produce coherent, contextually relevant
outputs across a wide range of domains. However, their deployment also raises important ethical
considerations regarding bias, misinformation, and accountability. Researchers continue to explore
methods for improving model transparency, robustness, and alignment with human values.
""".strip()

SAMPLE_HUMAN = """
Okay so I was talking to my friend yesterday and we got into this whole thing about whether
AI is actually useful or just overhyped. I mean, some of the stuff is cool, but honestly half
the time it feels like a fancy autocomplete. My friend thinks it'll replace programmers
in like 5 years. I'm way more skeptical. We ended up just agreeing to disagree and getting coffee.
""".strip()


def main():
    print("Starting API server...")
    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_DIR
    env.setdefault('API_KEY', '')  # Disable API-key auth for this local demo
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tool.api:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT_DIR,
        env=env,
    )

    server_ready = False
    try:
        # Wait for server to start (model loading can take 10-30s)
        print("Waiting for API server to start...")
        for _ in range(30):
            try:
                r = requests.get(f"{API_URL}/health", timeout=1)
                if r.status_code == 200:
                    server_ready = True
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)

        if not server_ready:
            raise RuntimeError("API server failed to start within 30 seconds.")

        # Health check
        print("\n--- Health check ---")
        r = requests.get(f"{API_URL}/health")
        print(r.json())

        # Detect sample AI text
        print("\n--- Detect AI text ---")
        r = requests.post(f"{API_URL}/detect", json={"text": SAMPLE_AI})
        print(r.json())

        # Detect sample human text
        print("\n--- Detect human text ---")
        r = requests.post(f"{API_URL}/detect", json={"text": SAMPLE_HUMAN})
        print(r.json())

        # Batch detection
        print("\n--- Batch detection ---")
        r = requests.post(f"{API_URL}/detect/batch", json={"texts": [SAMPLE_AI, SAMPLE_HUMAN]})
        print(r.json())

    finally:
        print("\nStopping API server...")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
        print("Stopped.")


if __name__ == '__main__':
    main()
