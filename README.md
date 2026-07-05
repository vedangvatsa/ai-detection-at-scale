# AI Detection at Scale

Cross-register stylometric detection of AI-generated text using 31 interpretable features. Evaluated on 2.77M texts across four registers and twelve model variants from the [RAID benchmark](https://github.com/liamdugan/raid).

This repository has been expanded into a production-hardened hybrid detector incorporating **stylometric ensembling**, **multi-class model source attribution**, and **local neural observer diagnostics** (surprisal analytics).

---

## Benchmark Results

### 1. Stylometric RF Baselines
| Setting / Dataset | Metric | Value |
|-------------------|--------|-------|
| Within-register   | AUC    | 0.933–0.978 |
| Cross-domain      | AUC    | 0.728 |
| Adversarial (Paraphrase) | AUC | 0.951 |
| GPT-4 detection   | AUC    | 0.983 |
| Throughput        | Speed  | 100 texts/sec (CPU, no GPU) |

### 2. External Kaggle Benchmarks (V3/V4 Runs)
* **Beemo V3 (`toloka/beemo`):** Evaluates expert-humanized/edited machine text.
  * **Result:** **`0.5256` AUC**
  * *Analysis:* Reflects the boundaries of pure stylometry on post-edited text, showing that human refinement breaks standard structural fingerprints (consistent with Artemova et al., 2025).
* **RAID V3/V4 (`liamdugan/raid` test-sampling):**
  * *Status:* Actively running in the background.

---

## Production Features & API Suite

### 1. Local Neural Observer Diagnostics
* Implemented in [neural_detector.py](file:///Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/tool/neural_detector.py).
* Evaluates token-level **Perplexity (PPL)** and **Burstiness** (surprisal standard deviation) using GPT-2 Small. 
* Utilizes lazy loading to prevent overhead on API startup.

### 2. Stylometric Model Attribution
* Trained in [train_attribution.py](file:///Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/scripts/train_attribution.py) and integrated in [attribution.py](file:///Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/tool/attribution.py).
* A multi-class Logistic Regression classifier trained on 210,000 balanced RAID features to predict the specific generative source group:
  * `OpenAI` (GPT-4 / ChatGPT)
  * `Meta Llama` (Llama-2)
  * `Mistral` (Mistral-7B / Mixtral)
  * `Cohere` (Command)
  * `MPT` (MPT-30B)
  * `Human` (Original Text)

### 3. Sentence-level Heatmap Analysis
* Implemented in [sentence_analyzer.py](file:///Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/tool/sentence_analyzer.py).
* Utilizes a 3-sentence sliding window to evaluate localized probability scores, returning character offsets (`start`, `end`) to allow visual formatting on frontend clients.

### 4. Production API Hardening
* Located in [api.py](file:///Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/tool/api.py).
* **CORS Support:** Enabled for web client requests.
* **Token-Bucket Rate Limiter:** Capped at 60 requests per minute per IP.
* **Response Caching:** Stores MD5 signatures of inputs to bypass redundant inference runs.

---

## Quickstart

```bash
git clone https://github.com/vedangvatsa/ai-detection-at-scale.git
cd ai-detection-at-scale

# Set up environment
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn joblib tqdm transformers torch

# Download pre-built data (3.5 GB) and models (4 GB)
python scripts/download_assets.py

# Start the inference API
uvicorn tool.api:app --host 0.0.0.0 --port 8000
```

### Try the endpoint:
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Furthermore, the results clearly demonstrate that this approach is effective.", "return_features": true}'
```

---

## Project Layout

```text
├── tool/                    Inference API + Feature Extractors
│   ├── api.py               Production API with Caching, Rate Limiting & CORS
│   ├── feature_extractor.py 31-feature extraction
│   ├── neural_detector.py   Perplexity & Burstiness observer (GPT-2)
│   ├── attribution.py       RAID multi-class source classifier
│   ├── sentence_analyzer.py Sliding-window sentence highlights
│   └── multilingual.py      5-language word lists
├── scripts/                 Pipeline & Analysis Scripts (01–14)
│   └── train_attribution.py Attribution model training script
├── results/                 CSV outputs + generated figures
├── models/                  Trained classifier joblibs
└── data/                    Corpus datasets
```

---

## Citation

```bibtex
@misc{ai_detection_at_scale,
  title={AI Detection at Scale: Cross-Register Stylometric Detection of AI-Generated Text},
  author={Vatsa, Vedang},
  year={2026},
  url={https://github.com/vedangvatsa/ai-detection-at-scale}
}
```
