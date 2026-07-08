# AI Detection at Scale

Cross-register stylometric detection of AI-generated text using 31 interpretable features. Evaluated on 2.77M texts across four registers and twelve model variants from the [RAID benchmark](https://github.com/liamdugan/raid).

This repository has been expanded into a production-hardened hybrid detector incorporating **stylometric ensembling**, **multi-class model source attribution**, and **local neural observer diagnostics** (surprisal analytics).

---

## Benchmark Results

### 1. Stylometric RF Baselines
| Setting / Dataset | Metric | Value | Notes |
|---|---|---|---|
| Within-register (5-fold CV, unmatched) | AUC | 0.933–0.978 | See Table 4 in paper. Academic/news AUC is inflated by the document-length confound described in Section 3.2; the cross-domain matrix diagonal reaches 1.000 for the same reason. |
| Within-register (length-matched) | AUC | 0.967–0.978 | Mean 0.970; length confound removed. See Table 9 in paper. |
| Cross-domain | AUC | 0.728 | Mean off-diagonal transfer AUC. |
| Adversarial (Paraphrase) | AUC | 0.951 | Reproduced by `scripts/06_adversarial_eval.py`; results saved in `results/adversarial_results.csv` (paraphrase row AUC = 0.951). |
| GPT-4 detection | AUC | 0.983 | Per-model evaluation in paper. |
| Throughput | Speed | 100 texts/sec | CPU, no GPU. |

### 2. Multi-Benchmark Evaluation Results

We evaluated our models against 4 major AI detection benchmarks, obtaining the following results:

| Benchmark | Model Configuration | Target Evaluation | Metric | Status |
|---|---|---|---|---|
| **Toloka Beemo** | Hybrid SOTA (BERT-mini + GPT-2 + Stylometrics) | Expert-humanized/edited machine text | **AUC: 0.7616** | ✅ Complete |
| **RAID Benchmark** | Stylometrics only (adversarial baseline) | Paraphrasing/Adversarial attacks (11 LLMs) | **AUC: 0.7360** | ✅ Complete |
| **MAGE Benchmark** | Stylometrics only (ChatGPT vs Human) | ChatGPT generation vs human | **AUC: 0.6955** | ✅ Complete |
| **TuringBench** | Stylometrics only (19 old/new generators) | Multiclass attribution / Turing test | **AUC: 0.4691** | ✅ Complete |

### 3. Public Detector Results (New)

Per-benchmark specialized detector selection, 2000 samples, 512 tokens (1024 for MAGE):

| Benchmark | Best Detector / Ensemble | AUC | Accuracy |
|---|---|---|---|
| **MAGE** | roberta-base-openai + stylometric (1024 tok) | **0.7801** | — |
| **HC3** | Hello-SimpleAI/chatgpt-detector-roberta (512 tok) | **0.9997** | 0.9940 |
| **TuringBench** | roberta-large-openai-detector (512 tok) | **0.9146** | 0.6665 |

* **TuringBench** improves from **0.4691** → **0.9146** by switching to `roberta-large-openai-detector`.
* **HC3** reaches near ceiling with the `chatgpt-detector-roberta` public model.
* **MAGE** benefits from combining public detector probability with stylometric features.
* **Public Detector API:** `tool/public_api.py` exposes `/detect/public`, `/detect/public/batch`, and `/public/detectors` endpoints for these models.
* **Beemo Hybrid SOTA:** Fine-tuning a semantic BERT-mini model and ensembling its logits with GPT-2 perplexity and stylometrics raised our Toloka Beemo performance from **`0.5256`** to **`0.7616` AUC**, highlighting the power of hybrid ensembling on expert-edited machine text.
* **Zero-Bandwidth Local Fallback:** The local neural observer (`tool/neural_detector.py`) attempts to load GPT-2 from the local HuggingFace cache first. If not cached, it runs offline using the stylometrics fallback directly, preventing heavy automatic internet downloads.

### 4. SOTA Model Results (New)

The MAGE paper's Longformer detector (`nealcly/detection-longformer`) evaluated at 512 tokens:

| Benchmark | Model | Tokens | AUC | Accuracy | Samples |
|---|---|---|---|---|---|
| **MAGE** | `nealcly/detection-longformer` | 512 | **0.9796** | 0.8940 | 2000 |
| **TuringBench** | `nealcly/detection-longformer` | 512 | 0.6729 | 0.6645 | 2000 |

* **MAGE** jumps from **0.7801** (public detector ensemble) to **0.9796** with the MAGE Longformer at 2000 samples, matching published SOTA.
* **TuringBench** remains best with `roberta-large-openai-detector` (**0.9146**); the MAGE Longformer does not generalize across generators.

### 5. TuringBench Fine-Tuned RoBERTa-large (New)

A `roberta-large` model was fine-tuned on the full TuringBench 19-generator training split (331k texts) on Kaggle and evaluated on the validation split:

| Model | Training data | Max length | Epochs | Batch size | Validation AUC | Validation Accuracy |
|---|---|---|---|---|---|---|
| **Fine-tuned roberta-large** | TuringBench train (full) | 256 | 1 | 48 | **0.9991** | **0.9948** |
| `roberta-large-openai-detector` (zero-shot) | — | 512 | — | — | 0.9146 | 0.6665 |

* Fine-tuning improves TuringBench validation AUC from **0.9146** to **0.9991** (+0.0845) and accuracy from **0.6665** to **0.9948**.
* The fine-tuned model weights are saved on Kaggle; the local copy is at `models/turingbench_roberta_large/` (1.3 GB, git-ignored). Kaggle notebook: `notebooks/kaggle_turingbench_finetune/kaggle_turingbench_finetune.ipynb`.
* Results are saved in `results/turingbench_finetuned.csv`.

### 6. DeBERTa-v3-large Fine-Tuning & Ensemble (In Progress)

A `microsoft/deberta-v3-large` model is being fine-tuned on the full TuringBench training split with 512 tokens, 3 epochs, batch 8, and gradient accumulation 8 (effective batch 64) on Kaggle. Earlier attempts on Kaggle P100 failed due to FP16 incompatibility, so the current run uses FP32.

| Artifact | Location |
|---|---|
| Fine-tuning script | `scripts/33_finetune_turingbench.py` |
| Kaggle fine-tuning notebook | `notebooks/kaggle_turingbench_finetune/kaggle_turingbench_finetune.ipynb` |
| Ensemble training script | `scripts/ensemble_turingbench.py` |
| Kaggle ensemble notebook | `notebooks/kaggle_turingbench_ensemble/kaggle_turingbench_ensemble.ipynb` |
| Run monitor | `scripts/monitor_kaggle_deberta.py` |

The ensemble script combines multiple fine-tuned models with a logistic regression over their AI-class probabilities.

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
