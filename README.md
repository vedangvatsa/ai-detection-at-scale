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
| GPT-4 detection | AUC | 0.983 | From per-model evaluation (Table 11 / `results/per_model_auc.csv`), reproduced by `scripts/05_mustdo_analyses.py`. |
| Throughput | Speed | 100 texts/sec | CPU, no GPU. |

### 2. Multi-Benchmark Evaluation Results

We evaluated our models against 4 major AI detection benchmarks, obtaining the following results:

| Benchmark | Model Configuration | Target Evaluation | Metric | Status |
|---|---|---|---|---|
| **Toloka Beemo** | Hybrid ensemble (BERT-mini + GPT-2 + stylometrics) | Expert-humanized/edited machine text | **AUC: 0.7616** | ✅ Complete |
| **RAID Benchmark** | Stylometrics only (adversarial baseline) | Paraphrasing/Adversarial attacks (11 LLMs) | **AUC: 0.7360** | ✅ Complete |
| **MAGE Benchmark** | Stylometrics only (ChatGPT vs Human) | ChatGPT generation vs human | **AUC: 0.6955** | ✅ Complete |
| **TuringBench** | Stylometrics only (19 old/new generators) | Multiclass attribution / Turing test | **AUC: 0.4691** | ✅ Complete |

### 3. Public Detector Results (New)

Per-benchmark evaluation of publicly available HuggingFace detectors, 2000 samples, 512 tokens (1024 for MAGE):

| Benchmark | Best Public Detector / Ensemble | AUC | Accuracy |
|---|---|---|---|
| **MAGE** | `roberta-base-openai-detector` + stylometric (1024 tok) | **0.7801** | — |
| **HC3** | `Hello-SimpleAI/chatgpt-detector-roberta` (512 tok) | **0.9997** | 0.9940 |
| **TuringBench** | `roberta-large-openai-detector` (512 tok) | **0.9146** | 0.6665 |

* **TuringBench** improves from **0.4691** → **0.9146** by switching to `roberta-large-openai-detector`.
* **HC3** reaches near ceiling with the `chatgpt-detector-roberta` public model.
* **MAGE** benefits from combining public detector probability with stylometric features.
* **Public Detector API:** `tool/public_api.py` exposes `/detect/public`, `/detect/public/batch`, and `/public/detectors` endpoints for these models.
* **Beemo Hybrid Ensemble:** Fine-tuning a semantic BERT-mini model and ensembling its logits with GPT-2 perplexity and stylometrics raised our Toloka Beemo performance from **`0.5256`** to **`0.7616` AUC**. This is an ensemble improvement over our own stylometric baseline, not a claim of state-of-the-art over all published methods.
* **Local Neural Observer:** `tool/neural_detector.py` computes perplexity and burstiness from a locally cached GPT-2 model. It requires the model to be pre-downloaded; otherwise it raises a `RuntimeError`. The API currently calls it on every request, so production deployments must cache GPT-2 beforehand.

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

### 2. Stylometric Model-Source Group Attribution
* Trained in [train_attribution.py](file:///Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/scripts/train_attribution.py) and integrated in [attribution.py](file:///Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/tool/attribution.py).
* A multi-class Logistic Regression classifier trained on up to 30,000 samples per source group (~210,000 total) from `data/corpus_features.parquet` to predict the generative source **group** (not the individual model):
  * `OpenAI` (GPT-2 / GPT-3 / GPT-4 / ChatGPT grouped together)
  * `Meta Llama`
  * `Mistral`
  * `Cohere`
  * `MPT`
  * `Human`
* The current training script does not report a held-out accuracy or confusion matrix. Run it to regenerate `models/attribution_classifier.joblib`.

### 3. Sentence-level Heatmap Analysis
* Implemented in [sentence_analyzer.py](file:///Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/tool/sentence_analyzer.py).
* Utilizes a 3-sentence sliding window to evaluate localized probability scores, returning character offsets (`start`, `end`) to allow visual formatting on frontend clients.

### 4. API Hardening (Development-Focused)
* Located in [api.py](file:///Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/tool/api.py).
* **CORS Support:** Enabled with `allow_origins=["*"]` for frontend integration. Set `CORS_ORIGINS` env var to restrict this in production.
* **Rate Limiting:** Token-bucket limit of 60 requests per minute per IP on `/detect`. The `/detect/batch` endpoint currently does not apply IP-level rate limiting.
* **Response Caching:** In-memory MD5-keyed cache (no TTL; entries evicted only when the 1000-entry limit is reached).
* **Known Production Gaps:** No API-key authentication, in-memory rate-limit state is not shared across workers, and GPT-2 must be pre-cached for neural signals to avoid `RuntimeError`.

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

## Known Limitations & Audit Notes

The following limitations have been identified and either fixed or explicitly documented. This list is meant to help reviewers and downstream users interpret the results correctly:

* **Within-register AUC is inflated by document length.** The paper acknowledges this confound (Section 3.2); the headline 0.933–0.978 range mixes length-matched and unmatched evaluations. Use the **cross-domain AUC 0.728** as the most trustworthy generalization metric.
* **GPT-4 detection AUC 0.983** is from the per-model breakdown in `results/per_model_auc.csv` (reproduced by `scripts/05_mustdo_analyses.py`), not a dedicated GPT-4-only benchmark.
* **Public detector scores** (HC3 0.9997, TuringBench 0.9146, MAGE 0.7801) are evaluations of existing HuggingFace models (`Hello-SimpleAI/chatgpt-detector-roberta`, `roberta-large-openai-detector`, etc.), not detectors we trained.
* **Model-source attribution** predicts model **groups** (OpenAI, Llama, Mistral, Cohere, MPT, Human), not individual models. `scripts/train_attribution.py` now reports held-out accuracy, a classification report, and a confusion matrix.
* **Calibration** now supports a trained Platt/isotonic calibrator (`scripts/train_calibration.py`). If no trained model is present, `tool/calibration.py` falls back to the original length-based heuristic.
* **Adversarial defense** in `tool/adversarial_defense.py` is a **character-level preprocessor** (homoglyph normalization, zero-width/control-char stripping, whitespace/punctuation cleanup). It does **not** defend against paraphrase, prompt injection, synonym substitution, or back-translation.
* **API production gaps** are partially closed: CORS is configurable via `CORS_ORIGINS`, cache has a TTL (`CACHE_TTL_SECONDS`), both `/detect` and `/detect/batch` now check API keys (`API_KEY`) and IP rate limits, and neural signal failures are caught with a safe fallback. Rate-limit state is now bounded by `RATE_LIMIT_MAX_IPS` with stale-entry eviction; the cache remains in-memory, so a Redis-backed deployment is still recommended for multiple workers.
* **Feature quality:** 11 of the 31 features are standard stylometric metrics; the remaining 20 are heuristic keyword-density and suffix-based counts. A duplicate `'flawed'` in the `NEGATIVE_WORDS` list was removed.
* **Test coverage** is minimal. `tests/test_adversarial_defense.py` verifies the defense preprocessor; broader unit tests should be added before deployment.

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
