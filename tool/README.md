# AI Text Detection — Tool & API

## Inference API

FastAPI REST endpoint wrapping the pre-trained joblib models.

### Quick start

```bash
pip install -r tool/requirements.txt
uvicorn tool.api:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/detect` | Detect single text |
| `POST` | `/detect/batch` | Detect up to 1000 texts |
| `POST` | `/detect/public` | Use a public HuggingFace detector |
| `POST` | `/detect/public/batch` | Batch public detector inference |
| `GET` | `/public/detectors` | List available public detectors |
| `GET` | `/health` | Health check + loaded models |
| `GET` | `/models` | List loaded models |
| `GET` | `/features` | List feature names |

### Public detector server

Use `tool/public_api.py` to start the API with public detector support:

```bash
uvicorn tool.public_api:app --host 0.0.0.0 --port 8000
```

### Example

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Furthermore, the results demonstrate that this approach is clearly effective.", "return_features": true}'
```

```json
{
  "ai_probability": 0.82,
  "register": "academic",
  "register_confidence": 0.91,
  "is_ai": true,
  "features": {"mtld": 45.2, "sent_cv": 0.12, ...},
  "processing_time_ms": 3.4
}
```

## Feature Extractor (`tool/feature_extractor.py`)

Extracts 31 stylometric features from text:

- **Original 11**: MTLD, sentence CV, mean sentence length, self-mention density, connector density, opener ratio, hedge density, booster density, char n-gram entropy, word repetition rate, punctuation entropy
- **Extended 20**: Flesch reading ease, Flesch-Kincaid grade, Gunning fog, SMOG index, positive/negative sentiment density, sentiment polarity, exclamation/question density, passive voice density, subordination density, preposition/adjective/adverb/nominalization density, capitalized entity/number/acronym/URL-email/quote density

```python
from tool.feature_extractor import extract_features
feats = extract_features("Some text to analyze...", extended=True)
```

## Multilingual Support (`tool/multilingual.py`)

Translated word lists for 5 languages: English, Spanish, French, German, Chinese.

```python
from tool.multilingual import get_word_lists, detect_language
lang = detect_language(text)
word_lists = get_word_lists(lang)
```

## Analysis Scripts

| Script | Description |
|--------|-------------|
| `scripts/10_tpr_at_low_fpr.py` | TPR at 0.01% and 0.1% FPR |
| `scripts/11_extract_extended_features.py` | Extract 31 features, train & compare |
| `scripts/12_humanized_eval.py` | Evaluate on humanized AI text |
| `scripts/13_binoculars_baseline.py` | Run Binoculars on same corpus |
| `scripts/14_multi_signal_ensemble.py` | Stylometric + neural meta-ensemble |
| `scripts/27_roberta_detector_benchmark.py` | Public RoBERTa OpenAI detector benchmark |
| `scripts/28_optimize_roberta_stylometric.py` | RoBERTa + stylometric logistic ensemble |
| `scripts/29_compare_public_detectors.py` | Compare public detectors on benchmarks |
| `scripts/30_advanced_ensemble.py` | 3 public detectors + stylometric ensemble |
| `scripts/31_per_benchmark_optimized.py` | Per-benchmark specialized detector selection |
| `tool/public_detector.py` | Wrapper for public HuggingFace detectors |
| `tool/public_api.py` | FastAPI app with public detector endpoints |
