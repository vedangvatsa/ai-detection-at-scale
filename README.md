# AI Detection at Scale

Cross-register stylometric detection of AI-generated text using 31 interpretable features. Evaluated on 2.77M texts across four registers and twelve model variants from the [RAID benchmark](https://github.com/liamdugan/raid).

## Results

| Metric | Value |
|--------|-------|
| Within-register AUC | 0.933–0.978 |
| Cross-domain AUC | 0.728 |
| Adversarial AUC (paraphrase) | 0.951 |
| GPT-4 detection AUC | 0.983 |
| Throughput | 100 texts/sec (CPU, no GPU) |

## Quickstart

```bash
git clone https://github.com/vedangvatsa/ai-detection-at-scale.git
cd ai-detection-at-scale

# Set up environment
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn joblib tqdm
pip install fastapi uvicorn pydantic  # for inference API

# Download pre-built data (3.5 GB) and models (4 GB)
python scripts/download_assets.py

# Start the inference API
uvicorn tool.api:app --host 0.0.0.0 --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Furthermore, the results clearly demonstrate that this approach is effective.", "return_features": true}'
```

## Data & Models

Pre-built assets are distributed via [GitHub Releases](https://github.com/vedangvatsa/ai-detection-at-scale/releases/tag/v1.0-data) (7.5 GB total). The download script fetches and reassembles them automatically.

| Asset | Size | Description |
|-------|------|-------------|
| `corpus_raw.parquet` | 2.1 GB | 2.77M texts (human + AI, 4 registers, 12 models) |
| `corpus_features.parquet` | 104 MB | 11 stylometric features extracted |
| `human_*.parquet` | 1.3 GB | Source human corpora by register |
| `register_classifier.joblib` | 1.9 GB | 4-way register classifier |
| `detector_all.joblib` | 1.2 GB | All-register RF detector |
| `detector_{register}.joblib` | 827 MB | Per-register RF detectors (×4) |

To rebuild from scratch instead of downloading, see the [pipeline scripts](#pipeline) below.

## Features

**11 original stylometric features**: MTLD, sentence length CV, mean sentence length, self-mention density, connector density, sentence-opener ratio, hedge density, booster density, character n-gram entropy, word repetition rate, punctuation entropy.

**20 extended features**: Flesch reading ease, Flesch-Kincaid grade, Gunning fog, SMOG, positive/negative sentiment density, sentiment polarity, exclamation/question density, passive voice, subordination, preposition/adjective/adverb/nominalization density, capitalized entity, number, acronym, URL-email, quote density.

Multilingual word lists available for English, Spanish, French, German, and Chinese (`tool/multilingual.py`).

## <a name="pipeline"></a>Rebuilding from Scratch

```bash
# 1. Download source data from HuggingFace
python scripts/01_fetch_data.py

# 2. Extract features
python scripts/02_extract_features.py        # 11 features
python scripts/11_extract_extended_features.py  # 31 features + comparison

# 3. Train models
python scripts/09_ensemble_and_defense.py    # Register-aware ensemble

# 4. Run analyses
python scripts/03_analyze.py                 # Effect sizes, sign analysis
python scripts/05_mustdo_analyses.py         # Length-matched, per-model, deployment
python scripts/06_adversarial_eval.py        # 8 attack types
python scripts/10_tpr_at_low_fpr.py          # TPR at 0.01%/0.1% FPR
python scripts/12_humanized_eval.py          # Humanized AI evaluation
python scripts/13_binoculars_baseline.py     # Binoculars direct comparison
python scripts/14_multi_signal_ensemble.py   # Stylometric + neural ensemble
```

## Project Layout

```text
├── tool/                  Inference API + feature extraction
│   ├── api.py             FastAPI endpoint (/detect, /detect/batch)
│   ├── feature_extractor.py   31-feature extraction
│   └── multilingual.py    5-language word lists
├── scripts/               Analysis pipeline (01–14)
├── results/               CSV outputs + figures
├── models/                Trained models (downloaded via release)
├── data/                  Corpus data (downloaded via release)
└── ai_detection_at_scale.md   Full paper
```

## License

MIT

## Citation

```bibtex
@misc{ai_detection_at_scale,
  title={AI Detection at Scale: Cross-Register Stylometric Detection of AI-Generated Text},
  author={Vatsa, Vedang},
  year={2026},
  url={https://github.com/vedangvatsa/ai-detection-at-scale}
}
```
