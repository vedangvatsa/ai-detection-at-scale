# AI Detection at Scale

Cross-register stylometric AI text detection using 31 interpretable features, evaluated on 2.77 million texts across four registers (academic, news, social, creative) and twelve generative model variants from the [RAID benchmark](https://github.com/liamdugan/raid).

## Key Results

- **Within-register AUC**: 0.933–0.978 (Random Forest, 5-fold CV)
- **Cross-domain AUC**: 0.728 (mean off-diagonal)
- **Adversarial robustness**: AUC 0.951 under paraphrase attacks
- **GPT-4 detection**: AUC 0.983
- **Throughput**: 100 texts/sec on single CPU core (no GPU required)

## Features

**Original 11 stylometric features**: MTLD, sentence length CV, mean sentence length, self-mention density, connector density, sentence-opener ratio, hedge density, booster density, character n-gram entropy, word repetition rate, punctuation entropy.

**Extended 20 features**: Flesch reading ease, Flesch-Kincaid grade, Gunning fog, SMOG index, positive/negative sentiment density, sentiment polarity, exclamation/question density, passive voice density, subordination density, preposition/adjective/adverb/nominalization density, capitalized entity/number/acronym/URL-email/quote density.

## Project Structure

```
ai-detection-at-scale/
├── ai_detection_at_scale.md    # Full paper
├── ai_detection_at_scale.pdf   # Compiled PDF
├── tool/                       # Inference API + shared modules
│   ├── api.py                  # FastAPI REST endpoint
│   ├── feature_extractor.py    # 31-feature extraction
│   ├── multilingual.py         # 5-language word lists
│   └── requirements.txt
├── scripts/                    # Analysis pipeline
│   ├── 01_fetch_data.py        # Download RAID + human corpora
│   ├── 02_extract_features.py  # Extract 11 features
│   ├── 03_analyze.py           # Cohen's d, sign analysis
│   ├── 04_generate_figures.py  # Paper figures
│   ├── 05_mustdo_analyses.py   # Length-matched, per-model
│   ├── 06_adversarial_eval.py  # 8 attack types
│   ├── 09_ensemble_and_defense.py  # Register-aware ensemble
│   ├── 10_tpr_at_low_fpr.py    # TPR at 0.01%/0.1% FPR
│   ├── 11_extract_extended_features.py  # 31-feature extraction
│   ├── 12_humanized_eval.py    # Humanized AI text evaluation
│   ├── 13_binoculars_baseline.py  # Binoculars direct comparison
│   └── 14_multi_signal_ensemble.py  # Stylometric + neural ensemble
├── results/                    # CSV outputs + figures
├── models/                     # Trained joblib models (gitignored)
└── data/                       # Raw + feature parquets (gitignored)
```

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn joblib tqdm
pip install fastapi uvicorn pydantic  # for API
```

### 2. Download data & extract features

```bash
python scripts/01_fetch_data.py       # Download RAID + human texts
python scripts/02_extract_features.py # Extract 11 features
```

### 3. Train models

```bash
python scripts/09_ensemble_and_defense.py  # Train register classifier + per-register detectors
```

### 4. Run analyses

```bash
python scripts/03_analyze.py              # Effect sizes, sign analysis
python scripts/05_mustdo_analyses.py      # Length-matched, per-model, deployment
python scripts/06_adversarial_eval.py     # Adversarial robustness
python scripts/10_tpr_at_low_fpr.py       # TPR at ultra-low FPR
python scripts/11_extract_extended_features.py  # 31-feature comparison
python scripts/12_humanized_eval.py       # Humanized AI evaluation
```

### 5. Start inference API

```bash
pip install -r tool/requirements.txt
uvicorn tool.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Furthermore, the results demonstrate clearly that this approach is effective.", "return_features": true}'
```

## Datasets

- **AI texts**: [RAID benchmark](https://huggingface.co/datasets/liamdugan/raid) (2.77M texts, 12 model variants)
- **Human texts**: Multi-source human corpora across academic, news, social, and creative registers

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
