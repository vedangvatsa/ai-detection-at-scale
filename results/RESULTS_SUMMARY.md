# AI Detection at Scale — Results Summary

## Model Overview

- **Primary model**: 31-feature stylometric RandomForest classifier
- **Architecture**: Register-aware ensemble (academic, news, social, creative) with fallback all-register detector
- **Feature set**: extended 31 stylometric features (see `tool/feature_extractor.py`)
- **Corpus**: 2.77M texts across 4 registers; 1.85M AI, 0.92M human
- **Code & assets**: [github.com/vedangvatsa/ai-detection-at-scale](https://github.com/vedangvatsa/ai-detection-at-scale)

## Overall Performance

| Metric | 11-feature | 31-feature |
|--------|------------|------------|
| AUC    | 0.9645     | **0.9826** |
| Accuracy | 0.9011   | **0.9361** |
| F1     | 0.8710     | **0.9168** |

The 31-feature model improves AUC by **+0.018** over the 11-feature baseline.

## Low False-Positive Rate Performance (all registers, 200k texts)

| FPR   | Threshold | TPR    |
|-------|-----------|--------|
| 0.01% | 0.995     | 0.134  |
| 0.1%  | 0.955     | 0.305  |
| 0.5%  | 0.865     | 0.478  |
| 1%    | 0.805     | 0.563  |
| 5%    | 0.580     | 0.779  |

Per-register numbers are in `results/tpr_at_low_fpr.csv`.

## Humanization Robustness (adversarial attacks)

| Attack strategy | AUC    | AUC drop |
|-----------------|--------|----------|
| clean baseline  | 0.9417 | —        |
| remove connectors | 0.9340 | -0.0076 |
| synonym swap    | 0.9403 | -0.0014 |
| vary length     | 0.9400 | -0.0017 |
| punctuation     | 0.9387 | -0.0030 |
| first person    | 0.9398 | -0.0019 |
| **combined**    | 0.9324 | **-0.0093** |

The combined humanization attack reduces AUC by less than 1 percentage point, showing strong robustness.

## Per-Register Error Rates (31-feature model)

| Register  | AUC   | FP rate | FN rate |
|-------------|-------|---------|---------|
| academic    | 1.000 | 0.010   | 0.000   |
| news        | 0.999 | 0.002   | 0.038   |
| social      | 0.982 | 0.027   | 0.134   |
| creative    | 0.982 | 0.055   | 0.061   |

Social and creative are the hardest registers, with higher false-negative rates (AI misclassified as human).

## Most Important Features (permutation importance, mean across registers)

1. `sent_cv` — sentence count / length variation
2. `mtld` — lexical diversity
3. `char_entropy` — character-level entropy
4. `rep_rate` — repetition rate
5. `quote_density` — quotation density
6. `question_density` — question density
7. `capitalized_entity_density` — named entity density
8. `punct_entropy` — punctuation entropy
9. `smog_index` — readability
10. `self_mention_density` — first-person mentions

Full per-register breakdown is in `results/per_register_feature_importance.csv`.

## Multi-Signal Ensemble

A meta-ensemble combining all-register and per-register stylometric signals was trained:

| Method | AUC | # signals |
|--------|-----|-----------|
| stylo_all | 1.000 | 1 |
| stylo_reg | 1.000 | 1 |
| stylo_31  | 0.999999934 | 1 |
| meta_ensemble | 1.000 | 2 |

Note: on the balanced evaluation sample, the existing signals already separate AI and human perfectly; the ensemble is a lightweight meta-classifier ready to incorporate additional signals (e.g., Binoculars, RAID-style benchmarks) as they become available.

## Neural Baseline Comparison

Direct comparison with **Binoculars** (an industry-leading neural detector) was attempted on Kaggle GPU, but the run was blocked by Kaggle DNS/internet issues. The notebook is at:

[Kaggle notebook](https://www.kaggle.com/code/vedangvatsa123/ai-detection-binoculars-benchmark)

The notebook can be rerun once Kaggle internet is fully functional or on a GPU machine with bandwidth. It uses `gpt2` (fast test) or `tiiuae/falcon-7b` (real comparison).

## API Demonstration

The FastAPI inference endpoint (`tool/api.py`) was verified locally:

- `/health` returns all loaded models and registers
- `/detect` returns AI probability, predicted register, and confidence
- `/detect/batch` handles multiple texts

Example:
```text
AI sample:    probability=0.96, register=creative, is_ai=True
Human sample: probability=0.185, register=social, is_ai=False
```

See `scripts/17_api_demo.py` for a runnable client demo.
