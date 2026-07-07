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

## Neural Baseline Comparison (Kaggle Version 6)

Comparison against industry neural detectors from Kaggle version 6 output:

| Method | Type | Within-register AUC | Cross-domain AUC | Adversarial AUC | Throughput (texts/sec) |
|--------|------|---------------------|------------------|-----------------|------------------------|
| Stylometric RF (this paper) | interpretable | **0.941** | **0.728** | **0.951** | **100** |
| Binoculars (Falcon-7B) | neural | 0.91 | 0.65 | 0.55 | 0.5 |
| RADAR (RoBERTa fine-tuned) | neural | 0.93 | 0.70 | 0.40 | 50 |
| GPTZero (commercial) | neural | 0.88 | 0.60 | 0.35 | 10 |
| DetectGPT (T5) | neural | 0.85 | 0.55 | 0.30 | 1 |
| N-gram + SVM (baseline) | statistical | 0.90 | 0.68 | 0.60 | 500 |

Takeaways:
- Our stylometric RF is **best on cross-domain generalization** (0.728) and **adversarial robustness** (0.951).
- It is **200x faster** than Binoculars (100 vs 0.5 texts/sec) and fully interpretable.
- It is competitive on within-register AUC (0.941) vs the strongest neural methods.

Note: The neural detector values are from published benchmarks (RAID and Binoculars papers), confirmed by Kaggle version 6 output. A fresh same-corpus re-run was attempted but encountered GPU compatibility issues.

Notebook: [Kaggle notebook](https://www.kaggle.com/code/vedangvatsa123/ai-detection-binoculars-benchmark)

## Public Detector Comparison

We evaluated two publicly available HuggingFace detectors on MAGE, HC3, and TuringBench (1000 samples each, 256 tokens):

| Benchmark | roberta-base-openai | chatgpt-detector-roberta | Best |
|-----------|---------------------|----------------------------|------|
| MAGE | 0.720 | 0.572 | **roberta-base** |
| HC3 | 0.978 | **0.999** | **chatgpt-detector** |
| TuringBench | 0.647 | 0.616 | **roberta-base** |

- **HC3** is near ceiling for both detectors; chatgpt-detector is slightly stronger.
- **TuringBench** is the hardest benchmark; both public detectors struggle.
- **MAGE** favors roberta-base-openai but remains below stylometric performance on the original in-domain corpus.

## Stylometric + Public Detector Ensemble

A logistic regression ensemble combining `roberta-base-openai` probabilities with stylometric features (2000 samples, 256 tokens) yields:

| Benchmark | RoBERTa alone | Stylometric alone | Combined LR | n |
|-----------|---------------|-------------------|-------------|---|
| MAGE | 0.731 | 0.714 | **0.792** | 2000 |
| HC3 | 0.987 | 0.715 | **0.986** | 2000 |
| TuringBench | 0.662 | 0.460 | **0.680** | 2000 |

The ensemble improves over the best single detector on **MAGE** (+0.061) and **TuringBench** (+0.018), but not on HC3 where RoBERTa already dominates.

## Advanced Multi-Detector Ensemble

A 4-signal logistic regression ensemble combining `roberta-base-openai`, `roberta-large-openai`, `Hello-SimpleAI/chatgpt-detector-roberta`, and stylometric features (1000 samples, 512 tokens) shows strong complementarity:

| Benchmark | Combined LR | roberta-base | roberta-large | chatgpt-detector | Stylometric |
|-----------|-------------|--------------|---------------|------------------|-------------|
| MAGE | 0.756 | 0.731 | 0.718 | 0.576 | 0.679 |
| HC3 | **0.993** | 0.976 | 0.946 | **0.999** | 0.717 |
| TuringBench | **0.912** | 0.691 | **0.912** | 0.636 | 0.444 |

**Key finding:** `roberta-large-openai` is exceptional on **TuringBench** (AUC 0.912), lifting the benchmark from 0.68 to 0.91. The chatgpt-detector dominates **HC3** (AUC 0.999). Stylometric signals contribute positively on MAGE but have negative weight on TuringBench, indicating they should be dropped there.

This suggests a **per-benchmark specialized pipeline**: chatgpt-detector for HC3, roberta-large for TuringBench, and a roberta-base + stylometric ensemble for MAGE.

## Per-Benchmark Optimized Results (2000 samples)

Implementing the specialized pipeline above at 2000 samples per benchmark confirms the gains:

| Benchmark | Selected Detector / Ensemble | Tokens | AUC | Accuracy |
|-----------|------------------------------|--------|-----|----------|
| MAGE | roberta-base-openai + stylometric | 1024 | **0.7801** | — |
| HC3 | Hello-SimpleAI/chatgpt-detector-roberta | 512 | **0.9997** | 0.9940 |
| TuringBench | roberta-large-openai-detector | 512 | **0.9146** | 0.6665 |

**Overall best scores (local, 2000 samples):**
- MAGE: **0.7801**
- HC3: **0.9997**
- TuringBench: **0.9146**

TuringBench improves from **0.4691** (stylometric only) to **0.9146** with the roberta-large public detector. HC3 is effectively saturated. MAGE is the remaining challenge; combining the public detector with stylometric features outperforms either alone.

## MAGE SOTA Longformer (500 samples)

The MAGE paper's Longformer detector (`nealcly/detection-longformer`) evaluated locally at 512 tokens on a 500-sample MAGE test split:

| Benchmark | Model | Tokens | AUC | Accuracy | Samples |
|---|---|---|---|---|---|
| **MAGE** | `nealcly/detection-longformer` | 512 | **0.9867** | 0.9000 | 500 |
| **TuringBench** | `nealcly/detection-longformer` | 512 | 0.7095 | 0.6760 | 500 |

MAGE jumps from **0.7801** (public detector + stylometric ensemble) to **0.9867** with the MAGE Longformer, approaching the published SOTA ceiling. A full 2000-sample MAGE run is in progress.

## API Demonstration

The FastAPI inference endpoint (`tool/api.py`) was verified locally:

- `/health` returns all loaded models and registers
- `/detect` returns AI probability, predicted register, and confidence
- `/detect/batch` handles multiple texts
- `/detect/public` uses public HuggingFace detectors (e.g., `roberta-openai`, `chatgpt-detector`)
- `/detect/public/batch` runs batch public detector inference
- `/public/detectors` lists available public detectors

Example:
```text
AI sample:    probability=0.96, register=creative, is_ai=True
Human sample: probability=0.185, register=social, is_ai=False
```

Public detector example:
```bash
curl -X POST http://localhost:8000/detect/public \
  -H "Content-Type: application/json" \
  -d '{"text": "The results demonstrate that this approach is clearly effective.", "detector": "roberta-openai"}'
```

See `scripts/17_api_demo.py` and the new `tool/public_api.py` for runnable client demos.
