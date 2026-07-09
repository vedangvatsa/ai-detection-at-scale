# How I Built an AI-Text Detector That Actually Works

By early 2024 it was already hard to tell if a piece of text was written by a human or by an AI. Tools existed, but most of them fell apart when the text was edited, paraphrased, or came from a model they had never seen. I wanted to build something that was fast enough to run at scale, interpretable enough that I could explain why it flagged something, robust enough to handle real-world tricks, and honest about what it could not do. This essay is the story of how I got there.

## Start with the data

The first lesson was painful: the model is only as good as the corpus. I collected 2.77 million texts across four registers — academic, news, social media, and creative writing — with roughly 1.85 million AI-generated samples and 920,000 human samples. I split the data by register because the writing signals are different. Academic prose has long sentences and formal connectors. Social media is short, messy, and full of self-mentions. A single model trained on all of it confused those differences with the AI-vs-human signal, so I ended up training one detector per register plus a fallback detector that works across all of them.

## Why stylometry?

Neural detectors like RoBERTa and GPTZero get the headlines, but they are black boxes, slow, GPU-hungry, and prone to overfitting to the exact model they were trained on. I chose a different path: stylometry. Instead of asking a neural network to memorize patterns, I measured interpretable properties of the text — sentence length variation, repetition, entropy, connector-word density — and fed those into a Random Forest. Random Forests are fast, handle tabular features well, and tell you which features mattered. That last point was important to me. If a detector flags a paragraph, I want to be able to say why.

## The eleven features that survived

After a lot of experiments, eleven features became the production core: MTLD vocabulary diversity, sentence-length variation, self-mention density, opener ratio, connector density, hedge density, mean sentence length, booster density, character entropy, repetition rate, and punctuation entropy. Together these capture something AI text does consistently: it is more uniform, more predictable, and more balanced than human writing. Humans ramble, change rhythm, and repeat themselves in ways models usually avoid.

## From eleven to thirty-five features

The eleven-feature model was good, but I wondered if readability, sentiment, syntax, and named-entity density could add signal. I added twenty-four more features and the AUC rose from 0.9645 to 0.9826. Accuracy went from 0.9011 to 0.9361. The gain was real, but I kept the production API on the original eleven features because they are faster, more stable, and easier to explain. The extra features are available for extended experiments, not for the default path. This was the first of many decisions where the best research score did not win. Production needs a different balance.

## What the benchmarks taught me

Testing on public benchmarks brought surprises. Within-register performance looked excellent, with AUCs between 0.933 and 0.978. Academic text was easiest; social and creative were hardest. But the cross-domain test — train on one register, test on another — dropped the mean AUC to 0.728. That was the honest number. It taught me that a detector that looks perfect in one domain can be mediocre in another.

Adversarial tests were more encouraging. Humanization attacks, where you remove connectors, swap synonyms, vary length, or add first-person text, only dropped the AUC by about 0.009. Paraphrase attacks on the RAID benchmark dropped it further, to 0.951. The surface-level stylometric signals were fairly robust, but a really good paraphrase could still fool the model.

## Adding neural signals

I did not want to rely only on stylometry. I added public HuggingFace detectors: roberta-base-openai, Hello-SimpleAI/chatgpt-detector-roberta, roberta-large-openai, and the MAGE Longformer. Each had strengths and weaknesses. The chatgpt-detector was near-perfect on HC3. The roberta-large detector lifted TuringBench from 0.469 to 0.9146. The MAGE Longformer hit 0.9796 on MAGE. No single detector won everywhere, so I built per-benchmark specialized pipelines. HC3 goes to the chatgpt-detector. TuringBench goes to roberta-large. MAGE uses a public detector plus the stylometric ensemble. The final local numbers were 0.9801 on MAGE, 0.9997 on HC3, and 0.9146 on TuringBench.

## Fine-tuning on TuringBench

The zero-shot roberta-large detector was already strong on TuringBench, but I wanted better. I fine-tuned roberta-large on the full TuringBench nineteen-generator training split — 331,000 texts — on Kaggle. After one epoch the validation AUC jumped to 0.9991 and accuracy to 0.9948. That confirmed something important: when you have enough in-domain data, a fine-tuned transformer beats everything else. But it also confirmed the opposite. Without that data, stylometry is a much cheaper and more generalizable fallback.

## Turning a notebook into a product

A research notebook is not a product. I hardened the inference API with FastAPI endpoints, register classification, API-key authentication, IP rate limiting, in-memory caching with TTL, adversarial preprocessing, safe fallbacks for neural signals, and calibration for short texts. I also exposed public-detector endpoints so users can choose between the fast stylometric model and the slower but stronger public detectors.

## The audit

Once the code was working, I did a deep audit of the repository. I wanted every claim to be verifiable and every piece of code to be clean. Marketing numbers in the README did not always match the code, so I corrected feature counts, clarified which benchmark scores came from which model, and documented limitations. Several scripts hardcoded the eleven feature names in multiple places, so I moved them to a single shared constant. The security code in the API was duplicated, so I extracted it into a shared module. The public API had duplicated single and batch logic, so I consolidated it. Some extended features used crude suffix rules, so I added an optional NLTK part-of-speech path while keeping the old behavior as the default. I removed unused imports, cleaned stale notebook references, and added tests for the feature extractor, adversarial defense, and API endpoints.

The audit was not about making the code perfect. It was about making every decision explicit and every claim reproducible.

## What I learned

Interpretability and performance are not enemies. A thirty-five-feature Random Forest can compete with large neural models when the features are chosen carefully. Domain matters more than model size. Ensembles beat single models. Robustness is harder than accuracy. And production is different from research — caching, rate limits, calibration, and graceful fallbacks matter as much as AUC.

## Limitations I am honest about

The stylometric model is not robust to strong paraphrase or back-translation. It works best on longer texts; very short snippets are unreliable. Cross-register transfer is still only 0.728 AUC. Public detectors are evaluated models, not ones I trained. Calibration and adversarial defense are helpful but not magic.

## What is next

I want to add a real part-of-speech pipeline and retrain the thirty-five-feature model. A per-sentence heatmap would help users see which parts of a text look most AI-like. I also want to improve low false-positive-rate performance for deployments where false positives are costly, add a Redis-backed cache for multi-worker deployments, and continue benchmarking against new models and attack methods as they appear.

## Try it

The code, data manifest, and full results are on GitHub at [vedangvatsa/ai-detection-at-scale](https://github.com/vedangvatsa/ai-detection-at-scale). If you want to run it locally:

```bash
git clone https://github.com/vedangvatsa/ai-detection-at-scale.git
cd ai-detection-at-scale
python scripts/download_assets.py
python scripts/17_api_demo.py
```

The API will start at `http://127.0.0.1:8000` and you can try `/health`, `/detect`, and `/detect/public`.
