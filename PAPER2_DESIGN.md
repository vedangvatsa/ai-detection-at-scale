# Paper 2: Research Design Document
## "Domain-Invariant vs Register-Dependent Stylometric Features for AI Text Detection: A Benchmark Study at Web Scale"

*Revised after 100-reviewer synthesis — v2.0*

---

## 1. Refined Research Questions

**Primary RQ:** Which of the core stylometric features are domain-invariant (AUC stable across domains) vs register-dependent (AUC degrades significantly under domain shift)?

**Secondary RQs:**
- RQ2: What is the performance cost of cross-domain generalization (train on domain A, test on domain B)?
- RQ3: Do feature value distributions shift significantly across registers, and does that shift predict classifier failure?
- RQ4: What minimum text length is required for reliable feature estimation per domain?

**Null hypothesis (per feature):** Feature F trained on source domain D₀ predicts AI authorship in target domain Dₜ at no better than majority-class baseline AUC.

---

## 2. Framing Change (from reviewer consensus)

**Old framing:** "Do simple features generalize at web scale?"
**New framing:** "We benchmark the domain-generalization profile of 8 stylometric features, identifying which are invariant and which are register-dependent. This establishes the operating limits of interpretable AI detection."

This is publishable whether the result is positive (features generalize) or negative (features don't) because establishing the **limits** is the contribution.

**Novelty claim (state explicitly in abstract):** This is the first cross-domain, cross-model stylometric generalization benchmark at scale (10M texts, 5 registers, 11+ models).

---

## 3. Corpus Design

### 3.1 Register Taxonomy (formal axis — not just "domain")

Separate **register** from **domain** per reviewer consensus:

| Register ID | Register Name | Example Sources | Human Sources | AI Sources |
|-------------|--------------|-----------------|---------------|------------|
| R1 | Academic/Scientific | PubMed, arXiv | PubMed abstracts, arXiv | RAID academic subset, M4 |
| R2 | News/Journalism | CC-News, RealNews | CC-News human | RAID news subset |
| R3 | Social/Conversational | Reddit, StackExchange | Reddit pushshift (pre-2022) | RAID Reddit subset, HC3 |
| R4 | Encyclopedic | Wikipedia | Wikipedia (human-edited) | RAID Wikipedia subset |
| R5 | Creative/Narrative | BookCorpus, essays | BookCorpus | RAID creative subset |

### 3.2 Domain-Matched Constraint
**Critical:** For each AI source, the human comparison texts must come from the **same register**. No cross-register comparisons as primary results (only secondary analysis).

### 3.3 Data Sources

| Source | Type | Est. Size | Register(s) | Notes |
|--------|------|-----------|-------------|-------|
| RAID (Dugan et al. 2024) | AI (11 models) | ~6M | R1–R5 | Core dataset; deduplicate against M4 |
| PubMed abstracts | Human | ~500K | R1 | Pre-2022 only for clean baseline |
| arXiv abstracts | Human | ~500K | R1 | Pre-2022 preferred |
| CC-News | Human | ~1M | R2 | Verify licensing |
| Reddit pushshift (pre-2022) | Human | ~1M | R3 | Ethics: publicly posted content |
| Wikipedia dumps | Human | ~500K | R4 | Note: no first-person by policy — flag self-mention feature |
| BookCorpus subset | Human | ~200K | R5 | Verify CC license |
| M4 (Wang et al. 2023) | AI (multiple) | ~200K | R1–R3 | Deduplicate vs RAID |
| HC3 (Guo et al. 2023) | AI (ChatGPT) | ~50K | R1, R3 | 2023 vintage — label model version |
| ShareGPT | AI (ChatGPT) | ~100K | R3 | Selection bias noted; use with caveat |

**Total target:** ~10M texts across both classes, stratified so no single source >30% of any register.

### 3.4 Minimum Length Filter (per register)
- R1 Academic: ≥100 words (same as Paper 1)
- R2 News: ≥150 words (full article paragraphs)
- R3 Social: ≥30 words (enforce — short texts unreliable for MTLD)
- R4 Encyclopedic: ≥100 words
- R5 Creative: ≥100 words

Document rejection rates per register.

### 3.5 Deduplication
- Cross-source MinHash LSH deduplication (Jaccard threshold 0.8)
- Report: number of duplicates removed per source pair

---

## 4. Feature Set (Updated)

### 4.1 Original 8 Features (Paper 1) — Retained with Flags

| # | Feature | Generalizes? | Known Confound | Action |
|---|---------|-------------|----------------|--------|
| 1 | MTLD (Lexical Diversity) | ✅ Likely | Length-sensitive below 30 words | Enforce min-length; report per-register |
| 2 | Sentence Length CV | ✅ Likely | Bullets/lists break sentence boundaries | Pre-process list items |
| 3 | Self-Mention Density | ⚠️ Register-dependent | Wikipedia = structurally 0; R4 excluded | Flag; exclude R4 from this feature |
| 4 | Sentence-Opener Connector Ratio | ✅ Moderate | Informal connectors in social media | Expand word list for R3 |
| 5 | Connector Density | ✅ Moderate | Same as above | Same fix |
| 6 | Hedge Density | ⚠️ Register-dependent | Academic hedges ≠ casual hedges | Build register-specific word lists |
| 7 | Mean Sentence Length | ⚠️ Moderate | Trivially confounded by text type | Report but don't use as primary |
| 8 | Booster Density | ❌ Weak | d=0.07 in Paper 1; weakest feature | Retain for comparison; do not lead with it |

### 4.2 New Features (3 domain-neutral additions, from reviewer consensus)

| # | Feature | Rationale |
|---|---------|-----------|
| 9 | **Character N-gram Entropy** | Measures character-level predictability; model-agnostic signal |
| 10 | **Within-Document Word Repetition Rate** | AI repeats key terms at characteristic rates regardless of domain |
| 11 | **Punctuation Entropy** | AI uses punctuation (commas, colons, semicolons) in characteristic patterns |

### 4.3 Feature Normalization
- All density features: per-1,000 words (same as Paper 1)
- All features: report raw distributions **and** domain-normalized (z-score within register) values
- MTLD: flag all texts under 50 words; exclude from MTLD analysis

---

## 5. Statistical Methodology

### 5.1 Primary Analysis
- **Effect sizes (Cohen's d)** per feature per register — this is the primary result, not p-values
- At N=10M, all p-values will be <0.001; never report significance as a finding
- Bootstrapped 95% CIs on all effect size estimates (B=1,000)
- Multiple testing correction: Benjamini-Hochberg (8 features × 5 registers = 40 tests)

### 5.2 Generalization Analysis (core contribution)
- **Cross-domain AUC matrix**: Train on register Rᵢ, test on Rⱼ for all i,j pairs — report 5×5 heatmap per feature
- **Domain-invariance score**: mean off-diagonal AUC per feature (higher = more invariant)
- **Performance cost of domain shift**: diagonal AUC − mean off-diagonal AUC per feature

### 5.3 Mixed-Effects Model
- Per-feature: fit linear mixed model with domain as random effect
- Tests whether feature distribution differences are stable after accounting for domain variance

### 5.4 Classifier Evaluation
- **Feature importance**: permutation importance (not Gini — addresses reviewer bias concern)
- **Calibration curves** per register — not just AUC
- **Domain-stratified train/test split** — no random split allowed
- **Ablation table**: 1-feature classifiers for all 11 features; top-2, top-4, all-11
- **Imbalanced evaluation**: test at realistic class ratios (1:10, 1:100 AI:human) not just balanced 50:50

### 5.5 Baseline Comparisons
- Majority-class baseline
- Unigram TF-IDF logistic regression per register
- (Optional) GPTZero API on 10K sample subset for comparison

---

## 6. Fairness Analysis (mandatory per reviewer consensus)

- Proxy for author language background: country of affiliation (where available from metadata)
- Report false positive rate by language background proxy
- Report feature distributions for likely non-native vs native English corpora
- **Scope statement in abstract**: "This study reports population-level statistics; stylometric features should not be used for individual-level attribution."

---

## 7. Ethics and Data Licensing

| Dataset | License | Concern | Resolution |
|---------|---------|---------|------------|
| RAID | Research use | None | Cite Dugan et al. 2024 |
| PubMed | Open access | None | Use API |
| arXiv | Open access | None | Use bulk S3 dump |
| CC-News | CC | Verify ToS | Check before download |
| Reddit pushshift | Public posts | Privacy at scale | Use pre-2022 only; no usernames retained |
| Wikipedia | CC-BY-SA | None | Standard use |
| ShareGPT | Unknown | Selection bias + license unclear | Mark as supplementary only |

Include ethics statement in paper.

---

## 8. Paper Structure

1. **Abstract** — 1 sentence RQ, 1 sentence method, 2 sentences key findings, 1 sentence implication
2. **Introduction** — Motivation → gap → contribution (explicit novelty claim) → paper overview
3. **Related Work** — RAID (Dugan 2024), TURINGBENCH (Uchendu 2020), M4 (Wang 2023), stylometrics for AI detection, domain adaptation in NLP
4. **Methodology**
   - Register taxonomy
   - Corpus construction with pipeline diagram (Figure 1)
   - Feature set (Table 1) with register flags
   - Statistical framework
5. **Results**
   - Feature distributions per register (Table 2 + Figure 2: violin plots)
   - Effect sizes per feature per register (Figure 3: heatmap)
   - Cross-domain generalization matrix (Figure 4: 5×5 AUC heatmap — the key figure)
   - Classification performance (Table 3: ablation + full model per register)
   - Calibration curves (Figure 5)
6. **Discussion**
   - Which features are domain-invariant (the answer to the paper)
   - Why register confounds specific features (theoretical grounding)
   - Adversarial robustness brief section
   - Fairness findings
7. **Limitations** — numbered list
8. **Conclusion** — 1-sentence practical recommendation
9. **References** — with DOIs

**Target length:** 8,000–9,000 words (ACL/EMNLP long paper format)

---

## 9. Key Figures to Generate

| Figure | Type | What it shows |
|--------|------|--------------|
| Fig 1 | Pipeline diagram | Data → features → classifier → evaluation |
| Fig 2 | Violin plots (11 features × 5 registers) | Distribution shift across registers |
| Fig 3 | Heatmap (feature × register, Cohen's d) | Effect size stability — the domain-invariance picture |
| Fig 4 | 5×5 AUC heatmap | Cross-domain generalization (KEY FIGURE) |
| Fig 5 | Calibration curves (per register) | Practical usability at each threshold |
| Fig 6 | Ablation bar chart | Accuracy with 1, 2, 4, 11 features |

---

## 10. What Paper 2 Explicitly Does NOT Do
- Does not use Paper 1's corpus or results
- Does not make individual-level detection claims
- Does not claim the 8 features are universally superior to neural detectors
- Does not use Gini importance as primary feature importance metric
- Does not report p-values as primary evidence of difference at N=10M

---

## 11. Revised Feature Generalization Hypothesis (pre-registration)

Based on reviewer consensus and domain knowledge, before running any analysis:

**Expected domain-invariant features (hypothesis):**
1. MTLD — most likely to generalize (model-level vocabulary behavior)
2. Sentence Length CV — AI structural uniformity is model-not-domain-driven
3. Character N-gram Entropy — domain-neutral signal

**Expected register-dependent features (hypothesis):**
4. Self-Mention Density — Wikipedia = 0 by policy; news = passive; only academic meaningful
5. Hedge Density — word lists are academic-specific
6. Booster Density — weakest signal even in Paper 1

**Uncertain:**
7. Connector Density — depends on word list breadth
8. Sentence-Opener Connector Ratio — possibly domain-neutral
9. Within-Document Repetition — untested, hypothesize domain-neutral
10. Punctuation Entropy — hypothesize domain-neutral
11. Mean Sentence Length — likely register-dependent (tweets vs essays)

---

*Document version: post-100-reviewer synthesis | Last updated: June 2026*
