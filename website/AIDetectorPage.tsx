import React from 'react';

const AIDetectorPage: React.FC = () => {
  return (
    <main className="min-h-screen bg-white text-slate-900">
      <article className="max-w-2xl mx-auto px-6 py-16">
        <header className="mb-12">
          <h1 className="text-3xl font-bold mb-4">How I Built an AI-Text Detector That Actually Works</h1>
          <p className="text-slate-500">An essay on data, features, models, benchmarks, and honest limits.</p>
        </header>

        <Prose>
          <p>
            By early 2024 it was already hard to tell if a piece of text was written by a human or by an AI. Tools existed, but most of them fell apart when the text was edited, paraphrased, or came from a model they had never seen. I wanted to build something that was fast enough to run at scale, interpretable enough that I could explain why it flagged something, robust enough to handle real-world tricks, and honest about what it could not do. This essay is the story of how I got there.
          </p>

          <h2>Start with the data</h2>
          <p>
            The first lesson was painful: the model is only as good as the corpus. I collected 2.77 million texts across four registers — academic, news, social media, and creative writing — with roughly 1.85 million AI-generated samples and 920,000 human samples. I split the data by register because the writing signals are different. Academic prose has long sentences and formal connectors. Social media is short, messy, and full of self-mentions. A single model trained on all of it confused those differences with the AI-vs-human signal, so I ended up training one detector per register plus a fallback detector that works across all of them.
          </p>

          <h2>Why stylometry?</h2>
          <p>
            Neural detectors like RoBERTa and GPTZero get the headlines, but they are black boxes, slow, GPU-hungry, and prone to overfitting to the exact model they were trained on. I chose a different path: stylometry. Instead of asking a neural network to memorize patterns, I measured interpretable properties of the text — sentence length variation, repetition, entropy, connector-word density — and fed those into a Random Forest. Random Forests are fast, handle tabular features well, and tell you which features mattered. That last point was important to me. If a detector flags a paragraph, I want to be able to say why.
          </p>

          <h2>The eleven features that survived</h2>
          <p>
            After a lot of experiments, eleven features became the production core: MTLD vocabulary diversity, sentence-length variation, self-mention density, opener ratio, connector density, hedge density, mean sentence length, booster density, character entropy, repetition rate, and punctuation entropy. Together these capture something AI text does consistently: it is more uniform, more predictable, and more balanced than human writing. Humans ramble, change rhythm, and repeat themselves in ways models usually avoid.
          </p>

          <h2>From eleven to thirty-five features</h2>
          <p>
            The eleven-feature model was good, but I wondered if readability, sentiment, syntax, and named-entity density could add signal. I added twenty-four more features and the AUC rose from 0.9645 to 0.9826. Accuracy went from 0.9011 to 0.9361. The gain was real, but I kept the production API on the original eleven features because they are faster, more stable, and easier to explain. The extra features are available for extended experiments, not for the default path. This was the first of many decisions where the best research score did not win. Production needs a different balance.
          </p>
          <div className="my-8 p-4 bg-slate-50 rounded-lg border border-slate-200">
            <ComparisonChart />
          </div>

          <h2>What the benchmarks taught me</h2>
          <p>
            Testing on public benchmarks brought surprises. Within-register performance looked excellent, with AUCs between 0.933 and 0.978. Academic text was easiest; social and creative were hardest. But the cross-domain test — train on one register, test on another — dropped the mean AUC to 0.728. That was the honest number. It taught me that a detector that looks perfect in one domain can be mediocre in another.
          </p>
          <p>
            Adversarial tests were more encouraging. Humanization attacks, where you remove connectors, swap synonyms, vary length, or add first-person text, only dropped the AUC by about 0.009. Paraphrase attacks on the RAID benchmark dropped it further, to 0.951. The surface-level stylometric signals were fairly robust, but a really good paraphrase could still fool the model.
          </p>

          <h2>Adding neural signals</h2>
          <p>
            I did not want to rely only on stylometry. I added public HuggingFace detectors: roberta-base-openai, Hello-SimpleAI/chatgpt-detector-roberta, roberta-large-openai, and the MAGE Longformer. Each had strengths and weaknesses. The chatgpt-detector was near-perfect on HC3. The roberta-large detector lifted TuringBench from 0.469 to 0.9146. The MAGE Longformer hit 0.9796 on MAGE. No single detector won everywhere, so I built per-benchmark specialized pipelines. HC3 goes to the chatgpt-detector. TuringBench goes to roberta-large. MAGE uses a public detector plus the stylometric ensemble. The final local numbers were 0.9801 on MAGE, 0.9997 on HC3, and 0.9146 on TuringBench.
          </p>

          <h2>Fine-tuning on TuringBench</h2>
          <p>
            The zero-shot roberta-large detector was already strong on TuringBench, but I wanted better. I fine-tuned roberta-large on the full TuringBench nineteen-generator training split — 331,000 texts — on Kaggle. After one epoch the validation AUC jumped to 0.9991 and accuracy to 0.9948. That confirmed something important: when you have enough in-domain data, a fine-tuned transformer beats everything else. But it also confirmed the opposite. Without that data, stylometry is a much cheaper and more generalizable fallback.
          </p>

          <h2>Turning a notebook into a product</h2>
          <p>
            A research notebook is not a product. I hardened the inference API with FastAPI endpoints, register classification, API-key authentication, IP rate limiting, in-memory caching with TTL, adversarial preprocessing, safe fallbacks for neural signals, and calibration for short texts. I also exposed public-detector endpoints so users can choose between the fast stylometric model and the slower but stronger public detectors.
          </p>

          <h2>The audit</h2>
          <p>
            Once the code was working, I did a deep audit of the repository. I wanted every claim to be verifiable and every piece of code to be clean. Marketing numbers in the README did not always match the code, so I corrected feature counts, clarified which benchmark scores came from which model, and documented limitations. Several scripts hardcoded the eleven feature names in multiple places, so I moved them to a single shared constant. The security code in the API was duplicated, so I extracted it into a shared module. The public API had duplicated single and batch logic, so I consolidated it. Some extended features used crude suffix rules, so I added an optional NLTK part-of-speech path while keeping the old behavior as the default. I removed unused imports, cleaned stale notebook references, and added tests for the feature extractor, adversarial defense, and API endpoints.
          </p>
          <p>The audit was not about making the code perfect. It was about making every decision explicit and every claim reproducible.</p>

          <h2>What I learned</h2>
          <p>
            Interpretability and performance are not enemies. A thirty-five-feature Random Forest can compete with large neural models when the features are chosen carefully. Domain matters more than model size. Ensembles beat single models. Robustness is harder than accuracy. And production is different from research — caching, rate limits, calibration, and graceful fallbacks matter as much as AUC.
          </p>

          <h2>Limitations I am honest about</h2>
          <p>
            The stylometric model is not robust to strong paraphrase or back-translation. It works best on longer texts; very short snippets are unreliable. Cross-register transfer is still only 0.728 AUC. Public detectors are evaluated models, not ones I trained. Calibration and adversarial defense are helpful but not magic.
          </p>

          <h2>What is next</h2>
          <p>
            I want to add a real part-of-speech pipeline and retrain the thirty-five-feature model. A per-sentence heatmap would help users see which parts of a text look most AI-like. I also want to improve low false-positive-rate performance for deployments where false positives are costly, add a Redis-backed cache for multi-worker deployments, and continue benchmarking against new models and attack methods as they appear.
          </p>

          <h2>Try it</h2>
          <p>
            The code, data manifest, and full results are on GitHub at{' '}
            <a href="https://github.com/vedangvatsa/ai-detection-at-scale" className="text-blue-600 underline">
              vedangvatsa/ai-detection-at-scale
            </a>.
            If you want to run it locally:
          </p>
          <pre className="my-4 p-4 bg-slate-900 text-slate-50 rounded-lg overflow-x-auto text-sm">
{`git clone https://github.com/vedangvatsa/ai-detection-at-scale.git
cd ai-detection-at-scale
python scripts/download_assets.py
python scripts/17_api_demo.py`}
          </pre>
          <p>
            The API will start at <code>http://127.0.0.1:8000</code> and you can try <code>/health</code>, <code>/detect</code>, and <code>/detect/public</code>.
          </p>
        </Prose>
      </article>
    </main>
  );
};

const Prose: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="prose prose-slate max-w-none leading-relaxed text-slate-700">
    {children}
  </div>
);

const ComparisonChart: React.FC = () => {
  const data = [
    { label: '11-feature AUC', value: 0.9645, color: '#94a3b8' },
    { label: '35-feature AUC', value: 0.9826, color: '#2563eb' },
    { label: '11-feature Accuracy', value: 0.9011, color: '#cbd5e1' },
    { label: '35-feature Accuracy', value: 0.9361, color: '#60a5fa' },
  ];
  return (
    <div>
      <h3 className="text-sm font-semibold mb-4 text-slate-700">11-feature vs 35-feature model</h3>
      <svg viewBox="0 0 400 160" className="w-full h-auto">
        {data.map((d, i) => {
          const y = i * 36 + 20;
          const width = d.value * 380;
          return (
            <g key={i}>
              <text x="0" y={y + 14} className="text-xs fill-slate-600" style={{ fontSize: 10 }}>
                {d.label}
              </text>
              <rect x="120" y={y} width={width} height={18} rx={4} fill={d.color} />
              <text x={128 + width} y={y + 14} className="text-xs fill-slate-700" style={{ fontSize: 10 }}>
                {d.value.toFixed(4)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

export default AIDetectorPage;
