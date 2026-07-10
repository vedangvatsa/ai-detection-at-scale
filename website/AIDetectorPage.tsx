import React, { useState } from 'react';

const FEATURE_DESCRIPTIONS: Record<string, string> = {
  mtld: 'Vocabulary Diversity (MTLD)',
  sent_cv: 'Sentence Length Variation',
  self_mention_density: 'Self-Mention (I, we, our)',
  opener_ratio: 'Sentence-Opening Connectors',
  connector_density: 'Connector Word Density',
  hedge_density: 'Hedging Words (maybe, likely)',
  mean_sent_len: 'Average Sentence Length',
  boost_density: 'Booster Words (clearly, indeed)',
  char_entropy: 'Character Trigram Entropy',
  rep_rate: 'Word Repetition Rate',
  punct_entropy: 'Punctuation Variety Entropy',
  noun_verb_ratio: 'Noun-to-Verb Ratio',
  adj_adv_ratio: 'Adjective-to-Adverb Ratio',
  pos_transition_entropy: 'Grammatical Sequence Variety',
  sent_length_std: 'Sentence Length Standard Deviation'
};

const AIDetectorPage: React.FC = () => {
  const [text, setText] = useState('');
  const [register, setRegister] = useState('all');
  const [sensitivity, setSensitivity] = useState('normal');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const handleDetect = async () => {
    if (wordCount < 20) {
      setError('Please enter at least 20 words to analyze.');
      return;
    }
    setError(null);
    setIsLoading(true);
    try {
      const response = await fetch(`${apiUrl}/detect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          register: register === 'all' ? null : register,
          return_features: true,
          sensitivity: sensitivity
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'API request failed.');
      }

      const data = await response.json();
      setResults(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Could not connect to the local detector server. Make sure it is running at the specified URL.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased">
      {/* Interactive Hero Header */}
      <section className="relative overflow-hidden bg-slate-900 border-b border-slate-800 py-16 px-6">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.15),rgba(255,255,255,0))]" />
        <div className="max-w-5xl mx-auto relative z-10">
          <header className="mb-10 text-center">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 mb-4">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
              SOTA Hybrid Cascading Classifier Active
            </span>
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-4">
              AI Text Detection at Scale
            </h1>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Test our register-aware ensemble pipeline. Paste any text to extract deep stylometric features, neural perplexity, and real-time explainability z-scores.
            </p>
          </header>

          {/* Interactive Sandbox Dashboard */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mt-12 bg-slate-900/50 backdrop-blur-md rounded-2xl border border-slate-800 p-6 md:p-8">
            {/* Input Panel */}
            <div className="lg:col-span-7 flex flex-col space-y-4">
              <div className="flex justify-between items-center">
                <label className="block text-sm font-semibold text-slate-300">
                  Input Text
                </label>
                <span className={`text-xs ${wordCount >= 20 ? 'text-slate-400' : 'text-amber-500 font-medium'}`}>
                  {wordCount} words {wordCount < 20 && '(Minimum 20 words required)'}
                </span>
              </div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste the text you want to analyze here..."
                className="w-full h-80 px-4 py-3 bg-slate-950 border border-slate-800 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-slate-100 placeholder-slate-600 resize-none"
              />
              
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">
                    Register Mode
                  </label>
                  <select
                    value={register}
                    onChange={(e) => setRegister(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-slate-300 focus:ring-1 focus:ring-blue-500"
                  >
                    <option value="all">Auto-detect Register</option>
                    <option value="news">News Register</option>
                    <option value="academic">Academic Register</option>
                    <option value="social">Social Media Register</option>
                    <option value="creative">Creative Writing Register</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">
                    Sensitivity
                  </label>
                  <select
                    value={sensitivity}
                    onChange={(e) => setSensitivity(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-slate-300 focus:ring-1 focus:ring-blue-500"
                  >
                    <option value="high">High (Low AI threshold)</option>
                    <option value="normal">Normal (Balanced)</option>
                    <option value="low">Low (Strict verification)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">
                    API Endpoint
                  </label>
                  <input
                    type="text"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-slate-300 focus:ring-1 focus:ring-blue-500 text-xs"
                  />
                </div>
              </div>

              <button
                onClick={handleDetect}
                disabled={isLoading || wordCount < 20}
                className="w-full py-3 px-4 mt-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-semibold rounded-xl transition duration-200 shadow-lg shadow-blue-900/20 flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Analyzing Linguistic Signatures...
                  </>
                ) : (
                  'Analyze Text'
                )}
              </button>

              {error && (
                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-xs text-red-400 mt-2">
                  {error}
                </div>
              )}
            </div>

            {/* Results Panel */}
            <div className="lg:col-span-5 flex flex-col justify-between border-t lg:border-t-0 lg:border-l border-slate-800 pt-6 lg:pt-0 lg:pl-8">
              {!results && !isLoading && (
                <div className="flex flex-col items-center justify-center text-center h-full py-12">
                  <svg className="w-12 h-12 text-slate-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <h3 className="text-base font-semibold text-slate-300">Ready to Analyze</h3>
                  <p className="text-xs text-slate-500 max-w-xs mt-1.5">
                    Paste your article or copy a generated text above to see the cascading confidence tiers, register classifications, and feature outlier z-scores.
                  </p>
                </div>
              )}

              {isLoading && (
                <div className="flex flex-col items-center justify-center text-center h-full py-12 space-y-4">
                  <div className="relative w-20 h-20">
                    <div className="absolute inset-0 rounded-full border-4 border-blue-500/10" />
                    <div className="absolute inset-0 rounded-full border-4 border-t-blue-500 animate-spin" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-300">Cascading Classifier Active</h3>
                    <p className="text-xs text-slate-500 mt-1 animate-pulse">
                      Running Level 1 Stylometrics...
                    </p>
                  </div>
                </div>
              )}

              {results && !isLoading && (
                <div className="space-y-6 h-full flex flex-col justify-between">
                  {/* Gauge */}
                  <div className="text-center">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                      AI Likelihood
                    </h3>
                    <div className="relative inline-flex items-center justify-center">
                      <svg className="w-36 h-36 transform -rotate-90">
                        <circle
                          cx="72"
                          cy="72"
                          r="64"
                          className="stroke-slate-800"
                          strokeWidth="8"
                          fill="transparent"
                        />
                        <circle
                          cx="72"
                          cy="72"
                          r="64"
                          className={`transition-all duration-1000 ${
                            results.ai_probability >= 0.5 ? 'stroke-red-500' : 'stroke-green-500'
                          }`}
                          strokeWidth="8"
                          fill="transparent"
                          strokeDasharray={402}
                          strokeDashoffset={402 - (402 * results.ai_probability)}
                        />
                      </svg>
                      <div className="absolute flex flex-col items-center">
                        <span className="text-3xl font-extrabold text-white">
                          {(results.ai_probability * 100).toFixed(1)}%
                        </span>
                        <span className={`text-[10px] font-bold uppercase tracking-wider mt-0.5 px-2 py-0.5 rounded ${
                          results.is_ai ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-green-500/10 text-green-400 border border-green-500/20'
                        }`}>
                          {results.is_ai ? 'Flagged as AI' : 'Likely Human'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Badges details grid */}
                  <div className="grid grid-cols-2 gap-3 bg-slate-950/80 border border-slate-800/80 p-3.5 rounded-xl text-xs">
                    <div>
                      <span className="text-[10px] block text-slate-500 font-semibold uppercase tracking-wider">Detected Register</span>
                      <span className="font-semibold text-slate-200 capitalize mt-0.5 block">{results.register}</span>
                    </div>
                    <div>
                      <span className="text-[10px] block text-slate-500 font-semibold uppercase tracking-wider">Cascade Path</span>
                      <span className="font-semibold text-slate-200 capitalize mt-0.5 block">
                        {results.cascade_level ? results.cascade_level.replace(/_/g, ' ') : 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] block text-slate-500 font-semibold uppercase tracking-wider">Response Time</span>
                      <span className="font-semibold text-slate-200 mt-0.5 block">{results.processing_time_ms} ms</span>
                    </div>
                    <div>
                      <span className="text-[10px] block text-slate-500 font-semibold uppercase tracking-wider">Neural Signatures</span>
                      <span className="font-semibold text-slate-200 mt-0.5 block">
                        {results.neural_signals?.perplexity ? `PPL: ${results.neural_signals.perplexity.toFixed(1)}` : 'Fallback'}
                      </span>
                    </div>
                  </div>

                  {/* Sentence Heatmap Mini View */}
                  {results.sentences && results.sentences.length > 0 && (
                    <div className="border border-slate-800/80 p-3.5 rounded-xl">
                      <span className="text-[10px] block text-slate-400 font-semibold uppercase tracking-wider mb-2">Sentence-Level Highlights</span>
                      <div className="text-xs leading-relaxed max-h-32 overflow-y-auto pr-1 text-slate-400">
                        {results.sentences.map((sentObj: any, sIdx: number) => {
                          const prob = sentObj.probability;
                          const style = {
                            backgroundColor: prob >= 0.5 
                              ? `rgba(239, 68, 68, ${Math.min(0.7, (prob - 0.5) * 2)})` 
                              : `rgba(34, 197, 94, ${Math.min(0.4, (0.5 - prob) * 2)})`,
                            color: prob >= 0.6 || prob <= 0.3 ? '#fff' : 'inherit'
                          };
                          return (
                            <span 
                              key={sIdx} 
                              style={style}
                              className="px-0.5 py-0.25 mx-0.5 rounded transition duration-200 inline"
                              title={`AI Likelihood: ${(prob * 100).toFixed(1)}%`}
                            >
                              {sentObj.sentence}{' '}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Explainability Outlier Plotting (Z-score dashboard) */}
          {results && results.explainability && (
            <div className="bg-slate-900/50 backdrop-blur-md rounded-2xl border border-slate-800 p-6 md:p-8 mt-8">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-6 flex items-center gap-2">
                <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Stylometric Outliers (Z-score Relative to Human norms)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-5 text-xs">
                {Object.entries(results.explainability).map(([key, valObj]: any) => {
                  const desc = FEATURE_DESCRIPTIONS[key] || key;
                  const z = valObj.z_score;
                  const cappedZ = Math.max(-3, Math.min(3, z));
                  const percentage = ((cappedZ + 3) / 6) * 100;
                  
                  return (
                    <div key={key} className="flex flex-col space-y-2 border-b border-slate-900 pb-3">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-slate-300">{desc}</span>
                        <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded ${
                          z > 1.0 ? 'bg-red-500/10 text-red-400' : z < -1.0 ? 'bg-blue-500/10 text-blue-400' : 'bg-slate-800 text-slate-400'
                        }`}>
                          Z: {z > 0 ? `+${z.toFixed(2)}` : z.toFixed(2)}
                        </span>
                      </div>
                      
                      <div className="relative h-6 bg-slate-950 border border-slate-900 rounded-md overflow-hidden flex items-center">
                        <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-slate-800 z-10" />
                        <div 
                          className={`absolute top-0 bottom-0 transition-all duration-1000 ${
                            z >= 0 ? 'left-1/2 bg-red-500/30' : 'bg-blue-500/30'
                          }`}
                          style={{
                            left: z >= 0 ? '50%' : `${percentage}%`,
                            width: `${Math.abs(percentage - 50)}%`
                          }}
                        />
                        <div className="absolute inset-x-0 flex justify-between px-2 text-[8px] uppercase tracking-wider text-slate-600 font-semibold pointer-events-none z-20">
                          <span>Human-like</span>
                          <span>Average</span>
                          <span>AI-like</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Main Essay Section */}
      <section className="bg-slate-900 border-t border-slate-800">
        <article className="max-w-2xl mx-auto px-6 py-16">
          <header className="mb-12">
            <h1 className="text-3xl font-bold mb-4 text-white">How I Built an AI-Text Detector That Actually Works</h1>
            <p className="text-slate-400">An essay on data, features, models, benchmarks, and honest limits.</p>
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
            <div className="my-8 p-4 bg-slate-900 rounded-lg border border-slate-800">
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
              <a href="https://github.com/vedangvatsa/ai-detection-at-scale" className="text-blue-500 underline">
                vedangvatsa/ai-detection-at-scale
              </a>.
              If you want to run it locally:
            </p>
            <pre className="my-4 p-4 bg-slate-950 text-slate-100 border border-slate-800 rounded-lg overflow-x-auto text-sm">
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
      </section>
    </main>
  );
};

const Prose: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="prose prose-invert max-w-none leading-relaxed text-slate-400">
    {children}
  </div>
);

const ComparisonChart: React.FC = () => {
  const data = [
    { label: '11-feature AUC', value: 0.9645, color: '#64748b' },
    { label: '35-feature AUC', value: 0.9826, color: '#3b82f6' },
    { label: '11-feature Accuracy', value: 0.9011, color: '#475569' },
    { label: '35-feature Accuracy', value: 0.9361, color: '#60a5fa' },
  ];
  return (
    <div>
      <h3 className="text-sm font-semibold mb-4 text-slate-300">11-feature vs 35-feature model</h3>
      <svg viewBox="0 0 400 160" className="w-full h-auto">
        {data.map((d, i) => {
          const y = i * 36 + 20;
          const width = d.value * 240;
          return (
            <g key={i}>
              <text x="0" y={y + 14} className="text-[10px] fill-slate-400" style={{ fontSize: 10 }}>
                {d.label}
              </text>
              <rect x="130" y={y} width={width} height={18} rx={4} fill={d.color} />
              <text x={138 + width} y={y + 14} className="text-xs fill-slate-300" style={{ fontSize: 10 }}>
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
