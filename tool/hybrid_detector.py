import os
import numpy as np
import joblib
from tool.feature_extractor import extract_features
from tool.neural_detector import compute_perplexity_and_burstiness

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ONNX_PATH = os.path.join(MODELS_DIR, 'deberta_onnx_quantized.onnx')
ENSEMBLERS_PATH = os.path.join(MODELS_DIR, 'beemo_register_ensemblers.joblib')

_tokenizer = None
_ort_session = None
_ensemblers = None

def _load_models():
    global _tokenizer, _ort_session, _ensemblers
    
    if _ensemblers is None:
        if not os.path.exists(ENSEMBLERS_PATH):
            raise FileNotFoundError(f"Register ensemblers not found at {ENSEMBLERS_PATH}. Run scripts/download_beemo_models.py first.")
        _ensemblers = joblib.load(ENSEMBLERS_PATH)
        
    if _ort_session is None:
        if not os.path.exists(ONNX_PATH):
            raise FileNotFoundError(f"Quantized DeBERTa ONNX model not found at {ONNX_PATH}. Run scripts/download_beemo_models.py first.")
        
        import onnxruntime as ort
        from transformers import AutoTokenizer
        
        # Load ONNX Inference Session
        _ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        _tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

def predict_hybrid(text: str, register: str = "all") -> float:
    """
    Computes SOTA hybrid prediction combining:
    - 11 stylometric features
    - GPT-2 perplexity & burstiness
    - Binoculars score
    - Fine-tuned DeBERTa Beemo-specific MGT probability (ONNX quantized)
    And routes to the register-specific ensembler.
    """
    _load_models()
    
    # 1. Extract 11 stylometric features
    feats = extract_features(text, extended=False)
    if feats is None:
        return 0.5 # Neutral fallback
        
    stylo_vector = [feats[k] for k in ['mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
                                      'connector_density', 'hedge_density', 'mean_sent_len',
                                      'boost_density', 'char_entropy', 'rep_rate', 'punct_entropy']]
                                      
    # 2. Extract Perplexity and Burstiness (offline cache-friendly)
    try:
        neural = compute_perplexity_and_burstiness(text)
        ppl, burst = neural['perplexity'], neural['burstiness']
    except Exception:
        ppl, burst = 50.0, 1.0 # Default fallback
        
    # 3. Extract Binoculars score (offline cache-friendly)
    try:
        from tool.neural_detector import compute_binoculars_score
        bino = compute_binoculars_score(text)
    except Exception:
        bino = 0.95  # Neutral fallback
        
    # 4. Extract semantic probability from quantized DeBERTa ONNX model
    try:
        inputs = _tokenizer(text, return_tensors="np", truncation=True, max_length=256, padding=True)
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        logits = _ort_session.run(None, ort_inputs)[0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        deberta_prob = float(probs[0][1])
    except Exception as e:
        print(f"Error executing DeBERTa ONNX inference: {e}")
        deberta_prob = 0.5
        
    # 5. Feed ensembled vector to the mapped register Logistic Regression ensembler
    ensemble_vector = np.array(stylo_vector + [ppl, burst, bino, deberta_prob]).reshape(1, -1)
    
    # Select register ensembler
    ensembler = _ensemblers.get(register, _ensemblers.get('all'))
    if ensembler is None:
        ensembler = list(_ensemblers.values())[0]
        
    final_prob = ensembler.predict_proba(ensemble_vector)[0][1]
    
    return float(final_prob)
