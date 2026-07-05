import os
import torch
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tool.feature_extractor import extract_features
from tool.neural_detector import compute_perplexity_and_burstiness

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
SEMANTIC_MODEL_PATH = os.path.join(MODELS_DIR, 'beemo_semantic_model')
ENSEMBLER_PATH = os.path.join(MODELS_DIR, 'beemo_hybrid_ensembler.joblib')

_tokenizer = None
_semantic_model = None
_ensembler = None
_device = None

def _get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    return _device

def _load_models():
    global _tokenizer, _semantic_model, _ensembler
    
    if _ensembler is None:
        if not os.path.exists(ENSEMBLER_PATH):
            raise FileNotFoundError(f"Beemo hybrid ensembler model not found at {ENSEMBLER_PATH}. Run scripts/download_beemo_models.py first.")
        _ensembler = joblib.load(ENSEMBLER_PATH)
        
    if _semantic_model is None:
        if not os.path.exists(SEMANTIC_MODEL_PATH):
            raise FileNotFoundError(f"Beemo semantic transformer model not found at {SEMANTIC_MODEL_PATH}. Run scripts/download_beemo_models.py first.")
        
        device = _get_device()
        _tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_MODEL_PATH)
        _semantic_model = AutoModelForSequenceClassification.from_pretrained(SEMANTIC_MODEL_PATH).to(device)
        _semantic_model.eval()

def predict_hybrid(text: str) -> float:
    """
    Computes SOTA hybrid prediction combining:
    - 11 stylometric features
    - GPT-2 perplexity & burstiness
    - Fine-tuned DistilBERT Beemo-specific MGT probability
    """
    _load_models()
    
    # 1. Extract 11 stylometric features
    feats = extract_features(text, extended=False)
    if feats is None:
        return 0.5 # Neutral fallback
        
    stylo_vector = [feats[k] for k in ['mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
                                      'connector_density', 'hedge_density', 'mean_sent_len',
                                      'boost_density', 'char_entropy', 'rep_rate', 'punct_entropy']]
                                      
    # 2. Extract Perplexity and Burstiness
    try:
        neural = compute_perplexity_and_burstiness(text)
        ppl, burst = neural['perplexity'], neural['burstiness']
    except Exception:
        ppl, burst = 50.0, 1.0 # Default fallback
        
    # 3. Extract semantic probability from fine-tuned DistilBERT
    device = _get_device()
    with torch.no_grad():
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
        logits = _semantic_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        sem_prob = probs[0][1].item()
        
    # 4. Feed ensembled vector to Logistic Regression
    ensemble_vector = np.array(stylo_vector + [ppl, burst, sem_prob]).reshape(1, -1)
    final_prob = _ensembler.predict_proba(ensemble_vector)[0][1]
    
    return float(final_prob)
