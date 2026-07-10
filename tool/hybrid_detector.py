import os
import numpy as np
import joblib
from tool.feature_extractor import extract_features, ORIGINAL_FEATURE_COLS
from tool.neural_detector import compute_perplexity_and_burstiness

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ONNX_PATH = os.path.join(MODELS_DIR, 'deberta_onnx_quantized.onnx')
ENSEMBLERS_PATH = os.path.join(MODELS_DIR, 'beemo_register_ensemblers.joblib')

_tokenizer = None
_ort_session = None
_ensemblers = None
_pytorch_model = None

def _load_models():
    global _tokenizer, _ort_session, _ensemblers, _pytorch_model
    
    if _ensemblers is None:
        if os.path.exists(ENSEMBLERS_PATH):
            _ensemblers = joblib.load(ENSEMBLERS_PATH)
        else:
            # Fallback to single hybrid ensembler mapped as register dictionary
            single_path = os.path.join(MODELS_DIR, 'beemo_hybrid_ensembler.joblib')
            if not os.path.exists(single_path):
                raise FileNotFoundError(f"No ensemblers found at {ENSEMBLERS_PATH} or {single_path}.")
            single_clf = joblib.load(single_path)
            _ensemblers = {
                "all": single_clf,
                "academic": single_clf,
                "news": single_clf,
                "social": single_clf,
                "creative": single_clf
            }
        
    if _ort_session is None and _pytorch_model is None:
        from transformers import AutoTokenizer
        
        # Try loading quantized ONNX first
        if os.path.exists(ONNX_PATH):
            try:
                import onnxruntime as ort
                _ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
                
                pytorch_path = os.path.join(MODELS_DIR, 'beemo_semantic_model')
                if os.path.exists(pytorch_path):
                    _tokenizer = AutoTokenizer.from_pretrained(pytorch_path)
                else:
                    _tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
                return
            except Exception as e:
                print(f"ONNX session load failed: {e}. Falling back to PyTorch model.")
                
        # PyTorch fallback
        pytorch_path = os.path.join(MODELS_DIR, 'beemo_semantic_model')
        if not os.path.exists(pytorch_path):
            raise FileNotFoundError(f"DeBERTa model not found at {ONNX_PATH} or {pytorch_path}.")
            
        from transformers import AutoModelForSequenceClassification
        import torch
        _pytorch_model = AutoModelForSequenceClassification.from_pretrained(pytorch_path)
        _pytorch_model.cpu()
        _pytorch_model.eval()
        _tokenizer = AutoTokenizer.from_pretrained(pytorch_path)

def predict_hybrid(text: str, register: str = "all") -> float:
    """
    Computes SOTA hybrid prediction combining:
    - 11 stylometric features
    - GPT-2 perplexity & burstiness
    - Binoculars score
    - Fine-tuned DeBERTa Beemo-specific MGT probability (ONNX or PyTorch)
    And routes to the register-specific ensembler.
    """
    _load_models()
    
    # 1. Extract 11 stylometric features
    feats = extract_features(text, extended=False)
    if feats is None:
        return 0.5 # Neutral fallback
        
    stylo_vector = [feats[k] for k in ORIGINAL_FEATURE_COLS]
                                      
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
        
    # 4. Extract semantic probability from quantized DeBERTa ONNX model or PyTorch fallback
    deberta_prob = 0.5
    try:
        if _ort_session is not None:
            inputs = _tokenizer(text, return_tensors="np", truncation=True, max_length=256, padding="max_length")
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            logits = _ort_session.run(None, ort_inputs)[0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            deberta_prob = float(probs[0][1])
        elif _pytorch_model is not None:
            import torch
            inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
            # Remove token_type_ids if present (some DeBERTa implementations do not support it)
            inputs.pop('token_type_ids', None)
            
            with torch.no_grad():
                outputs = _pytorch_model(**inputs)
                logits = outputs.logits.cpu().numpy()
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            deberta_prob = float(probs[0][1])
    except Exception as e:
        print(f"Error executing DeBERTa inference: {e}")
        deberta_prob = 0.5
        
    # Select register ensembler
    ensembler = _ensemblers.get(register, _ensemblers.get('all'))
    if ensembler is None:
        ensembler = list(_ensemblers.values())[0]
        
    # 5. Feed ensembled vector to the mapped register Logistic Regression ensembler
    # Dynamically match features count (old ensembler has 14 features without binoculars, new has 15)
    num_features = getattr(ensembler, 'n_features_in_', 14)
    if num_features == 14:
        ensemble_vector = np.array(stylo_vector + [ppl, burst, deberta_prob]).reshape(1, -1)
    else:
        ensemble_vector = np.array(stylo_vector + [ppl, burst, bino, deberta_prob]).reshape(1, -1)
        
    final_prob = ensembler.predict_proba(ensemble_vector)[0][1]
    
    return float(final_prob)
