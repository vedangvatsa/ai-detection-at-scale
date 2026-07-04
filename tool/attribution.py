#!/usr/bin/env python3
"""
Model source attribution module.
Uses the pre-trained attribution classifier to predict the likely writing source
(human, openai, llama, mistral, cohere, mpt, other).
"""
import os
import joblib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'attribution_classifier.joblib')

_model = None

def get_attribution_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        else:
            print(f"WARNING: Attribution model not found at {MODEL_PATH}")
    return _model

def attribute_source(feature_vector, is_ai_probability):
    """
    Predicts the model source of a text given its feature vector.
    Fails back to 'human' if the overall AI probability is low (< 0.25).
    """
    # If the general classifier says it is highly likely human, attribute to human directly
    if is_ai_probability < 0.25:
        return {
            "source_model": "human",
            "confidence": round(1.0 - is_ai_probability, 4)
        }
        
    model = get_attribution_model()
    if model is None:
        return {
            "source_model": "unknown",
            "confidence": 0.0
        }
        
    X = np.array([feature_vector])
    
    try:
        pred_group = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        classes = model.classes_
        
        idx = list(classes).index(pred_group)
        confidence = float(probs[idx])
        
        # Override group if predicted human but we are confident it is AI
        if pred_group == 'human' and is_ai_probability > 0.6:
            # Pick the second most likely model (the highest non-human model)
            non_human_indices = [i for i, c in enumerate(classes) if c != 'human']
            if non_human_indices:
                best_nh_idx = max(non_human_indices, key=lambda i: probs[i])
                pred_group = classes[best_nh_idx]
                confidence = float(probs[best_nh_idx])
                
        # Beautify display names
        display_names = {
            'human': 'Human',
            'openai': 'OpenAI (GPT-4 / ChatGPT)',
            'llama': 'Meta Llama',
            'mistral': 'Mistral AI',
            'cohere': 'Cohere',
            'mpt': 'MosaicML MPT',
            'other': 'Other AI'
        }
        
        return {
            "source_model": display_names.get(pred_group, pred_group),
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        print(f"Error during attribution: {e}")
        return {
            "source_model": "unknown",
            "confidence": 0.0
        }
