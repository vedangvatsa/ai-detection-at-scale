import math

def calibrate_probability(probability: float, word_count: int, target_length: int = 50, steepness: float = 0.08) -> float:
    """
    Calibrates prediction probability based on document length.
    Short texts naturally exhibit high variance in perplexity and stylometrics,
    increasing the risk of false positives.
    
    This function applies a length-conditioned weight that pulls the probability
    towards 0.5 (neutral boundary) for documents shorter than `target_length`.
    """
    if word_count >= target_length:
        return probability
        
    # Logistic weight scaling based on word count
    # At word_count = target_length, weight ~ 0.5 (starts fully trusting)
    # At word_count = 10, weight ~ 0.07 (almost completely dampens)
    # At word_count = 30, weight ~ 0.31 (moderately dampens)
    x = word_count - (target_length / 2.0)
    weight = 1.0 / (1.0 + math.exp(-steepness * x))
    
    # Interpolate probability towards 0.5
    calibrated = 0.5 + weight * (probability - 0.5)
    
    return float(calibrated)
