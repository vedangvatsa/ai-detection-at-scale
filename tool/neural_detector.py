#!/usr/bin/env python3
"""
Neural feature extraction using a lightweight pretrained transformer (GPT-2 Small).
Computes token-level perplexity and burstiness.
"""
import sys
import numpy as np

# Lazy load PyTorch and Transformers to keep startup latency low
_model = None
_tokenizer = None
_device = None

def get_model_and_tokenizer():
    global _model, _tokenizer, _device
    if _model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Device selection: MPS (Mac), CUDA (Nvidia), or CPU
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")

        print(f"Loading neural observer model 'gpt2' on {_device}...", file=sys.stderr)
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _model = AutoModelForCausalLM.from_pretrained("gpt2").to(_device)
        _model.eval()
        
        # Ensure padding token is set
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
            _model.config.pad_token_id = _model.config.eos_token_id
            
        print("Neural observer model loaded successfully.", file=sys.stderr)
        
    return _model, _tokenizer, _device

def compute_perplexity_and_burstiness(text: str, max_length: int = 512):
    """
    Computes Perplexity and Burstiness of a given text using GPT-2.
    
    Perplexity is the exponentiated average negative log-likelihood of tokens.
    Burstiness is the standard deviation of token log-likelihoods.
    """
    if not text or not isinstance(text, str) or len(text.strip().split()) < 5:
        return {"perplexity": 0.0, "burstiness": 0.0}

    import torch
    
    model, tokenizer, device = get_model_and_tokenizer()
    
    # Preprocess text slightly (clean newlines, limit length)
    text = text.strip()
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length, 
        padding=True,
        return_attention_mask=True
    ).to(device)
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # We need at least 3 tokens to compute standard deviation (burstiness)
    if input_ids.size(1) < 3:
        return {"perplexity": 0.0, "burstiness": 0.0}
        
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        
    # Align logits and labels for perplexity (N-1 shifts)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Loss shape: (batch_size * (seq_len - 1))
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    
    # Apply attention mask to ignore pad tokens
    masked_loss = loss * shift_mask
    sum_loss = masked_loss.sum(dim=-1)
    n_tokens = shift_mask.sum(dim=-1)
    
    # Average token cross entropy
    mean_loss = sum_loss / torch.clamp(n_tokens, min=1)
    perplexity = torch.exp(mean_loss).cpu().item()
    
    # Extract non-zero token losses for burstiness (std dev)
    token_losses_np = masked_loss.cpu().numpy()[0]
    mask_np = shift_mask.cpu().numpy()[0]
    active_losses = token_losses_np[mask_np == 1]
    
    if len(active_losses) > 1:
        burstiness = float(np.std(active_losses))
    else:
        burstiness = 0.0
        
    return {
        "perplexity": round(perplexity, 4),
        "burstiness": round(burstiness, 4)
    }

if __name__ == "__main__":
    import json
    if len(sys.argv) > 1:
        sample_text = sys.argv[1]
    else:
        sample_text = "Furthermore, the results demonstrate that this approach is clearly effective."
    
    res = compute_perplexity_and_burstiness(sample_text)
    print(json.dumps(res, indent=2))
