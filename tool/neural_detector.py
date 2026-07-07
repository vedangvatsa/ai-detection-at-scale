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
_performer_model = None

def get_model_and_tokenizer(load_performer=False):
    global _model, _tokenizer, _device, _performer_model
    if _model is None or (load_performer and _performer_model is None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Device selection: MPS (Mac), CUDA (Nvidia), or CPU
        if _device is None:
            if torch.backends.mps.is_available():
                _device = torch.device("mps")
            elif torch.cuda.is_available():
                _device = torch.device("cuda")
            else:
                _device = torch.device("cpu")

        if _tokenizer is None:
            try:
                _tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
            except Exception as e:
                print(f"GPT-2 tokenizer not found in local cache ({e}).", file=sys.stderr)
                raise RuntimeError("GPT-2 model not cached locally.")

        if _model is None:
            print(f"Loading neural observer model 'gpt2' on {_device}...", file=sys.stderr)
            try:
                _model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True).to(_device)
                _model.eval()
            except Exception as e:
                print(f"GPT-2 model not found in local cache ({e}).", file=sys.stderr)
                raise RuntimeError("GPT-2 model not cached locally.")

        if load_performer and _performer_model is None:
            print(f"Loading neural performer model 'gpt2-medium' on {_device}...", file=sys.stderr)
            try:
                _performer_model = AutoModelForCausalLM.from_pretrained("gpt2-medium", local_files_only=True).to(_device)
                _performer_model.eval()
            except Exception as e:
                print(f"GPT-2 Medium model not found in local cache ({e}).", file=sys.stderr)
                raise RuntimeError("GPT-2 Medium model not cached locally.")

        # Ensure padding token is set
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
            _model.config.pad_token_id = _model.config.eos_token_id
            if _performer_model is not None:
                _performer_model.config.pad_token_id = _performer_model.config.eos_token_id

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

def compute_binoculars_score(text: str, max_length: int = 512) -> float:
    """
    Computes the Binoculars contrastive cross-perplexity score.
    Binoculars = ln(performer_perplexity) / cross_entropy(performer, observer)
    """
    if not text or not isinstance(text, str) or len(text.strip().split()) < 10:
        return 0.95  # Default baseline neutral score
        
    import torch
    
    # Load model and performer model
    model, tokenizer, device = get_model_and_tokenizer(load_performer=True)
    
    global _performer_model
    if _performer_model is None:
        raise RuntimeError("Performer model not loaded.")
        
    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
        return_attention_mask=True
    ).to(device)
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    if input_ids.size(1) < 3:
        return 0.95
        
    with torch.no_grad():
        obs_logits = model(input_ids, attention_mask=attention_mask).logits
        perf_logits = _performer_model(input_ids, attention_mask=attention_mask).logits
        
    shift_labels = input_ids[..., 1:].contiguous()
    shift_obs_logits = obs_logits[..., :-1, :].contiguous()
    shift_perf_logits = perf_logits[..., :-1, :].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    perf_loss = loss_fct(shift_perf_logits.view(-1, shift_perf_logits.size(-1)), shift_labels.view(-1))
    perf_loss = perf_loss.view(shift_labels.size()) * shift_mask
    sum_perf_loss = perf_loss.sum(dim=-1)
    n_tokens = shift_mask.sum(dim=-1)
    
    mean_perf_loss = sum_perf_loss / torch.clamp(n_tokens, min=1)
    perf_ppl = torch.exp(mean_perf_loss).cpu().item()
    
    perf_probs = torch.softmax(shift_perf_logits, dim=-1)
    obs_log_probs = torch.log_softmax(shift_obs_logits, dim=-1)
    
    cross_entropy_token = -torch.sum(perf_probs * obs_log_probs, dim=-1)
    masked_cross_entropy = cross_entropy_token * shift_mask
    sum_cross_entropy = masked_cross_entropy.sum(dim=-1)
    mean_cross_entropy = (sum_cross_entropy / torch.clamp(n_tokens, min=1)).cpu().item()
    
    if mean_cross_entropy > 0:
        binoculars = float(np.log(perf_ppl) / mean_cross_entropy)
    else:
        binoculars = 0.95
        
    return round(binoculars, 4)

if __name__ == "__main__":
    import json
    if len(sys.argv) > 1:
        sample_text = sys.argv[1]
    else:
        sample_text = "Furthermore, the results demonstrate that this approach is clearly effective."
    
    res = compute_perplexity_and_burstiness(sample_text)
    print(json.dumps(res, indent=2))
