# Kaggle ONNX Export Script
# 1. Add this as a Kaggle Notebook or run it in a Kaggle environment
# 2. Make sure you have a Hugging Face token in your Kaggle Secrets named "HF_TOKEN"
# 3. Add the following packages via pip:
#    !pip install transformers onnx onnxruntime huggingface_hub

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi, login
from kaggle_secrets import UserSecretsClient

# --- Configuration ---
MODEL_HUB_ID = "vedangvatsa/vedang-turingbench-roberta-large"
# ---------------------

# 1. Authenticate with Hugging Face
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("Successfully logged into Hugging Face.")
except Exception as e:
    print("Could not find HF_TOKEN in Kaggle Secrets. Please add it.")
    raise e

# 2. Wrapper Class for ONNX Export
class ModelOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

print(f"Downloading model {MODEL_HUB_ID}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_HUB_ID)

print("Exporting to ONNX...")
wrapper = ModelOnnxWrapper(model)
wrapper.cpu()
wrapper.eval()

dummy_input = {
    "input_ids": torch.ones(1, 256, dtype=torch.long),
    "attention_mask": torch.ones(1, 256, dtype=torch.long)
}

onnx_temp = "model_temp.onnx"
onnx_processed = "model_processed.onnx"
onnx_quant = "roberta_large_quantized.onnx"

torch.onnx.export(
    wrapper,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    onnx_temp,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    opset_version=18
)

print("Pre-processing ONNX model for shape inference...")
from onnxruntime.quantization.shape_inference import quant_pre_process
try:
    quant_pre_process(
        input_model_path=onnx_temp,
        output_model_path=onnx_processed,
        skip_optimization=False
    )
    model_to_quantize = onnx_processed
except Exception as e:
    print(f"Pre-processing failed: {e}. Quantizing raw model instead.")
    model_to_quantize = onnx_temp
    
print("Quantizing ONNX model to INT8...")
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_to_quantize,
    onnx_quant,
    weight_type=QuantType.QInt8
)

# Clean up temp files
if os.path.exists(onnx_temp): os.remove(onnx_temp)
if os.path.exists(onnx_processed): os.remove(onnx_processed)

print(f"Quantization complete! Uploading {onnx_quant} to Hugging Face...")

api = HfApi()
api.upload_file(
    path_or_fileobj=onnx_quant,
    path_in_repo=onnx_quant,
    repo_id=MODEL_HUB_ID,
    repo_type="model",
)

print("Upload successful! The quantized ONNX model is now available on Hugging Face.")
