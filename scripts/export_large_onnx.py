#!/usr/bin/env python3
import os
import torch
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_HUB_ID = "vedangvatsa/vedang-turingbench-roberta-large"

class ModelOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def main():
    local_model_dir = os.path.join(MODELS_DIR, "roberta_large_semantic_model")
    onnx_temp = os.path.join(MODELS_DIR, "roberta_large_semantic_model.onnx")
    onnx_quant = os.path.join(MODELS_DIR, "roberta_large_onnx_quantized.onnx")
    
    os.makedirs(local_model_dir, exist_ok=True)
    
    print(f"Downloading tokenizer and model from {MODEL_HUB_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB_ID)
    tokenizer.save_pretrained(local_model_dir)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_HUB_ID)
    
    print("Exporting to ONNX...")
    wrapper = ModelOnnxWrapper(model)
    wrapper.cpu()
    wrapper.eval()
    
    dummy_input = {
        "input_ids": torch.ones(1, 256, dtype=torch.long),
        "attention_mask": torch.ones(1, 256, dtype=torch.long)
    }
    
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
    onnx_processed = os.path.join(MODELS_DIR, "roberta_large_semantic_model_processed.onnx")
    
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
    if os.path.exists(onnx_temp):
        os.remove(onnx_temp)
    if os.path.exists(onnx_processed):
        os.remove(onnx_processed)
        
    print(f"Quantization complete! {onnx_quant} is ready.")
    print(f"Tokenizer saved to {local_model_dir}.")

if __name__ == "__main__":
    main()
