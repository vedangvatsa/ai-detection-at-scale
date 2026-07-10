#!/usr/bin/env python3
import os
import torch
import joblib
from transformers import AutoModelForSequenceClassification

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

class ModelOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def main():
    model_path = os.path.join(MODELS_DIR, "beemo_semantic_model")
    onnx_temp = os.path.join(MODELS_DIR, "deberta_semantic_model.onnx")
    onnx_quant = os.path.join(MODELS_DIR, "deberta_onnx_quantized.onnx")
    
    print("Loading PyTorch model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    wrapper = ModelOnnxWrapper(model)
    wrapper.cpu()
    wrapper.eval()
    
    print("Exporting to ONNX...")
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
    onnx_processed = os.path.join(MODELS_DIR, "deberta_semantic_model_processed.onnx")
    
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
        
    print("Quantization complete! deberta_onnx_quantized.onnx is ready.")

    # Convert single ensembler to register ensembler dictionary
    hybrid_path = os.path.join(MODELS_DIR, "beemo_hybrid_ensembler.joblib")
    ensemblers_path = os.path.join(MODELS_DIR, "beemo_register_ensemblers.joblib")
    
    if os.path.exists(hybrid_path):
        print("Loading single ensembler and mapping to register dictionary...")
        single_clf = joblib.load(hybrid_path)
        ensemblers = {
            "all": single_clf,
            "academic": single_clf,
            "news": single_clf,
            "social": single_clf,
            "creative": single_clf
        }
        joblib.dump(ensemblers, ensemblers_path)
        print("Successfully created beemo_register_ensemblers.joblib.")
    else:
        print("Error: beemo_hybrid_ensembler.joblib not found!")

if __name__ == "__main__":
    main()
