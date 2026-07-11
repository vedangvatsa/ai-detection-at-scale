#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    
    print(f"Loading PyTorch model from {MODEL_HUB_ID}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_HUB_ID)
    model.eval()
    
    print("Loading tokenizer from roberta-large...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    
    device = torch.device("cpu")
    print(f"Using device for feature extraction: {device}")
    model.to(device)
    
    # Load Beemo dataset
    print("Loading Toloka Beemo dataset...")
    beemo = load_dataset("toloka/beemo")
    df = beemo['train'].to_pandas()
    
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]
    
    def extract_pairs(dataframe):
        records = []
        for _, row in dataframe.iterrows():
            h = row['human_output']
            m = row['model_output']
            if isinstance(h, str) and len(h.strip()) >= 20:
                records.append({'text': h, 'label': 0})
            if isinstance(m, str) and len(m.strip()) >= 20:
                records.append({'text': m, 'label': 1})
        return pd.DataFrame(records)
        
    train_data = extract_pairs(train_df)
    test_data = extract_pairs(test_df)
    
    print(f"Train pairs: {len(train_data)}, Test pairs: {len(test_data)}")
    
    def extract_embeddings(dataframe):
        embeddings = []
        labels = []
        batch_size = 16
        for i in tqdm(range(0, len(dataframe), batch_size)):
            batch = dataframe.iloc[i:i+batch_size]
            texts = batch['text'].tolist()
            lbls = batch['label'].tolist()
            
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.roberta(**inputs)
                sequence_output = outputs[0]
                x = sequence_output[:, 0, :]
                x = model.classifier.dense(x)
                x = torch.tanh(x)
                
            embeddings.append(x.cpu().numpy())
            labels.extend(lbls)
            
        return np.vstack(embeddings), np.array(labels)
        
    print("Extracting train embeddings...")
    X_train, y_train = extract_embeddings(train_data)
    
    print("Extracting test embeddings...")
    X_test, y_test = extract_embeddings(test_data)
    
    # Train Logistic Regression
    print("Training Logistic Regression on embeddings...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    
    train_preds = clf.predict_proba(X_train)[:, 1]
    test_preds = clf.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    print(f"\nLinear Probe Train AUC: {train_auc:.4f}")
    print(f"Linear Probe Test AUC: {test_auc:.4f}")
    
    # Overwrite classifier out_proj weights in the PyTorch model
    print("\nInjecting Logistic Regression weights into PyTorch model...")
    weight_coef = clf.coef_[0]
    bias_intercept = clf.intercept_[0]
    
    with torch.no_grad():
        model.classifier.out_proj.weight[0, :] = 0.0
        model.classifier.out_proj.weight[1, :] = torch.tensor(weight_coef, dtype=torch.float32)
        model.classifier.out_proj.bias[0] = 0.0
        model.classifier.out_proj.bias[1] = torch.tensor(bias_intercept, dtype=torch.float32)
        
    # Verification
    print("Verifying mathematical equivalence...")
    sample_text = test_data.iloc[0]['text']
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    emb = X_test[0].reshape(1, -1)
    sk_probs = clf.predict_proba(emb)[0]
    print(f"PyTorch Probs: {probs}")
    print(f"Sklearn Probs: {sk_probs}")
    assert np.allclose(probs, sk_probs, atol=1e-5), "Probs are not mathematically close!"
    print("Assertion passed! PyTorch model updated successfully.")
    
    # Save the updated model and tokenizer locally
    print(f"\nSaving updated model to {local_model_dir}...")
    model.cpu()
    model.save_pretrained(local_model_dir)
    tokenizer.save_pretrained(local_model_dir)
    
    # Export to ONNX
    print("Exporting updated model to ONNX...")
    wrapper = ModelOnnxWrapper(model)
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
    
    # Pre-processing ONNX
    print("Pre-processing ONNX model...")
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
    print("Everything updated successfully!")

if __name__ == '__main__':
    main()
