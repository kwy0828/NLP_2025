from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import os

# FastAPI 앱 초기화
app = FastAPI()

# --- Configuration for model and tokenizer paths ---
# These paths will correspond to where files are copied in the Dockerfile
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/app/model_assets/tokenizer")
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "/app/model_assets/model.onnx")

# --- Load model and tokenizer ---
try:
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # Fallback or error handling can be added here
    # For now, let's re-raise to make it clear if loading fails during startup
    raise

# 요청 Body 모델 정의
class Item(BaseModel):
    text: str

@app.post("/predict/")
async def predict(item: Item):
    try:
        # 1. 입력 텍스트 토크나이징
        inputs = tokenizer(
            item.text, return_tensors="np",
            padding="max_length", truncation=True, max_length=128
        )
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        # BERT ONNX models usually take token_type_ids as well.
        # If your ONNX model expects it, uncomment the following:
        # if 'token_type_ids' in inputs:
        #    ort_inputs['token_type_ids'] = inputs['token_type_ids'].astype(np.int64)

        # 2. ONNX 런타임으로 추론
        ort_logits = ort_session.run(None, ort_inputs)[0]

        # 3. 결과 처리 (클래스 0 또는 1로 가정)
        prediction = int(np.argmax(ort_logits, axis=1)[0])

        return {"prediction": prediction, "text": item.text, "status": "success"}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e), "status": "error"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# To run this FastAPI server locally (outside Docker):
# 1. Save this as main.py in your project root.
# 2. Create a directory structure like this in your project root:
#    ./model_assets/tokenizer/  <- Copy contents of ./results_bert_finetune_07/checkpoint-63/* here
#    ./model_assets/model.onnx  <- Copy ./onnx_models/bert_classifier_nsmc_finetuned.onnx here
# 3. Run: uvicorn main:app --reload --port 8000
