# =============================================
# [9/9] 종합 실습 & 배포
# =============================================
# 목표: 학습된 모델을 ONNX로 변환해 보고, FastAPI를 이용한 API 서버 코드를 작성합니다.

# !pip install transformers torch onnx onnxruntime fastapi uvicorn python-multipart

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort
import numpy as np
import os

# --- 1. ONNX 변환 및 추론 ---
print("--- 1. ONNX 변환 및 추론 ---")

# 파인튜닝된 모델 경로 (07_finetune_bert.py의 output_dir)
base_dir = os.getcwd()
finetuned_model_path = os.path.join(base_dir, "results_bert_finetune_07", "checkpoint-63")
# ONNX 모델을 저장할 경로
onnx_output_dir = "./onnx_models"
onnx_model_filename = "bert_classifier_nsmc_finetuned.onnx"
onnx_model_path = os.path.join(onnx_output_dir, onnx_model_filename)

# ONNX 모델 저장 디렉토리 생성
if not os.path.exists(onnx_output_dir):
    os.makedirs(onnx_output_dir)
    print(f"Created directory: {onnx_output_dir}")

print(f"Loading fine-tuned model and tokenizer from: {finetuned_model_path}")

# 모델 및 토크나이저 로드
# try-except 블록으로 모델 로딩 실패 시 대체 모델 사용
try:
    if not os.path.exists(finetuned_model_path):
        raise FileNotFoundError(f"Fine-tuned model path does not exist: {finetuned_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_path)
    print("Successfully loaded fine-tuned model from ./results_bert_finetune_07/checkpoint-63")
except Exception as e:
    print(f"Failed to load fine-tuned model from {finetuned_model_path}: {e}")
    print("Falling back to base klue/bert-base model for ONNX conversion demonstration.")
    model_name_fallback = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name_fallback)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_fallback)
    onnx_model_path = os.path.join(onnx_output_dir, "bert_classifier_base.onnx") # Fallback ONNX path

model.eval() # 평가 모드로 설정

# 더미 입력 생성
dummy_text = "이것은 ONNX 변환을 위한 샘플 문장입니다."
print(f"Creating dummy input for ONNX export with text: '{dummy_text}'")
dummy_input = tokenizer(dummy_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

# torch.onnx.export로 모델 변환
print(f"Exporting model to ONNX format at: {onnx_model_path}")
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    onnx_model_path,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch_size'}, 
                  'attention_mask': {0: 'batch_size'}, 
                  'logits': {0: 'batch_size'}},
    opset_version=14
)
print(f"모델이 {onnx_model_path}에 저장되었습니다.")

# ONNX Runtime으로 추론
print("\n--- ONNX Runtime으로 추론 테스트 ---")
ort_session = ort.InferenceSession(onnx_model_path)

# PyTorch 추론 결과와 비교
with torch.no_grad():
    torch_logits = model(**dummy_input).logits.numpy()

# ONNX 추론
ort_inputs = {
    ort_session.get_inputs()[0].name: dummy_input['input_ids'].numpy(),
    ort_session.get_inputs()[1].name: dummy_input['attention_mask'].numpy()
}
ort_logits = ort_session.run(None, ort_inputs)[0]

# 결과 비교
try:
    np.testing.assert_allclose(torch_logits, ort_logits, rtol=1e-03, atol=1e-05)
    print("ONNX 추론 결과가 PyTorch 결과와 일치합니다!")
except AssertionError as e:
    print(f"ONNX 추론 결과와 PyTorch 결과가 다릅니다:\n{e}")
    print(f"PyTorch logits: {torch_logits}")
    print(f"ONNX logits: {ort_logits}")


# --- 2. FastAPI 서버 구현 (로컬 실행용 코드) ---
print("\n\n--- 2. FastAPI 서버 구현 (로컬 실행용 코드 예시) ---")
print("--- 아래는 로컬 환경에서 main.py로 저장하여 실행할 FastAPI 서버 코드 예시입니다. ---")

fastapi_code = """
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import os

# FastAPI 앱 초기화
app = FastAPI()

# 모델 및 토크나이저 로드 (앱 시작 시 한 번만)
# 실제 배포 시에는 여기서 ONNX 모델 경로와 토크나이저를 정확히 지정해야 합니다.
# 예: ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "./onnx_models/bert_classifier_nsmc_finetuned.onnx")
#     TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./results_bert_finetune_07/checkpoint-63") 
# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
# ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# 노트북 예제와 동일하게 klue/bert-base 및 기본 onnx 파일명 사용
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base") 
ort_session = ort.InferenceSession("bert_classifier.onnx") # 이 파일은 FastAPI 서버와 같은 위치에 있어야 함

# 요청 Body 모델 정의
class Item(BaseModel):
    text: str

@app.post("/predict/")
async def predict(item: Item):
    # 1. 입력 텍스트 토크나이징
    inputs = tokenizer(
        item.text, return_tensors="np", 
        padding="max_length", truncation=True, max_length=128
    )
    ort_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }

    # 2. ONNX 런타임으로 추론
    ort_logits = ort_session.run(None, ort_inputs)[0]
    
    # 3. 결과 처리 (클래스 0 또는 1로 가정)
    prediction = int(np.argmax(ort_logits, axis=1)[0])
    # probabilities = np.exp(ort_logits) / np.sum(np.exp(ort_logits), axis=-1, keepdims=True)
    # probability_of_predicted_class = float(probabilities[0, prediction])
    
    return {"prediction": prediction, "text": item.text}

# 실행 방법 (터미널에서 main.py가 있는 디렉토리에서 실행):
# 1. 필요한 패키지 설치: pip install fastapi uvicorn python-multipart transformers onnxruntime numpy
# 2. 위 코드를 main.py로 저장
# 3. ONNX 모델 파일 (예: bert_classifier.onnx)을 main.py와 같은 디렉토리에 위치시키거나 경로 수정
# 4. FastAPI 서버 실행: uvicorn main:app --reload
"""
print(fastapi_code)


# --- 3. Docker 이미지화 (로컬 실행용 파일 예시) ---
print("\n\n--- 3. Docker 이미지화 (로컬 실행용 파일 예시) ---")
print("--- 아래는 FastAPI 앱을 Docker로 패키징하기 위한 Dockerfile 및 requirements.txt 예시입니다. ---")

dockerfile_content = """
# 1. 베이스 이미지 선택
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 앱 소스코드와 모델 파일 복사
#    main.py는 FastAPI 코드를 담고 있어야 합니다.
#    bert_classifier.onnx는 변환된 ONNX 모델 파일입니다.
#    토크나이저 파일들도 필요하다면 함께 복사해야 합니다. (transformers 라이브러리가 허브에서 다운로드하도록 둘 수도 있음)
COPY ./main.py .
COPY ./bert_classifier.onnx . # 또는 onnx_models/bert_classifier_nsmc_finetuned.onnx 등 실제 경로
# 만약 results_07의 토크나이저를 사용한다면 해당 폴더도 COPY해야 합니다.
# COPY ./results_bert_finetune_07/checkpoint-63 /app/results_bert_finetune_07_checkpoint 

# 5. 포트 노출
EXPOSE 8000

# 6. 앱 실행 명령어
#    ONNX_MODEL_PATH와 TOKENIZER_PATH 환경 변수를 설정하여 main.py에서 사용하도록 할 수 있습니다.
#    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD uvicorn main:app --host 0.0.0.0 --port 8000
"""

requirements_content = """
fastapi
uvicorn[standard] # For standard ASGI server features
python-multipart # For form data
onnxruntime
transformers
torch # Usually onnxruntime doesn't strictly need torch, but transformers might for some tokenizer functionalities
numpy
"""

print("\n[Dockerfile 내용]")
print(dockerfile_content)
print("\n[requirements.txt 내용]")
print(requirements_content)
print("\n실행 명령어 예시 (터미널에서 Dockerfile, main.py, onnx 모델 등이 있는 디렉토리에서 실행):")
print("1. (필요시) main.py, bert_classifier.onnx, results_bert_finetune_07/checkpoint-63 (모델/토크나이저) 등을 현재 디렉토리에 준비")
print("2. docker build -t nlp-api .")
print("3. docker run -d -p 8000:8000 nlp-api")

print("\nScript finished.")
