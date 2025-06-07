# 1. 베이스 이미지 선택
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 앱 소스코드와 모델 파일 복사
COPY main.py .

# Create a directory for model assets
RUN mkdir -p /app/model_assets

# Copy the fine-tuned ONNX model
# Source: ./onnx_models/bert_classifier_nsmc_finetuned.onnx (relative to Docker build context)
# Destination: /app/model_assets/model.onnx (as referenced in main.py)
COPY ./onnx_models/bert_classifier_nsmc_finetuned.onnx /app/model_assets/model.onnx

# Copy the fine-tuned tokenizer files
# Source: ./results_bert_finetune_07/checkpoint-63 (relative to Docker build context)
# Destination: /app/model_assets/tokenizer (as referenced in main.py)
COPY ./results_bert_finetune_07/checkpoint-63 /app/model_assets/tokenizer/

# 5. 포트 노출
EXPOSE 8000

# 6. 앱 실행 명령어
# main.py uses environment variables TOKENIZER_PATH and ONNX_MODEL_PATH
# with defaults /app/model_assets/tokenizer and /app/model_assets/model.onnx respectively.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
