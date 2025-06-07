# =============================================
# [7/9] 프리트레인 모델①: BERT
# =============================================
# 목표: BERT 모델을 Hugging Face 라이브러리로 불러와 감성 분류 태스크에 파인튜닝합니다.

# --- 1. 필수 패키지 설치 ---
# !pip install transformers datasets scikit-learn torch

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments as HFTrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import os

# Ensure GPU is available, otherwise use CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- 2. 데이터셋 및 모델/토크나이저 로드 ---
print("--- Loading dataset and model/tokenizer ---")
# 이번에는 한국어 데이터셋인 NSMC(Naver Sentiment Movie Corpus)를 사용해봅니다.
# Specify a cache directory within the project to keep things organized
cache_dir = os.path.join(os.getcwd(), ".cache_hf_datasets_07")
os.makedirs(cache_dir, exist_ok=True)
print(f"Using Hugging Face cache directory: {cache_dir}")

dataset = load_dataset("nsmc", cache_dir=cache_dir)
model_name = "klue/bert-base"  # 한국어 BERT 모델

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Loading model {model_name} for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(DEVICE) # Move model to the selected device

# --- 3. 데이터 전처리 ---
print("--- Preprocessing data ---")
def tokenize_function(examples):
    # max_length, batch_size 등 다양한 설정 비교 가능
    return tokenizer(examples["document"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 실습을 위해 데이터셋 일부만 사용
print("Selecting a smaller subset for training and evaluation...")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

# --- 4. Trainer API로 학습 스크립트 작성 ---
print("--- Setting up Trainer ---")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 학습 인자 설정
output_directory = os.path.join(os.getcwd(), "results_bert_finetune_07")
logging_directory = os.path.join(os.getcwd(), "logs_bert_finetune_07")
os.makedirs(output_directory, exist_ok=True)
os.makedirs(logging_directory, exist_ok=True)

training_args = HFTrainingArguments(
    output_dir=output_directory,
    num_train_epochs=1,  # 간단한 실습을 위해 1 에폭만
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir=logging_directory,
    logging_steps=10, # Log more frequently for smaller datasets
    eval_strategy="epoch", # Using the parameter name from inspect.signature
    save_strategy="epoch", # Save model at the end of each epoch
    load_best_model_at_end=True, # Load the best model found during training
    report_to="none" # Disable wandb or other integrations for simple script run
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# --- 5. 학습 및 평가 ---
print("BERT 파인튜닝을 시작합니다...")
trainer.train()

print("\n학습 완료! 평가를 시작합니다...")
eval_results = trainer.evaluate()
print(f"평가 결과: {eval_results}")

# --- 6. 간단한 추론 ---
print("\n--- Simple Inference Example ---")
text = "이 영화 정말 재미있어요! 배우들 연기가 최고네요."
print(f"입력: '{text}'")

# Ensure tokenizer produces tensors on the same device as the model
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
print(f"예측: {'긍정(1)' if predicted_class_id == 1 else '부정(0)'}")

print("\nScript finished.")
