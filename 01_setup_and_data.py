# =============================================
# [1/9] 환경 셋업 & 데이터 준비
# =============================================
# 목표: Python NLP 실습에 필요한 패키지를 설치하고, 실습용 코퍼스를 내려받아 구조를 확인합니다.

# --- 1. 필수 패키지 설치 ---
# Colab 환경에서는 아래 명령어로 !를 붙여 실행합니다.
# !pip install torch torchtext transformers datasets

# --- 2. Hugging Face 데이터셋 로딩 및 확인 ---
from datasets import load_dataset

print("--- IMDb 데이터셋 로딩 ---")
# 'imdb' 또는 'sst2' 등 원하는 데이터셋을 로드할 수 있습니다.
# cache_dir을 지정하면 Colab 세션이 끊겨도 다시 다운로드할 필요가 없어 편리합니다.
imdb_dataset = load_dataset("imdb", cache_dir="./.cache")

print("\n데이터셋 구조:")
print(imdb_dataset)

print("\n학습 데이터 샘플 10개:")
for i in range(10):
    print(f"[{i+1}] Label: {imdb_dataset['train'][i]['label']}, Text: {imdb_dataset['train'][i]['text'][:80]}...")

# --- 3. 토크나이저(BPE vs WordPiece) 간단 비교 ---
from transformers import AutoTokenizer

# BERT는 WordPiece를 사용합니다.
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# GPT-2는 BPE를 사용합니다.
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

sample_text = "Natural Language Processing is fascinating!"

print(f"\n--- 원문: {sample_text} ---")

# WordPiece (BERT) 토크나이징
# 단어의 시작이 아닌 조각(subword)은 '##' 접두사가 붙습니다.
bert_tokens = bert_tokenizer.tokenize(sample_text)
print(f"\nBERT (WordPiece) 토큰: {bert_tokens}")

# BPE (GPT-2) 토크나이징
# 띄어쓰기를 'Ġ' 문자로 표현하여 처리합니다.
gpt2_tokens = gpt2_tokenizer.tokenize(sample_text)
print(f"GPT-2 (BPE) 토큰: {gpt2_tokens}")

print("\n비교: WordPiece는 ##로 서브워드를, BPE는 Ġ(공백)으로 단어 시작을 구분하는 경향이 있습니다.")
