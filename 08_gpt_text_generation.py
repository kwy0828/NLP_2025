# =============================================
# [8/9] 프리트레인 모델②: GPT 계열 & 생성 태스크
# =============================================
# 목표: GPT 모델의 텍스트 생성 방식을 이해하고, 다양한 샘플링 기법을 실습합니다.

# !pip install transformers torch

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 모델 및 토크나이저 로드 ---
# 한국어 GPT 모델 (KoGPT2)
model_name = "skt/kogpt2-base-v2"
print(f"--- Loading tokenizer and model for {model_name} ---")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

# --- 2. 간단한 텍스트 생성 ---
prompt = "인공지능 모델은 인간에게"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

print(f"\n--- 프롬프트: {prompt} ---")
# 기본(Greedy Search) 생성
print("Generating text with default settings (greedy search)...")
sample_outputs = model.generate(input_ids, max_length=50)
print("\n[기본 생성 결과]")
print(tokenizer.decode(sample_outputs[0], skip_special_tokens=True))

# --- 3. generate() 파라미터 변화 실험 ---
print("\n--- Experimenting with generate() parameters ---")
def generate_text(current_prompt, temp, top_k, top_p):
    print(f"\n--- Generating with: temp={temp}, top_k={top_k}, top_p={top_p} ---")
    print(f"Prompt: {current_prompt}")
    input_ids = tokenizer.encode(current_prompt, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        max_length=100, # Increased max_length for more varied output
        do_sample=True, # 샘플링을 하려면 True로 설정
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id # Suppress warning for pad_token_id
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))

prompt_for_sampling = "오늘 날씨가 좋아서,"

# Temperature: 높을수록 창의적(랜덤), 낮을수록 결정적
generate_text(prompt_for_sampling, temp=0.7, top_k=50, top_p=1.0) # 약간 창의적
generate_text(prompt_for_sampling, temp=1.2, top_k=50, top_p=1.0) # 매우 창의적

# Top-k sampling: 확률이 높은 k개 중에서만 샘플링
generate_text(prompt_for_sampling, temp=1.0, top_k=10, top_p=1.0) # 상위 10개 단어만 사용

# Top-p (Nucleus) sampling: 확률 합이 p를 넘는 최소 단어 집합에서 샘플링
generate_text(prompt_for_sampling, temp=1.0, top_k=0, top_p=0.92) # 확률 합 92% 내 단어만 사용 (top_k=0 to disable it)

# --- 4. 전이 태스크 적용 (Prompting) ---
print("\n--- 프롬프트를 이용한 작업 수행 (요약 예시) ---")

# 요약
summary_prompt = """
텍스트: 자연어 처리(NLP)는 인공지능의 한 분야로, 컴퓨터가 인간의 언어를 이해하고, 해석하며, 생성할 수 있도록 하는 기술이다. 최근 트랜스포머 아키텍처의 등장으로 NLP 기술은 비약적인 발전을 이루었으며, 기계 번역, 텍스트 요약, 챗봇 등 다양한 분야에 응용되고 있다.
요약:
"""
print(f"Prompt for summarization:\n{summary_prompt}")
input_ids = tokenizer.encode(summary_prompt, return_tensors='pt').to(device)
summary_output = model.generate(
    input_ids, 
    max_length=150, 
    repetition_penalty=1.2, # To reduce repetitive generation
    pad_token_id=tokenizer.eos_token_id # Suppress warning
)
print("\n[요약 예시 결과]")
print(tokenizer.decode(summary_output[0], skip_special_tokens=True))

print("\nScript finished.")
