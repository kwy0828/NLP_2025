# =============================================
# [4/9] Seq2Seq 기본 구조
# =============================================
# 목표: 인코더-디코더 구조를 이해하고, 간단한 기계 번역 모델을 구현합니다.

# --- 1. 기본 설정 및 데이터 준비 ---
# !pip install torch torchtext transformers datasets sacrebleu
# !python -m spacy download en_core_web_sm
# !python -m spacy download fr_core_news_sm

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader # DataLoader is not used in the current script logic
from datasets import load_dataset
import spacy
import random
from collections import Counter

print("--- 2. 데이터셋 및 토크나이저 준비 ---")
# Multi30k 데이터셋 (영어 -> 프랑스어) 로드 -> iwslt2017로 변경
print("Loading iwslt2017 dataset (en-fr)...")
# Using a small subset for faster processing in script conversion context
raw_datasets = load_dataset("iwslt2017", "iwslt2017-en-fr", split="train[:1%]") 

# Spacy 토크나이저 로드
print("Loading spacy tokenizers...")
try:
    spacy_en = spacy.load("en_core_web_sm")
    spacy_fr = spacy.load("fr_core_news_sm")
except OSError:
    print("Spacy models 'en_core_web_sm' or 'fr_core_news_sm' not found.")
    print("Please run: ")
    print("python -m spacy download en_core_web_sm")
    print("python -m spacy download fr_core_news_sm")
    print("And ensure they are installed in your environment.")
    exit()

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

# 간단한 어휘집(Vocabulary) 클래스
class Vocab:
    def __init__(self, tokenizer, min_freq=2):
        self.tokenizer = tokenizer
        self.min_freq = min_freq
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        
    def build_vocab(self, sentences):
        print(f"Building vocab with {len(sentences)} sentences...")
        counter = Counter()
        for i, sentence in enumerate(sentences):
            if i % 100 == 0:
                print(f"  Tokenizing sentence {i}/{len(sentences)}")
            counter.update(self.tokenizer(sentence))
        
        for word, count in counter.items():
            if count >= self.min_freq: # Use min_freq parameter
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx
        print(f"Vocabulary size: {len(self.itos)}")

# 어휘집 생성
print("Building vocabularies...")
SRC_VOCAB = Vocab(tokenize_en)
TRG_VOCAB = Vocab(tokenize_fr)

# Accessing data from iwslt2017 structure
# The dataset is a list of dictionaries, each with a 'translation' key
# which is another dictionary {'en': 'english_sentence', 'fr': 'french_sentence'}
src_sentences = [d['translation']['en'] for d in raw_datasets]
trg_sentences = [d['translation']['fr'] for d in raw_datasets]

SRC_VOCAB.build_vocab(src_sentences)
TRG_VOCAB.build_vocab(trg_sentences)

print("--- 3. Encoder, Decoder 클래스 작성 ---")
class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim)

    def forward(self, src):
        # src: [src_len, batch_size]
        embedded = self.embedding(src) # [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        # input: [batch_size]
        input = input.unsqueeze(0) # [1, batch_size]
        embedded = self.embedding(input) # [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0)) # [batch_size, output_dim]
        return prediction, hidden, cell

print("--- 4. Seq2Seq 모델 래핑 ---")
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = len(TRG_VOCAB.itos)
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        
        # 첫 번째 입력은 <sos> 토큰
        input_token = TRG_VOCAB.stoi['<sos>']
        # Ensure input is a tensor of shape [batch_size] filled with <sos> token index
        input_tensor = torch.full((batch_size,), input_token, device=self.device, dtype=torch.long)
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_tensor, hidden, cell)
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_tensor = trg[t] if teacher_force else top1
            
        return outputs

print("--- 5. 모델 학습 (간략화된 루프) ---")
INPUT_DIM = len(SRC_VOCAB.itos)
OUTPUT_DIM = len(TRG_VOCAB.itos)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

enc = EncoderRNN(INPUT_DIM, ENC_EMB_DIM, HID_DIM).to(DEVICE)
dec = DecoderRNN(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM).to(DEVICE)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

# (실제 학습 코드는 데이터 전처리, 배치화, 학습 루프 등 더 복잡합니다.)
print("Seq2Seq 모델 구조가 정의되었습니다.")
print("실습 과제: Teacher Forcing 비율을 0.0, 0.5, 1.0으로 바꿔가며 학습 속도와 번역 품질 변화를 관찰해보세요.")
# (BLEU 스코어 계산은 sacrebleu 라이브러리를 사용해 구현할 수 있습니다)
print("Script finished.")
