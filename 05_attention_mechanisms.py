# =============================================
# [5/9] 어텐션 기법①: Bahdanau & Luong
# =============================================
# 목표: Seq2Seq의 성능을 획기적으로 개선한 두 가지 주요 어텐션 메커니즘을 구현하고 비교합니다.

# --- 1. 기본 설정 (Seq2Seq 단계와 유사) ---
# !pip install torch torchtext transformers datasets spacy
# !python -m spacy download en_core_web_sm
# !python -m spacy download fr_core_news_sm

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import spacy
import random # Though not used in this script, often part of Seq2Seq training
from collections import Counter

print("--- Data and Vocab Setup (from 04_seq2seq_basics.py) ---")
print("Loading iwslt2017 dataset (en-fr)...")
# Using a small subset for faster processing in script conversion context
raw_datasets = load_dataset("iwslt2017", "iwslt2017-en-fr", split="train[:1%]")

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

class Vocab:
    def __init__(self, tokenizer, min_freq=2):
        self.tokenizer = tokenizer
        self.min_freq = min_freq
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        
    def build_vocab(self, sentences):
        counter = Counter()
        for sentence in sentences:
            counter.update(self.tokenizer(sentence))
        for word, count in counter.items():
            if count >= self.min_freq:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx
        print(f"Built vocabulary of size: {len(self.itos)}")

SRC_VOCAB = Vocab(tokenize_en)
TRG_VOCAB = Vocab(tokenize_fr)
src_sentences = [d['translation']['en'] for d in raw_datasets]
trg_sentences = [d['translation']['fr'] for d in raw_datasets]
SRC_VOCAB.build_vocab(src_sentences)
TRG_VOCAB.build_vocab(trg_sentences)

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim) # Original uses LSTM

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell # Return outputs for attention

# Hyperparameters (assuming from a typical Seq2Seq setup)
INPUT_DIM = len(SRC_VOCAB.itos)
OUTPUT_DIM = len(TRG_VOCAB.itos)
ENC_EMB_DIM = 128 # Smaller for faster script run, adjust as needed
DEC_EMB_DIM = 128 # Smaller for faster script run, adjust as needed
HID_DIM = 256     # Smaller for faster script run, adjust as needed
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- 2. Bahdanau (Additive) Attention 모듈 구현 ---
print("\n--- Defining Bahdanau Attention --- ")
class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn_W1 = nn.Linear(hid_dim, hid_dim)       # For decoder_hidden
        self.attn_W2 = nn.Linear(hid_dim, hid_dim)       # For encoder_outputs
        self.attn_v = nn.Linear(hid_dim, 1, bias=False)  # To compute the score

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, hid_dim] (previous decoder hidden state s_{t-1})
        # encoder_outputs: [src_len, batch_size, hid_dim] (encoder hidden states h_s)
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        # decoder_hidden becomes [batch_size, src_len, hid_dim]
        decoder_hidden_repeated = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Permute encoder_outputs to [batch_size, src_len, hid_dim]
        encoder_outputs_permuted = encoder_outputs.permute(1, 0, 2)

        # Calculate energy: score(s_{t-1}, h_s) = v_a^T tanh(W_a s_{t-1} + U_a h_s)
        # self.attn_W1(decoder_hidden_repeated) is W_a s_{t-1}
        # self.attn_W2(encoder_outputs_permuted) is U_a h_s
        energy = torch.tanh(self.attn_W1(decoder_hidden_repeated) + self.attn_W2(encoder_outputs_permuted))
        # energy shape: [batch_size, src_len, hid_dim]
        
        # Get attention scores (alpha_t)
        # attention_scores shape: [batch_size, src_len]
        attention_scores = self.attn_v(energy).squeeze(2) 
        
        return F.softmax(attention_scores, dim=1)

# --- 3. 어텐션을 적용한 Decoder 구현 ---
print("--- Defining Attention Decoder --- ")
class AttnDecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # GRU input: concatenated (embedding + context_vector)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim) 
        self.fc_out = nn.Linear(hid_dim, output_dim) # From GRU hidden to output vocab

    def forward(self, input_token, decoder_hidden, encoder_outputs):
        # input_token: [batch_size] (current input token y_t)
        # decoder_hidden: [batch_size, hid_dim] (previous decoder hidden state s_{t-1})
        # encoder_outputs: [src_len, batch_size, hid_dim] (encoder hidden states h_s)
        
        input_token = input_token.unsqueeze(0) # [1, batch_size]
        embedded = self.embedding(input_token).squeeze(0) # [batch_size, emb_dim]

        # Calculate attention weights (alpha_t)
        # decoder_hidden is s_{t-1}
        a = self.attention(decoder_hidden, encoder_outputs) # [batch_size, src_len]
        a = a.unsqueeze(1) # [batch_size, 1, src_len] for bmm
        
        encoder_outputs_permuted = encoder_outputs.permute(1, 0, 2) # [batch_size, src_len, hid_dim]
        
        # Calculate context vector (c_t)
        context_vector = torch.bmm(a, encoder_outputs_permuted).squeeze(1) # [batch_size, hid_dim]
        
        # Prepare RNN input: concatenate embedded input token and context vector
        # rnn_input = [y_t, c_t]
        rnn_input = torch.cat((embedded, context_vector), dim=1) # [batch_size, emb_dim + hid_dim]
        
        # Pass through GRU
        # rnn_input needs to be [1, batch_size, emb_dim + hid_dim] for GRU
        # decoder_hidden (s_{t-1}) needs to be [1, batch_size, hid_dim] for GRU
        output, hidden = self.rnn(rnn_input.unsqueeze(0), decoder_hidden.unsqueeze(0))
        # output: [1, batch_size, hid_dim] (new output o_t)
        # hidden: [1, batch_size, hid_dim] (new hidden state s_t)
        
        # Prediction
        prediction = self.fc_out(output.squeeze(0)) # [batch_size, output_dim]
        return prediction, hidden.squeeze(0) # Return prediction and new hidden state

# --- 4. 모델 래핑 및 과제 안내 ---
# (실제 학습을 위해서는 Seq2Seq 모델도 어텐션을 사용하도록 수정해야 합니다.)
# Example (commented out):
# enc = EncoderRNN(INPUT_DIM, ENC_EMB_DIM, HID_DIM).to(DEVICE)
# attn = BahdanauAttention(HID_DIM).to(DEVICE)
# dec = AttnDecoderRNN(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attn).to(DEVICE)
# # model = Seq2Seq(enc, dec, DEVICE) # Seq2Seq 클래스도 수정 필요 (to handle encoder_outputs and pass to decoder)

print("\nBahdanau 어텐션 모듈과 이를 사용한 Decoder가 정의되었습니다.")
print("\n[실습 과제]")
print("1. Luong Attention (dot, general)을 별도 모듈로 구현해보세요.")
print("   - Dot: score(h_t, h_s) = h_t^T * h_s")
print("   - General: score(h_t, h_s) = h_t^T * W_a * h_s")
print("2. 어텐션을 적용했을 때와 아닐 때의 번역 품질(BLEU 스코어)을 비교해보세요.")
print("Script finished.")
