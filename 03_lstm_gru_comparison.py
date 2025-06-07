# =============================================
# [3/9] LSTM & GRU
# =============================================
# 목표: 장기 의존성 문제를 해결한 LSTM, GRU의 성능을 RNN과 비교합니다.

# --- 1. 기본 설정 (이전 단계와 유사) ---
# !pip install torch torchtext transformers datasets scikit-learn matplotlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

# --- 2. 데이터 준비 (02_rnn_classification.ipynb에서 가져옴) ---
print("Loading tokenizer and dataset...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
imdb_dataset = load_dataset("imdb", cache_dir="./.cache")

print("Preparing data subsets...")
# 간단한 실습을 위해 데이터 수를 줄입니다.
train_texts = imdb_dataset['train']['text'][:2000]
train_labels = imdb_dataset['train']['label'][:2000]
test_texts = imdb_dataset['test']['text'][:500]
test_labels = imdb_dataset['test']['label'][:500]

print("Tokenizing and padding data...")
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

print("Creating TensorDatasets...")
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

print("Creating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# --- 3. RNN / LSTM / GRU 모델 정의 ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True) # Default n_layers=1
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        output, hidden = self.rnn(embedded)
        last_hidden = hidden.squeeze(0) # Correct for single layer
        return self.fc(last_hidden)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.lstm(embedded)
        last_hidden = hidden[-1, :, :] # Correct for multi-layer
        return self.fc(last_hidden)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        output, hidden = self.gru(embedded)
        last_hidden = hidden[-1, :, :] # Correct for multi-layer
        return self.fc(last_hidden)

# --- 4. 모델 학습 및 평가 함수 ---
def train_and_evaluate(model_name, model, train_loader, test_loader, epochs=3):
    print(f"\n--- {model_name} Performance ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss/len(train_loader):.4f}")

    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            total_acc += (preds == labels).sum().item()
            total_count += labels.size(0)

    end_time = time.time()
    accuracy = total_acc / total_count
    print(f"Test Accuracy: {accuracy:.4f}, Training time: {end_time - start_time:.2f}s")
    return accuracy

# --- 5. 성능 비교 실험 ---
VOCAB_SIZE = tokenizer.vocab_size
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 2
N_LAYERS = 2 # 레이어 수를 2로 늘려 LSTM/GRU 실험
EPOCHS_TO_TRAIN = 3 # 각 모델별 학습 에폭 수

# RNN 모델 (단일 레이어)
rnn_model = RNNClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
train_and_evaluate("RNN (1 layer)", rnn_model, train_loader, test_loader, epochs=EPOCHS_TO_TRAIN)

# LSTM 모델
lstm_model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS)
train_and_evaluate(f"LSTM ({N_LAYERS} layers)", lstm_model, train_loader, test_loader, epochs=EPOCHS_TO_TRAIN)

# GRU 모델
gru_model = GRUClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS)
train_and_evaluate(f"GRU ({N_LAYERS} layers)", gru_model, train_loader, test_loader, epochs=EPOCHS_TO_TRAIN)

print("\n결론: 일반적으로 LSTM과 GRU가 RNN보다 더 높은 성능을 보이며, 장기 의존성 포착에 유리합니다.")
print("Script finished.")
