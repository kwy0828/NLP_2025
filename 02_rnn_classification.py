# =============================================
# [2/9] 순환 신경망(RNN) & 텍스트 분류
# =============================================
# 목표: RNN의 원리를 이해하고, PyTorch를 이용해 IMDb 감성 분석 모델을 직접 구현합니다.

# --- 1. 기본 설정 및 데이터 준비 ---
# !pip install torch torchtext transformers datasets scikit-learn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- 2. 데이터셋 로드 및 토크나이징 ---
print("Loading tokenizer and dataset...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
imdb_dataset = load_dataset("imdb", cache_dir="./.cache")

# 간단한 실습을 위해 데이터 수를 줄입니다.
print("Preparing data subsets...")
train_texts = imdb_dataset['train']['text'][:2000]
train_labels = imdb_dataset['train']['label'][:2000]
test_texts = imdb_dataset['test']['text'][:500]
test_labels = imdb_dataset['test']['label'][:500]

# 토크나이징 및 패딩
# max_length를 고정하여 모든 시퀀스 길이를 맞춥니다.
print("Tokenizing and padding data...")
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

# TensorDataset으로 변환
print("Creating TensorDatasets...")
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

# DataLoader 생성
print("Creating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# --- 3. RNN 모델 정의 ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # nn.RNN 사용. batch_first=True는 (batch, seq, feature) 입력을 위함
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids) # (batch_size, seq_len, embed_dim)
        # ouput: (batch, seq_len, hidden_dim), hidden: (1, batch, hidden_dim)
        output, hidden = self.rnn(embedded)
        # 마지막 시점의 hidden state를 사용
        last_hidden = hidden.squeeze(0) # (batch, hidden_dim)
        return self.fc(last_hidden)

# 모델 인스턴스화
print("Instantiating model...")
VOCAB_SIZE = tokenizer.vocab_size
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 2 # 긍정(1)/부정(0)

model = RNNClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 4. 학습 및 평가 루프 ---
epochs = 5
history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

print("Starting training...")
for epoch in range(epochs):
    model.train()
    total_loss, total_acc = 0, 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_acc += accuracy_score(labels.cpu(), preds.cpu())
        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)} | Current Avg Loss: {total_loss / (batch_idx + 1):.4f}")

    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc = total_acc / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")

# --- 5. 학습 곡선 시각화 ---
print("Plotting training history...")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# plt.show() # In a script, this might block or require a GUI.
# Consider saving the plot instead if running in a non-interactive environment.
plt.savefig("02_rnn_classification_history.png")
print("Training history plot saved to 02_rnn_classification_history.png")
print("Script finished.")
