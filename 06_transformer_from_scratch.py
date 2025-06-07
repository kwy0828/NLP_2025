# =============================================
# [6/9] 셀프 어텐션(Self-Attention) & Transformer
# =============================================
# 목표: 트랜스포머의 핵심인 멀티헤드 셀프어텐션의 구조를 직접 구현하며 이해합니다.

import torch
import torch.nn as nn
import math

# --- 1. Scaled Dot-Product Attention 구현 ---
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, n_heads, seq_len, d_k)
        # In a typical Transformer, for self-attention in encoder, Q, K, V are same.
        # For encoder-decoder attention, Q is from decoder, K, V are from encoder.
        d_k = K.size(-1) # Dimension of K
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # 마스킹: 어텐션 스코어 행렬에서 마스크 값이 0인 위치를 아주 작은 값(-1e9)으로 치환
            # This prevents attention to certain positions (e.g., padding tokens, future tokens in decoder)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = torch.softmax(scores, dim=-1) # Softmax over the last dimension (keys)
        context = torch.matmul(attention, V) # Weighted sum of V based on attention scores
        return context, attention

# --- 2. Multi-Head Attention 구현 ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads # Dimension of K per head
        self.d_v = d_model // n_heads # Dimension of V per head (usually same as d_k)
        self.d_model = d_model
        
        self.W_Q = nn.Linear(d_model, d_model) # Query projection
        self.W_K = nn.Linear(d_model, d_model) # Key projection
        self.W_V = nn.Linear(d_model, d_model) # Value projection
        self.W_O = nn.Linear(d_model, d_model) # Output projection
        
        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, seq_len, d_model)
        batch_size = Q.size(0)
        
        # 1. Linear projections & split into n_heads
        # Original Q, K, V: (batch_size, seq_len, d_model)
        # After W_Q, W_K, W_V: (batch_size, seq_len, d_model)
        # view and transpose to: (batch_size, n_heads, seq_len, d_k)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # If mask is not None, it needs to be reshaped for multi-head attention
        # mask: (batch_size, 1, seq_len, seq_len) or (batch_size, seq_len, seq_len)
        # It should be broadcastable with scores: (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            mask = mask.unsqueeze(1) # Add head dimension for broadcasting

        # 2. Apply attention on all heads
        # context: (batch_size, n_heads, seq_len, d_v)
        # attn: (batch_size, n_heads, seq_len, seq_len)
        context, attn = self.attention(q_s, k_s, v_s, mask=mask)
        
        # 3. Concat heads and additional linear layer
        # context: (batch_size, seq_len, n_heads * d_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.W_O(context) # (batch_size, seq_len, d_model)
        return output # Note: original notebook returned (output, attn), but typically MHA returns only output

# --- 3. 작은 Transformer 모델 빌드 (개념적 구조) ---
# 실제 모델은 Positional Encoding, FeedForward Network, LayerNorm 등이 추가됩니다.
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src: (batch_size, src_seq_len, d_model)
        # src_mask: (batch_size, src_seq_len, src_seq_len) or (batch_size, 1, src_seq_len, src_seq_len)
        
        # Multi-Head Attention -> Add & Norm
        # In EncoderLayer, Q, K, V for self_attn are all 'src'
        _src = self.self_attn(Q=src, K=src, V=src, mask=src_mask)
        src = self.norm1(src + self.dropout(_src)) # Add & Norm
        
        # FeedForward -> Add & Norm
        _src = self.ffn(src)
        src = self.norm2(src + self.dropout(_src)) # Add & Norm
        return src

print("ScaledDotProductAttention, MultiHeadAttention, EncoderLayer가 정의되었습니다.")
print("\n[실습 과제]")
print("1. PositionalEncoding 클래스를 구현해보세요. (sin, cos 함수 사용)")
print("2. DecoderLayer를 구현해보세요. (Masked Self-Attention + Encoder-Decoder Attention)")
print("3. 이 구성 요소들을 조립해 전체 Transformer 모델을 완성하고, Seq2Seq 과제에 적용해보세요.")
print("Script finished.")
