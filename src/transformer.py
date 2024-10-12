import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.transpose(0, 1)
    def forward(self, x):
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hidden, num_heads, seq_len, d_k) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d_k = torch.tensor(d_k)

        self.W_q = nn.Linear(num_hidden, 2 * num_heads * num_hidden)
        self.W_k = nn.Linear(num_hidden, 2 * num_heads * num_hidden)
        self.W_v = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_o = nn.Linear(num_heads * num_hidden, num_hidden)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.mask = self.get_mask(self.seq_len)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=num_heads)
        self._lambda_init = torch.rand(1)
        self._lambda = nn.Parameter(self._lambda_init.clone())
    
    def get_mask(self, size):
        device = next(self.parameters()).device
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)  
        return mask.unsqueeze(0).unsqueeze(0)  

    def forward(self, query, key, values, dropout=0.1, mask=None):
        query = self.W_q(query).view(-1, self.num_heads, self.seq_len, 2 * self.num_hidden)
        key = self.W_k(key).view(-1, self.num_heads, self.seq_len, 2 * self.num_hidden)
        values = self.W_v(values).view(-1, self.num_heads, self.seq_len, self.num_hidden)

        #split query into [q1;q2] and same for keys [k1;k2]
        query_1 = query[:, :, :, :self.num_hidden]
        query_2 = query[:, :, :, self.num_hidden:]

        key_1 = key[:, :, :, :self.num_hidden]
        key_2 = key[:, :, :, self.num_hidden:]

        QK_T_1 = torch.matmul(query_1, key_1.mT) / torch.sqrt(self.d_k)
        QK_T_2 = torch.matmul(query_2, key_2.mT) / torch.sqrt(self.d_k)

        QK_T_1_norm = self.softmax(QK_T_1)
        QK_T_2_norm = self.softmax(QK_T_2)

        #eq 1
        attention_scores = (QK_T_1_norm - self._lambda * QK_T_2_norm)

        if mask:
            self.mask = self.mask.to(query.device)
            attention_scores = attention_scores.masked_fill(self.mask == 1, float('-inf'))

        attention_scores = self.dropout(attention_scores) 
        output = torch.matmul(attention_scores, values)  
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads , self.seq_len, self.num_hidden)  
        
        output = self.group_norm(output)
        
        output = output * (1 - self._lambda_init)
        output = torch.cat([output[:, i, :, :] for i in range(self.num_heads)], dim=-1)

        output = self.W_o(output)  
        return output  

class FeedForward(nn.Module):
    def __init__(self, num_hidden, num_ffn_hidden) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_ffn_hidden = num_ffn_hidden

        self.W_1 = nn.Linear(num_hidden, num_ffn_hidden)
        self.W_2 = nn.Linear(num_ffn_hidden, num_hidden)

    def forward(self, x):
        return self.W_2(F.relu(self.W_1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, num_hidden, num_heads, seq_len) -> None:
        super().__init__()
        self.multihead_attention = MultiHeadAttention(num_hidden=num_hidden, num_heads=num_heads, seq_len=seq_len, d_k=1)
        self.feed_forward = FeedForward(num_hidden=num_hidden, num_ffn_hidden=2 * num_hidden)
        self.layer_norm1 = nn.LayerNorm(num_hidden)
        self.layer_norm2 = nn.LayerNorm(num_hidden)
    
    def forward(self, input_with_pos):
        x = self.multihead_attention(input_with_pos, input_with_pos, input_with_pos)
        
        x = x + input_with_pos
        x = self.layer_norm1(x)

        x_final = self.feed_forward(x)
        x = x + x_final

        x = self.layer_norm2(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, n_heads, seq_len, num_hidden) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.decoders = nn.ModuleList([TransformerDecoderLayer(num_hidden, n_heads, seq_len) for i in range(num_layers)])

    def forward(self, x, encoder_output):
        for layer in self.decoders:
            x = layer(x, encoder_output)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, n_heads, seq_len, num_hidden) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.encoders = nn.ModuleList([TransformerEncoderLayer(num_hidden, n_heads, seq_len) for i in range(num_layers)])
    def forward(self, x):
        for layer in self.encoders:
            x = layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, num_hidden, num_heads, seq_len) -> None:
        super().__init__()
        self.multihead_attention_masked = MultiHeadAttention(num_hidden=num_hidden, num_heads=num_heads, seq_len=seq_len, d_k=1)
        self.multihead_attention = MultiHeadAttention(num_hidden=num_hidden, num_heads=num_heads, seq_len=seq_len, d_k=1)
        
        self.feed_forward = FeedForward(num_hidden=num_hidden, num_ffn_hidden= 2 * num_hidden)
        self.layer_norm1 = nn.LayerNorm(num_hidden)
        self.layer_norm2 = nn.LayerNorm(num_hidden)
        self.layer_norm3 = nn.LayerNorm(num_hidden)
    
    def forward(self, output_with_pos, encoder_output):
        x = self.multihead_attention_masked(output_with_pos, output_with_pos, output_with_pos, mask=True)

        x = x + output_with_pos
        x = self.layer_norm1(x)

        x_attention = self.multihead_attention(encoder_output, encoder_output, x)

        x = x + x_attention
        x = self.layer_norm2(x)

        x_forward = self.feed_forward(x)

        x = x + x_forward
        x = self.layer_norm3(x)
        return x

class Transformer(nn.Module):
    def __init__(self, encoder_layers_num, decoder_layers_num, num_hidden, num_heads, seq_len, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(encoder_layers_num, num_heads, seq_len, num_hidden)
        self.decoder = TransformerDecoder(decoder_layers_num, num_heads, seq_len, num_hidden)
        self.pos_enc = PositionalEncoding(embedding_dim, max_len=seq_len)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.embedding(x)
        y = self.embedding(y)

        x = self.pos_enc(x)
        y = self.pos_enc(y)

        enc_output = self.encoder(x)
        dec_output = self.decoder(y, enc_output)
        output = self.linear(dec_output)

        return output
