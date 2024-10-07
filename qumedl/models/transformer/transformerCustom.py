import math
import torch
from torch import nn


# Custom Causal Self-Attention
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_length: int,
        dropout_rate: float,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.attention_proj = nn.Linear(d_model, d_model * 3, bias=bias)
        self.output_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.residual_dropout = nn.Dropout(dropout_rate)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_length, max_seq_length)).view(
                1, 1, max_seq_length, max_seq_length
            ),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        query, key, value = self.attention_proj(x).split(self.d_model, dim=2)

        query = query.view(
            batch_size, seq_len, self.n_heads, self.d_model // self.n_heads
        ).transpose(1, 2)
        key = key.view(
            batch_size, seq_len, self.n_heads, self.d_model // self.n_heads
        ).transpose(1, 2)
        value = value.view(
            batch_size, seq_len, self.n_heads, self.d_model // self.n_heads
        ).transpose(1, 2)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(
            self.d_model // self.n_heads
        )
        attention_scores = attention_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context = attention_probs @ value
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        return self.residual_dropout(self.output_proj(context))


# Custom Feedforward Network
class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


# Custom Transformer Block
class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, max_seq_length: int, dropout_rate: float
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_length, dropout_rate)
        self.ffn = FeedForward(d_model, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# Modified Transformer Model
class CausalMolTransformerCustom(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        model_dim: int,
        n_attn_heads: int,
        n_encoder_layers: int,
        dropout: float,
        padding_token_idx: int,
        max_seq_length: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_token_idx
        )
        self.max_seq_length = max_seq_length
        self.d_model = model_dim
        self.padding_token_idx = padding_token_idx

        # Positional Encoding (Manual Calculation)
        self.pos_encoding = self.create_positional_encoding(max_seq_length, model_dim)

        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout)

        # Transformer Layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(model_dim, n_attn_heads, max_seq_length, dropout)
                for _ in range(n_encoder_layers)
            ]
        )

        self.norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, vocab_size, bias=False)

    def create_positional_encoding(self, max_seq_length: int, model_dim: int):
        positions = torch.arange(max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )
        pos_encoding = torch.zeros(max_seq_length, model_dim)
        pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        pos_encoding[:, 1::2] = torch.cos(positions * div_term)
        return pos_encoding.unsqueeze(0)

    def forward(self, inputs: torch.Tensor):
        seq_length = inputs.size(1)
        embedding = self.embedding(inputs)  # Get token embeddings
        pos_embedding = self.pos_encoding[:, :seq_length, :].to(
            embedding.device
        )  # Get positional encodings
        x = self.dropout(embedding + pos_embedding)  # Apply dropout to input

        for layer in self.layers:
            x = layer(x)  # Pass through transformer layers

        x = self.norm(x)  # Final layer normalization
        logits = self.output_proj(x)  # Project to vocab size
        return logits

    def generate(
        self, inputs: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ):
        outputs = inputs
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(outputs)
                logits = logits[:, -1, :]  # Get logits for the last token
                probs = torch.distributions.Categorical(logits=logits / temperature)
                next_token = probs.sample()
                outputs = torch.cat((outputs, next_token.unsqueeze(1)), dim=1)
        return outputs


# Loss function and other components remain the same
