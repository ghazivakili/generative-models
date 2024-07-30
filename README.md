# Qu-MeDL: Quantum Assisted Deep Learning for Medical Applications

## Installation

Inside of a python virtual environment (e.g. Conda) run
```shell
pip install -e ".[dev]"
```




```bash 
                ┌─────────────────────────────────────┐
                │          Prior Generation           │
                │        RBM/QBAG (prior samples)     │
                └──────────────────┬──────────────────┘
                                   │
                                   v
                ┌─────────────────────────────────────┐
                │                Inputs               │
                │        (token indices, prior)       │
                └──────────────────┬──────────────────┘
                                   │
                                   v
                ┌─────────────────────────────────────┐
                │          Embedding Layer            │
                │      nn.Embedding (vocab_size)      │
                └──────────────────┬──────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │         Prior Opration/Add          │
                │       ProjectXAddY, Multiplication  |
                |               or Concatenate        │
                └──────────────────┬──────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │         Positional Encoding         │
                │        PositionalEncoding           │
                └──────────────────┬──────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │          Linear Projection          │
                │             nn.Linear               │
                └──────────────────┬──────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │        Transformer Encoder          │
                │ nn.TransformerEncoder (Layers)      │
                └──────────────────┬──────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │            Output Layer             │
                │          nn.Linear (lm_head)        │
                └──────────────────┬──────────────────┘
                                   │
                                   v
                ┌─────────────────────────────────────┐
                │                Logits               │
                │      (batch_size, seq_len, vocab)   │
                └─────────────────────────────────────┘
                                   │
                                   v
                ┌─────────────────────────────────────┐
                │           Generate Tokens           │
                │    (iteratively appends new tokens  │
                │               to the input)         │
                └─────────────────────────────────────┘
                
                ┌─────────────────────────────────────┐
                │         Transformer Configuration   │
                │    vocab_size: int                  │
                │    embedding_dim: int               │
                │    prior_dim: int                   │
                │    model_dim: int                   │
                │    n_attn_heads: int                │
                │    n_encoder_layers: int            │
                │    hidden_act: nn.Module            │
                │    dropout: float                   │
                │    padding_token_idx: int           │
                └─────────────────────────────────────┘

                ┌─────────────────────────────────────┐
                │          Transformer Type           │
                │  Molecular Prior-assisted Transformer│
                │             (CausalMolPAT)          │
                └─────────────────────────────────────┘


```

More general 

model_dim = embedding_dim = 256 # should be embedding_dim/n_attn_heads


```bash 
               +---------------------------------+
               |         Prior Generation        |
               |           (RBM/QBAG)            |
               +--------------+------------------+
                              |
                              v
               +-------------------------------+
               |            Inputs             |
               |  (Token Indices, Prior Samples)|
               +--------------+----------------+
                              |
                              v
               +-------------------------------+
               |        Embedding Layer        |
               |        (nn.Embedding)         |
               +--------------+----------------+
                              |
                              v
               +-------------------------------+
               |  Prior Opration/Addition or |
               |        Concatenation          |
               |   (ProjectXAddY/Concatenate)  |
               +--------------+----------------+
                              |
                              v
               +-------------------------------+
               |      Positional Encoding      |
               |    (PositionalEncoding)       |
               +--------------+----------------+
                              |
                              v
               +-------------------------------+
               |       Linear Projection       |
               |         (nn.Linear)           |
               +--------------+----------------+
                              |
                              v
               +-------------------------------+
               |     Transformer Encoder       |
               | (nn.TransformerEncoder Layers)|
               +--------------+----------------+
                              |
                              v
               +-------------------------------+
               |         Output Layer          |
               |        (nn.Linear)            |
               +--------------+----------------+
                              |
                              v
               +-------------------------------+
               |      Generate Function        |
               |  (Iterative Token Generation) |
               +-------------------------------+


``` 


