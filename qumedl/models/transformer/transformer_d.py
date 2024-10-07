import torch
from torch import nn

from qumedl.models.layers import PositionalEncoding
from .utils import create_lookahead_mask, create_padding_mask


class CausalMolTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        model_dim: int,
        n_attn_heads: int,
        n_encoder_layers: int,
        hidden_act: nn.Module,
        dropout: float,
        padding_token_idx: int,
    ) -> None:
        super().__init__()

        self._vocab_size = vocab_size

        # inputs to embedding layer are [START] + [PAD] + Vocab tokens
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim=embedding_dim, padding_idx=padding_token_idx
        )
        self.padding_token_idx = padding_token_idx

        self.pe = PositionalEncoding(d_model=embedding_dim, dropout=dropout)

        self.projection = nn.Linear(embedding_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_attn_heads,
            dim_feedforward=model_dim,
            dropout=dropout,
            activation=hidden_act,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers
        )

        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        inputs: torch.Tensor,
        mask_attention: bool = False,
        mask_pad: bool = False,
        return_last_hidden_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input tensor of shape (b, seq_len)
        """
        # (b, seq_len) -> (b, seq_len, emb_dim)
        embed = self.embedding(inputs)

        # (b, seq_len, emb_dim) ->  (b, seq_len, emb_dim)
        embed = self.pe(embed)

        # (b, seq_len, emb_dim) -> (b, seq_len, d_model)
        proj = self.projection(embed)

        attn_mask = None
        pad_mask = None

        if mask_attention:
            attn_mask = create_lookahead_mask(inputs, device=inputs.device)

        if mask_pad:
            pad_mask = create_padding_mask(
                inputs, self.padding_token_idx, device=inputs.device
            )

        # (b, seq_len, d_model) -> (b, seq_len, d_model)
        last_hidden_state = self.encoder(
            proj,
            mask=attn_mask,
            src_key_padding_mask=pad_mask,
            is_causal=True if mask_attention else False,
        )

        # (b, seq_len, d_model) -> (b, seq_len, vocab_size)
        logits = self.lm_head(last_hidden_state)

        if return_last_hidden_state:
            return logits, last_hidden_state

        return logits

    def generate(
        self,
        inputs: torch.tensor,
        max_new_tokens: int,
        random_seed: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples from the underlying model and return the raw form along with the
        conditional probabilities of each of the sequences.

        Args:
            n_samples (int): then number of samples to generate.
            max_new_tokens (int): the maximum number of tokens to generate.
            random_seed (Optional[int], optional): an optional random seed for reproducibility. Defaults to None.
            temperature (float, optional): temperature to use when sampling from the logits. Defaults to 1.0.

        Returns:
            torch.Tensor: the raw generated sequences.
        """

        random_generator = None

        if random_seed:
            random_generator = torch.Generator()
            random_generator.manual_seed(random_seed)

        outputs = inputs

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # logits -> (batch_size, ?<seq_len>, vocab_size)
                logits = self(outputs, mask_pad=True)

                # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size)
                logits = logits[:, -1, :]

                cat_distribution = torch.distributions.Categorical(
                    logits=logits / temperature
                )

                # sampled_token_indices -> (batch_size, )
                sampled_token_indices = cat_distribution.sample()

                outputs = torch.cat(
                    (outputs, sampled_token_indices.unsqueeze(1)), dim=1
                )

        return outputs

        # cat_distribution = torch.distributions.Categorical(
        #     logits=logits / self._sampling_temperature
        # )
        #         sampled_token_indices = cat_distribution.sample()
        #         samples = torch.cat(
        #             [samples, sampled_token_indices.unsqueeze(1)], dim=1
        #         )
        # self.set_train_state()
        # return samples[:, 1:]  # Remove the start of sentence token
