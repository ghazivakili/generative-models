import torch
from torch import nn
from torch.nn import Module

from qumedl.models.layers import Concatenate, PositionalEncoding, Repeat, ProjectXAddY
from .utils import create_lookahead_mask, create_padding_mask


class CausalMolPAT(nn.Module):
    """Molecular Prior-assisted Transformer. Adds prior samples to token embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        prior_dim: int,
        model_dim: int,
        n_attn_heads: int,
        n_encoder_layers: int,
        hidden_act: Module,
        dropout: float,
        padding_token_idx: int,
    ) -> None:
        """Initialize the model.

        Args:
            vocab_size (int): the number of tokens in the vocabulary.
            embedding_dim (int): dimension of the token embeddings.
            prior_dim (int): dimension of the prior samples.
            model_dim (int): dimension of the model. Can usually be the same as embedding_dim.
            n_attn_heads (int): number of attention heads. More attention heads can improve model performance but
                is more computationally expensive.
            n_encoder_layers (int): number of encoder layers. More layers can improve model performance but
                is more computationally expensive.
            hidden_act (Module): activation function to use in the feedforward layers.
            dropout (float): dropout rate. Recommended to be a value between 0.1 and 0.3. and not exceed 0.5.
            padding_token_idx (int): index of the padding token in the vocabulary.

        Please see ``forward`` and ``generate`` for more details on generating sequences using the model.
        """
        super().__init__()

        self._vocab_size = vocab_size
        self.prior_dim = prior_dim
        self.padding_token_idx = padding_token_idx

        # inputs to embedding layer are [START] + [PAD] + Vocab tokens
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim=embedding_dim, padding_idx=padding_token_idx
        )

        # to decouple dimension of prior samples and embedding dimension
        # we project prior samples to the embedding dimension
        self.projectx_addy = ProjectXAddY(prior_dim, target_dim=embedding_dim)

        # placeholder for visibility
        self.repeat: Module = nn.Identity()

        # expand positional encoding to add positional info to samples from prior
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
        prior_samples: torch.FloatTensor,
        mask_attention: bool = False,
        mask_pad: bool = False,
        return_last_hidden_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Performs forward pass through the model.

        Args:
            x (torch.Tensor): input sequence of token indices with shape (batch_size, seq_len) and dtype:int64.
            prior_samples (torch.Tensor): samples from a prior distribution with shape (batch_size, sample_dim) and dtype:float32.
            mask_attention (bool): wether to create an look-ahead attention mask, ensuring "future" tokens are not attended to.
            mask_pad (bool): wether to create a padding mask, ensuring padding tokens are not attended to.
            return_last_hidden_state (bool): wether to return the hidden state of the tensors returned from the last encoder block.

        Returns:
            torch.Tensor: output logits with shape (batch_size, seq_len, n_tokens) and dtype:float32,
                and (optionally) the output of the last encoder block.
        """
        attn_mask = None
        pad_mask = None

        if mask_attention:
            attn_mask = create_lookahead_mask(inputs, device=inputs.device).bool()

        if mask_pad:
            pad_mask = create_padding_mask(
                inputs, self.padding_token_idx, device=inputs.device
            )

        # x: (batch_size, seq_len), prior_samples: (batch_size, sample_dim)

        # x_emb: (batch_size, seq_len, embedding_dim)
        x_emb = self.embedding(inputs)

        self.repeat = Repeat("b o d -> b (repeat o) d", repeat=x_emb.shape[1])

        # x_emb -> (batch_size, seq_len, embedding_dim)
        # project repeated samples to embedding dimension and add them to x_emb
        x_emb = self.projectx_addy(self.repeat(prior_samples.unsqueeze(1)), x_emb)

        # x_emb: (batch_size, seq_len, sample_dim + embedding_dim)
        x_emb = self.pe(x_emb)

        # x_emb: (batch_size, seq_len, model_dim)
        x_emb = self.projection(x_emb)

        # x: (batch_size, seq_len, n_tokens)
        encoder_output = self.encoder(
            x_emb, src_key_padding_mask=pad_mask, mask=attn_mask
        )
        logits = self.lm_head(encoder_output)

        if return_last_hidden_state:
            return logits, encoder_output

        return logits

    def generate(
        self,
        inputs: torch.Tensor,
        prior_samples: torch.Tensor,
        max_new_tokens: int,
        random_seed: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples from the underlying model and return the raw form along with the
        conditional probabilities of each of the sequences.

        Args:
            inputs (torch.Tensor): initial inputs to transformer.
            prior_samples (torch.Tensor): samples from prior.
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

        # we will concatenate the generated tokens to the inputs
        outputs = inputs

        # stop_sample_generation_maks = torch.zeros((inputs.shape[0],), device=inputs.device)

        with torch.no_grad():
            for index in range(max_new_tokens):
                # logits -> (batch_size, ?<seq_len>, vocab_size)
                logits = self(outputs, prior_samples=prior_samples, mask_pad=True)

                # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size)
                logits = logits[:, -1, :]

                cat_distribution = torch.distributions.Categorical(
                    logits=logits / temperature
                )

                # sampled_token_indices -> (batch_size, )
                sampled_token_indices = cat_distribution.sample()

                # TODO: see if we want to stop generation at model level or when decoding
                # Sequences which have not previously generated a stop token continue to be updated, otherwise a stop token
                # will be appended instead of the generated token. If stop_token_id is not specified mask will remain all zeros
                # as initialized and condition will yield True for all sequences hence the generated token will be used
                # if stop_token_id is not None:
                #     sampled_token_indices = torch.where(
                #         torch.logical_not(stop_sample_generation_maks),
                #         sampled_token_indices,
                #         other=stop_token_id
                #     )

                #     stop_sample_generation_maks = torch.logical_or(
                #         stop_sample_generation_maks, sampled_token_indices == stop_token_id
                #     )

                outputs = torch.cat(
                    (outputs, sampled_token_indices.unsqueeze(1)), dim=1
                )

        return outputs


class CausalConcatlMolPAT(nn.Module):
    """Molecular Prior-assisted Transformer. Combines prior samples with token embeddings by concatenation."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        prior_dim: int,
        model_dim: int,
        n_attn_heads: int,
        n_encoder_layers: int,
        hidden_act: Module,
        dropout: float,
        padding_token_idx: int,
    ) -> None:
        """Initialize the model.

        Args:
            vocab_size (int): the number of tokens in the vocabulary.
            embedding_dim (int): dimension of the token embeddings.
            prior_dim (int): dimension of the prior samples.
            model_dim (int): dimension of the model. Can usually be the same as embedding_dim.
            n_attn_heads (int): number of attention heads. More attention heads can improve model performance but
                is more computationally expensive.
            n_encoder_layers (int): number of encoder layers. More layers can improve model performance but
                is more computationally expensive.
            hidden_act (Module): activation function to use in the feedforward layers.
            dropout (float): dropout rate. Recommended to be a value between 0.1 and 0.3. and not exceed 0.5.
            padding_token_idx (int): index of the padding token in the vocabulary.

        Please see ``forward`` and ``generate`` for more details on generating sequences using the model.
        """
        super().__init__()

        self._vocab_size = vocab_size
        self.prior_dim = prior_dim
        self.padding_token_idx = padding_token_idx

        # inputs to embedding layer are [START] + [PAD] + Vocab tokens
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim=embedding_dim, padding_idx=padding_token_idx
        )
        self.concatenate = Concatenate()

        # placeholder for visibility
        self.repeat: Module = nn.Identity()

        # expand positional encoding to add positional info to samples from prior
        self.pe = PositionalEncoding(d_model=embedding_dim + prior_dim, dropout=dropout)

        self.projection = nn.Linear(embedding_dim + prior_dim, model_dim)

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
        prior_samples: torch.FloatTensor,
        mask_attention: bool = False,
        mask_pad: bool = False,
        return_last_hidden_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Performs forward pass through the model.

        Args:
            x (torch.Tensor): input sequence of token indices with shape (batch_size, seq_len) and dtype:int64.
            prior_samples (torch.Tensor): samples from a prior distribution with shape (batch_size, sample_dim) and dtype:float32.
            mask_attention (bool): wether to create an look-ahead attention mask, ensuring "future" tokens are not attended to.
            mask_pad (bool): wether to create a padding mask, ensuring padding tokens are not attended to.
            return_last_hidden_state (bool): wether to return the hidden state of the tensors returned from the last encoder block.

        Returns:
            torch.Tensor: output logits with shape (batch_size, seq_len, n_tokens) and dtype:float32,
                and (optionally) the output of the last encoder block.
        """
        attn_mask = None
        pad_mask = None

        if mask_attention:
            attn_mask = create_lookahead_mask(inputs, device=inputs.device).bool()

        if mask_pad:
            pad_mask = create_padding_mask(
                inputs, self.padding_token_idx, device=inputs.device
            )

        # x: (batch_size, seq_len), prior_samples: (batch_size, sample_dim)

        # x_emb: (batch_size, seq_len, embedding_dim)
        x_emb = self.embedding(inputs)

        # x_emb: (batch_size, seq_len, sample_dim + embedding_dim)
        self.repeat = Repeat("b o d -> b (repeat o) d", repeat=x_emb.shape[1])
        x_emb = self.concatenate(x_emb, self.repeat(prior_samples.unsqueeze(1)))

        # x_emb: (batch_size, seq_len, sample_dim + embedding_dim)
        x_emb = self.pe(x_emb)

        # x_emb: (batch_size, seq_len, model_dim)
        x_emb = self.projection(x_emb)

        # x: (batch_size, seq_len, n_tokens)
        encoder_output = self.encoder(
            x_emb, src_key_padding_mask=pad_mask, mask=attn_mask
        )
        logits = self.lm_head(encoder_output)

        if return_last_hidden_state:
            return logits, encoder_output

        return logits

    def generate(
        self,
        inputs: torch.Tensor,
        prior_samples: torch.Tensor,
        max_new_tokens: int,
        random_seed: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples from the underlying model and return the raw form along with the
        conditional probabilities of each of the sequences.

        Args:
            inputs (torch.Tensor): initial inputs to transformer.
            prior_samples (torch.Tensor): samples from prior.
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

        # we will concatenate the generated tokens to the inputs
        outputs = inputs

        with torch.no_grad():
            for index in range(max_new_tokens):
                # logits -> (batch_size, ?<seq_len>, vocab_size)
                logits = self(outputs, prior_samples=prior_samples, mask_pad=True)

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
