from typing import Dict, Optional, Tuple

import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        embedding_dim: int,
        lstm_hidden_size: int,
        n_layers: int,
        padding_token_index: int | None,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        """A PyTorch implementation of a LSTM-based model for next-token-prediction tasks.

        Args:
            input_vocab_size (int): number of tokens in the input vocabulary. This shouldn't include tokens like END.
            output_vocab_size (int): number of tokens in the output vocabulary. This shouldn't include tokens like PAD or START.
            embedding_dim (int): dimension of the embedding layer.
            lstm_hidden_size (int): dimension of the hidden state of the LSTM.
            n_layers (int): number of layers in the LSTM.
            padding_token_index (int, optional): index of the padding token in the vocabulary.
            dropout (float, optional): dropout probability. Defaults to 0.0.
                Has no effect if ``n_layers`` is 1.
            temperature (float, optional): temperature for sampling. Defaults to 1.0.
                Lower values lead to more conservative sampling, while higher values lead to more exploratory sampling.
        """
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self._D = 1
        self.padding_token_index = padding_token_index

        self._sampling_temperature = temperature

        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(
            input_vocab_size, embedding_dim, padding_idx=padding_token_index
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )

        self.output_head = nn.Linear(self.lstm_hidden_size, self.output_vocab_size)

    def _transpose_state(
        self, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transpose the hidden state and cell state tensors to be compatible with ``torch.nn.LSTM`` or ``torch.nn.DataParallel``,
        depending on the starting shape of the state tensors.

        ``torch.nn.LSTM`` expects the hidden state and cell state tensors to have shape (n_layers, batch_size, hidden_size).
        However, the input tensors to ``torch.nn.DataParallel`` are expected to have shape (batch_size, n_layers, hidden_size),
        and are returned in this format. Therefore, to ensure that outputs of ``torch.nn.DataParallel`` are compatible with ``torch.nn.LSTM``,
        and vice versa, we transpose the hidden state and cell state tensors.

        If ``state`` initially has shape (n_layers, batch_size, hidden_size), then we transpose to (batch_size, n_layers, hidden_size).
        If ``state`` initially has shape (batch_size, n_layers, hidden_size), then we transpose to (n_layers, batch_size, hidden_size).
        """
        h, c = state
        h = h.transpose_(0, 1).contiguous()
        c = c.transpose_(0, 1).contiguous()
        return h, c

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network.

        Args:
            inputs (torch.Tensor): tensor of shape (batch_size, seq_length) and dtype torch.long
            initial_state (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): tuple of tensors giving the last hidden state and cell state of the LSTM.
                Each of these tensors have shape (batch_size, n_lstm_hidden_layers, lstm_hidden_size). Defaults to None.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: tuple of tensors,
                * output of the LSTM of shape (batch_size, seq_length, vocab_size)
                * last hidden state and cell state of the LSTM. Each of these tensors have shape (batch_size, n_lstm_layers, lstm_hidden_size).
        """
        # (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
        embedded_inputs = self.embedding(inputs)

        if initial_state is not None:
            # to make outputs of torch.nn.DataParallel compatible with torch.nn.LSTM
            # (batch_size, n_layers, hidden_size), (n_layers, batch_size, hidden_size)
            initial_state = self._transpose_state(initial_state)

        # (batch_size, sequence_length, embedding_dim) -> (batch_size, sequence_length, hidden_size)
        output_seq, state = self.lstm(embedded_inputs, initial_state)

        # to make outputs of torch.nn.LSTM compatible with torch.nn.DataParallel
        # (n_layers, batch_size, hidden_size), (batch_size, n_layers, hidden_size)
        state = self._transpose_state(state)

        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, vocab_size)
        logits = self.output_head(output_seq)

        return logits, state

    def hidden_state_shape(self, batch_size: int) -> Tuple[int, int, int]:
        return (batch_size, self.n_layers, self.lstm.hidden_size)

    @property
    def hidden_size(self) -> int:
        return self.lstm_hidden_size

    @property
    def sampling_temperature(self) -> float:
        return self._sampling_temperature

    def generate(
        self,
        start_inputs: torch.Tensor,
        seq_len: int,
        random_seed: Optional[int] = None,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """Generate samples from the underlying model and return the raw form along with the
        conditional probabilities of each of the sequences.

        Args:
            start_inputs (torch.Tensor): a tensor of "start" tokens of shape (n_samples, 1).
            seq_len (int): the desired maximum length of the generated sequences.
            random_seed (Optional[int], optional): an optional random seed for reproducibility. Defaults to None.
            return_probs (bool, optional): whether to return the conditional probabilities of each of the sequences. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the raw generated sequences and the associated probabilities.
        """

        random_generator = None
        if random_seed:
            random_generator = torch.Generator()
            random_generator.manual_seed(random_seed)

        # (batch_size, 1)
        inputs = start_inputs
        n_samples = start_inputs.shape[0]

        # (n_layers, batch_size, lstm_hidden_size)
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # (batch_size, seq_len)
        outputs = torch.zeros((n_samples, seq_len))
        probabilities = torch.zeros((n_samples, seq_len, self.output_vocab_size))

        with torch.no_grad():
            for index in range(0, seq_len):
                # (batch_size, 1) -> (batch_size, 1, vocab_size)
                logits, hidden_state = self(inputs, hidden_state)

                # (batch_size, 1, vocab_size) -> (batch_size, vocab_size)
                logits = logits.squeeze(1)

                # (batch_size, vocab_size) -> (batch_size, vocab_size)
                prob_dist = torch.distributions.Categorical(
                    logits=logits / self.sampling_temperature
                )

                # (batch_size, vocab_size) -> (batch_size, )
                sampled_token_indices = prob_dist.sample()

                outputs[:, index] = sampled_token_indices
                probabilities[:, index, :] = prob_dist.probs

                # (batch_size, 1)
                inputs = sampled_token_indices.unsqueeze(1)

        if return_probs:
            return outputs, probabilities

        return outputs

    @property
    def config(self) -> Dict:
        """Returns model configuration."""
        d = {
            "input_vocab_size": self.input_vocab_size,
            "output_vocab_size": self.output_vocab_size,
            "embedding_dim": self.embedding_dim,
            "latent_dim": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "bidirectional": True if self._D == 2 else False,
            "padding_token_index": self.padding_token_index,
        }

        return d
