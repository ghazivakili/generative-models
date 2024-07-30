from enum import Enum
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from ..layers import AddXY, Concatenate, Projection


class CombinationMethod(Enum):
    """Method for combining the embeddings of inputs and prior samples.

    CONCAT - concatenate the embeddings and prior samples. Forming a higher dimensional encoding.
    ADD - add the embeddings and prior samples. Forming an encoding that keeps the same dimensionality as the embeddings.
    """

    CONCAT = "concat"
    ADD = "add"


class PriorSampleFormat(Enum):
    """Format of prior samples.

    VEC - a single prior sample is a 1D vector. Batched samples are 2D tensors.
    VEC_SEQ_2D - a single prior sample is a 2D sequence of vectors. Batched samples are 3D tensors.
    """

    VEC = "vec"
    VEC_SEQ_2D = "vec_seq_2d"


class PriorAssistedLSTM(nn.Module):
    """Base model class used for the PriorAssistedLSTM.

    General flow is as follows:

    Inputs (b, seq_len)                         Prior Samples (b, ?, ?)
            ⬇                                           ⬇
    Embedding (b, seq_len, d0)                   PreprocessPriorSamples (b, seq_len, ?)
                                    ⬇
                            Combine (b, seq_len, ?)
                                    ⬇
                Linear Projection (b, seq_len, lstm_input_dim)
                                    ⬇
                 LSTM (b, seq_len, n_directions * hidden_dim)
                                    ⬇
                    Classifier Head (b, seq_len, output_dim)

    Combine - can be any layer that combines the prior samples with the embeddings. For example adding them or concatenating them.
    PreprocessPriorSamples - can be any layer that pre-processes the prior samples, and ensures that they have the correct shape.
    """

    def _make_projection_block(
        self, combination_method: CombinationMethod, act: nn.Module
    ) -> nn.Module:
        """Creates a projection block (linear + activation) for the appropriate combination method.

        Args:
            combination_method (CombinationMethod): method of combining inputs and prior samples.
            act (nn.Module): activation function for the linear projection layer.

        Raises:
            ValueError: if an invalid combination method is provided.

        Returns:
            nn.Module: projection block [Linear + Activation]
        """
        output_dim = self.lstm.input_size

        config_map = {
            CombinationMethod.CONCAT: {
                "input_dim": self.embedding_dim + self.sample_dim,
            },
            CombinationMethod.ADD: {
                "input_dim": self.embedding_dim,
            },
        }

        selected_config = config_map.get(combination_method)
        if selected_config is None:
            raise ValueError(f"Invalid combination method '{combination_method}'")

        return Projection(
            input_dim=selected_config["input_dim"], output_dim=output_dim, act=act
        )

    def _make_combination_layer(
        self, combination_method: CombinationMethod
    ) -> nn.Module:
        """Creates and returns appropriate combination layer for the given combination method.

        Args:
            combination_method (CombinationMethod): method used to combine the embeddings and prior samples.

        Raises:
            ValueError: if an invalid combination method is provided.

        Returns:
            nn.Module: combination layer.
        """

        combination_layer_map = {
            CombinationMethod.CONCAT: Concatenate(dim=-1),
            CombinationMethod.ADD: AddXY(),
        }

        combination_layer = combination_layer_map.get(combination_method)

        if combination_layer is not None:
            raise ValueError(f"Invalid combination method '{combination_method}'")

        return combination_layer

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        embedding_dim: int,
        lstm_hidden_size: int,
        sample_dim: int,
        n_layers: int,
        start_token_index: int,
        padding_token_index: int,
        dropout: float = 0.0,
        temperature: float = 1.0,
        linear_projection_activation_fn: nn.Module = nn.Identity(),
        combination_method: CombinationMethod = CombinationMethod.CONCAT,
        prior_sample_format: PriorSampleFormat = PriorSampleFormat.VEC,
    ):
        """A PyTorch implementation of a LSTM-based model for next-token-prediction tasks.

        Args:
            input_vocab_size (int): number of tokens in the input vocabulary. This shouldn't include tokens like END.
            output_vocab_size (int): number of tokens in the output vocabulary. This shouldn't include tokens like PAD or START.
            embedding_dim (int): dimension of the embedding layer.
            lstm_hidden_size (int): dimension of the hidden state of the LSTM.
            sample_dim (int): dimension of the prior samples.
            n_layers (int): number of layers in the LSTM.
            combination_layer (nn.Module): layer that combines the prior samples with the embeddings.
            linear_projection (nn.Module): layer that projects the combined embeddings and prior samples to the LSTM input dimension.
            start_token_index (int): index of the START token in the vocabulary.
            padding_token_index (int, optional): index of the padding token in the vocabulary.
            dropout (float, optional): dropout probability. Defaults to 0.0.
                Has no effect if ``n_layers`` is 1.
            temperature (float, optional): temperature for sampling from the conditional distribution. Defaults to 1.0.
            linear_projection_activation_fn (nn.Module, optional): activation function for the linear projection layer that comes before the LSTM block.
                Defaults to nn.Identity().
            combination_method (CombinationMethod, optional): method for combining the embeddings and prior samples. Defaults to CombinationMethod.CONCAT.
            prior_sample_format (bool, optional): format of the prior samples. Defaults to PriorSampleFormat.VEC, meaning that prior samples are vectors.
                Other options are PriorSampleFormat.VEC_SEQ_2D, meaning that prior samples are 2D sequences of vectors.
        """
        super().__init__()

        # TODO: output vocab should include [STOP] token
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.embedding_dim = embedding_dim
        self.sample_dim = sample_dim
        self.n_layers = n_layers

        self.start_token_index = start_token_index
        self.padding_token_index = padding_token_index
        self._sampling_temperature = temperature
        self._combination_method = combination_method
        self._prior_sample_format = prior_sample_format

        self.special_tokens = {
            "start": start_token_index,
            "padding": padding_token_index,
        }

        # model layers
        self.embedding = nn.Embedding(
            input_vocab_size, embedding_dim, padding_idx=padding_token_index
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            bidirectional=False,
            batch_first=True,  # using False helps alleviate headaches with DataParallel and hidden state shapes
            num_layers=n_layers,
            dropout=dropout,
        )

        self.combine = self._make_combination_layer(
            combination_method=combination_method
        )

        # project combination to lstm input dim
        self.linear_projection = self._make_projection_block(
            combination_method=combination_method, act=linear_projection_activation_fn
        )

        self.output_head = nn.Linear(self.lstm.hidden_size, self.output_vocab_size)

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

    def _repeat_prior_samples_if_needed(
        self, prior_samples: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        if self._prior_sample_format == PriorSampleFormat.VEC:
            return prior_samples.unsqueeze(1).repeat(1, seq_len, 1)

        return prior_samples

    def forward(
        self,
        inputs: torch.Tensor,
        prior_samples: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model.

        Args:
            inputs (torch.Tensor): input sequence of integers, where each integer corresponds to a token in a corpus. Shape: (b, seq_len).
            prior_samples (torch.Tensor): samples from a prior distribution. Shape: (b, ?, ?), where <b> is the batch size. Prior samples may either be,
                a 3D tensor or a 2D tensor. The final dimension may or may not be known ahead of time.
            initial_state (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): tuple of tensors giving the last hidden state and cell state of the LSTM.
                Each of these tensors have shape (batch_size, n_lstm_hidden_layers, lstm_hidden_size). Defaults to None.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: tuple of tensors,
                the first being the output of the LSTM of shape (batch_size, seq_length, vocab_size) and the second being
                the last hidden state and cell state of the LSTM. Each of these tensors have shape (batch_size, n_lstm_layers, lstm_hidden_size).
        """

        # can only concat similar tensors so we first expand to 3D, then repeat to match shape of input
        # prior_samples shape: (b, seq_len, [?]) -> (b, seq_len, ?)
        prior_samples = self._repeat_prior_samples_if_needed(
            prior_samples, seq_len=inputs.shape[1]
        )

        # (b, seq_len) -> (b, seq_len, d0)
        outputs = self.embedding(inputs)

        # [(b, seq_len, ?), (b, seq_len, d0)] -> (b, seq_len, ??)
        combined = self.combine(outputs, prior_samples)

        # (b, seq_len, ??) -> (b, seq_len, lstm_input_dim)
        outputs = self.linear_projection(combined)

        if initial_state is not None:
            # to make outputs of torch.nn.DataParallel compatible with torch.nn.LSTM
            # (batch_size, n_layers, hidden_size), (n_layers, batch_size, hidden_size)
            initial_state = self._transpose_state(initial_state)

        # outputs: (b, seq_len, n_directions * hidden_dim)
        # hidden_state: (h, c) where h, c have shape (b, n_directions * n_layers, hidden_dim)
        outputs, state = self.lstm(outputs, initial_state)

        # to make outputs of torch.nn.LSTM compatible with torch.nn.DataParallel
        # (n_layers, batch_size, hidden_size), (batch_size, n_layers, hidden_size)
        state = self._transpose_state(state)

        outputs = self.output_head(outputs)

        return outputs, state

    def hidden_state_shape(self, batch_size: int) -> Tuple[int, int, int]:
        return (self.n_layers, batch_size, self.lstm.hidden_size)

    @property
    def hidden_size(self) -> int:
        return self.lstm.hidden_size

    @property
    def sampling_temperature(self) -> float:
        return self._sampling_temperature

    def _make_xo(self, n_samples: int, device: torch.device) -> torch.Tensor:
        return torch.full((n_samples, 1), self.start_token_index, device=device)

    def generate(
        self,
        prior_samples: torch.Tensor,
        seq_len: int,
        random_seed: Optional[int] = None,
        return_probs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from the underlying model and return the raw form along with the
        conditional probabilities of each of the sequences.

        Args:
            n_samples (int): then number of samples to generate.
            seq_len (int): the desired maximum length of the generated sequences.
            random_seed (Optional[int], optional): an optional random seed for reproducibility. Defaults to None.
            return_probs (bool, optional): whether or not to return the conditional probabilities of each of the sequences. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the raw generated sequences and the associated probabilities.
        """

        if self._prior_sample_format == PriorSampleFormat.VEC:
            assert (
                prior_samples.ndim == 2
            ), f"Expecting samples to be 2D tensors, got {prior_samples.ndim}D."
        elif self._prior_sample_format == PriorSampleFormat.VEC_SEQ_2D:
            assert (
                prior_samples.ndim == 3
            ), f"Expecting samples to be 3D tensors, got {prior_samples.ndim}D."
        else:
            raise ValueError(
                f"Invalid prior sample format '{self._prior_sample_format}'"
            )

        random_generator = None
        n_samples = prior_samples.shape[0]

        device = prior_samples.device

        if random_seed:
            random_generator = torch.Generator()
            random_generator.manual_seed(random_seed)

        inputs = self._make_xo(n_samples, device=device)  # (batch_size, 1)
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        outputs = torch.zeros((n_samples, seq_len), device=device)
        seq_probabilities = torch.ones(
            (n_samples, seq_len, self.output_vocab_size), device=device
        )

        with torch.no_grad():
            for index in range(seq_len):
                # logits -> (batch_size, 1, vocab_size)
                # hidden_state -> (batch_size, D*n_layers, hidden_size), (batch_size, D*n_layers, hidden_size)

                sample = (
                    prior_samples[:, index, :]
                    if prior_samples.ndim == 3
                    else prior_samples
                )
                logits, hidden_state = self(inputs, sample, hidden_state)

                # (batch_size, 1, vocab_size) -> (batch_size, vocab_size)
                logits = logits.squeeze(1)

                cat_distribution = Categorical(
                    logits=logits / self.sampling_temperature
                )

                # sampled_token_indices -> (batch_size, )
                sampled_token_indices = cat_distribution.sample()

                outputs[:, index] = sampled_token_indices
                seq_probabilities[:, index, :] = cat_distribution.probs

                # inputs -> (batch_size, 1)
                inputs = sampled_token_indices.unsqueeze(1)

        if return_probs:
            return outputs, seq_probabilities

        return outputs
