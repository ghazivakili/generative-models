from typing import Callable

import torch
from torch import nn

from qumedl.training.tensor_batch import TensorBatch


def compute_transformer_loss(
    model: nn.Module,
    tensor_batch: TensorBatch,
    prior_samples: torch.Tensor | None = None,
    pad_index=None,
) -> torch.Tensor:
    """Computes and returns the loss for a single batch of data for a
    Causal modeling Transformer.

    Args:
        model (nn.Module): transformer model to train.
        tensor_batch (TensorBatch): batch of tensor inputs and targets.
        prior_samples (torch.Tensor, optional): an optional batch of samples from a prior.
            If provided, assumes that ``model`` can accept "inputs" and ``prior_samples`` in its ``forward`` method.
    """
    loss_function = torch.nn.CrossEntropyLoss()
    inputs: torch.Tensor = tensor_batch.inputs
    targets: torch.Tensor = tensor_batch.targets_or_labels.long()

    class_logits: torch.Tensor

    if prior_samples is None:
        class_logits = model(inputs)
    else:
        class_logits = model(
            inputs, prior_samples=prior_samples, mask_attention=True, mask_pad=True
        )

    # need to shift by one for causal modeling
    vocab_size = class_logits.shape[-1]

    shift_logits = class_logits[:, :-1]
    shift_targets = targets[:, 1:]

    # flatten batches to (bs * seq_len, vocab_size) and (bs * seq_len, )
    loss: torch.Tensor = loss_function(
        shift_logits.reshape((-1, vocab_size)).contiguous(),
        shift_targets.reshape((-1,)).contiguous(),
    )

    return loss


def compute_transformer_loss_vlad(
    model: nn.Module,
    tensor_batch: TensorBatch,
    pad_index,
    prior_samples: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes and returns the loss for a single batch of data for a
    Causal modeling Transformer.

    Args:
        model (nn.Module): transformer model to train.
        tensor_batch (TensorBatch): batch of tensor inputs and targets.
        prior_samples (torch.Tensor, optional): an optional batch of samples from a prior.
            If provided, assumes that ``model`` can accept "inputs" and ``prior_samples`` in its ``forward`` method.
    """
    loss_function = torch.nn.CrossEntropyLoss()
    inputs: torch.Tensor = tensor_batch.inputs
    targets: torch.Tensor = tensor_batch.targets_or_labels.long()

    class_logits: torch.Tensor

    if prior_samples is None:
        class_logits = model(inputs, mask_attention=True, mask_pad=True)
    else:
        class_logits = model(
            inputs, prior_samples=prior_samples, mask_attention=True, mask_pad=True
        )

    # need to shift by one for causal modeling
    # vocab_size = class_logits.shape[-1]

    shift_logits = class_logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()

    # flatten batches to (bs * seq_len, vocab_size) and (bs * seq_len, )
    # loss: torch.Tensor = loss_function(
    #     shift_logits.reshape((-1, vocab_size)).contiguous(),
    #     shift_targets.reshape((-1,)).contiguous(),
    # )

    # logits = logits[:, :-1, :].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=pad_index,
    )

    return loss


# def loss_fn(logits, token_indices, pad_index, remove_sos_token: bool = True):
#     if remove_sos_token:
#         token_indices = token_indices[:, 1:].contiguous()
#     # Logits are always computed with SOS, regardless of whether the user has provided
#     # sequences with SOS token or not. If the user did not provide sequences with SOS
#     # then logits are computed by first adding SOS to every sequence. In this case, we
#     # get rid of the last token. If user did provide sequences with SOS, in the
#     # conditional above, we got rid of the SOS token, so we still get rid of the last
#     # logit:
#     logits = logits[:, :-1, :].contiguous()
#     loss = nn.functional.cross_entropy(
#         logits.view(-1, logits.size(-1)), token_indices.view(-1), ignore_index=pad_index
#     )
#     return loss
