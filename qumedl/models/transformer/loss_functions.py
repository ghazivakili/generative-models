from typing import Callable

import torch
from torch import nn

from qumedl.training.tensor_batch import TensorBatch


def compute_transformer_loss(
    model: nn.Module,
    tensor_batch: TensorBatch,
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
    vocab_size = class_logits.shape[-1]

    shift_logits = class_logits[:, :-1]
    shift_targets = targets[:, 1:]

    # flatten batches to (bs * seq_len, vocab_size) and (bs * seq_len, )
    loss: torch.Tensor = loss_function(
        shift_logits.reshape((-1, vocab_size)).contiguous(),
        shift_targets.reshape((-1,)).contiguous(),
    )

    return loss
