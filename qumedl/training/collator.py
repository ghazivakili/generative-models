import torch
from torch.utils.data import default_collate
from .tensor_batch import TensorBatch


class TensorBatchCollator:
    def __init__(self) -> None:
        pass

    def __call__(self, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        collated_inputs: torch.Tensor = default_collate(inputs)[0]
        collated_targerts = collated_inputs.clone()

        # sequence shifting for causal modeling should be done during training
        return TensorBatch(collated_inputs, collated_targerts)
