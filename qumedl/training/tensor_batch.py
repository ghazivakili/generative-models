import torch


class TensorBatch:
    def __init__(
        self,
        inputs: torch.Tensor,
        targets_or_labels: torch.Tensor,
        **input_tensors: torch.Tensor,
    ):
        self.seq_len = inputs[-2] if inputs.ndim == 3 else inputs[-1]
        self.batch_size = inputs.shape[0]

        self._tensors = {
            "inputs": inputs,
            "targets_or_labels": targets_or_labels,
        }

        for input_tensor_name, input_tensor in input_tensors.items():
            self._tensors[input_tensor_name] = input_tensor

    @property
    def inputs(self) -> torch.Tensor:
        return self._tensors["inputs"]

    @inputs.setter
    def inputs(self, new_inputs: torch.Tensor) -> None:
        self._tensors["inputs"] = new_inputs

    @property
    def targets_or_labels(self) -> torch.Tensor:
        return self._tensors["targets_or_labels"]

    @targets_or_labels.setter
    def targets_or_labels(self, new_targets_or_labels: torch.Tensor) -> torch.Tensor:
        self._tensors["targets_or_labels"] = new_targets_or_labels

    def to(self, target) -> None:
        for tensor_name in self._tensors:
            self._tensors[tensor_name] = self._tensors[tensor_name].to(target)

    def get(self, tensor_name: str) -> torch.Tensor:
        return self._tensors[tensor_name]
