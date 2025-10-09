"""Script model for inference."""

import torch
from torch import nn
from torch.nn import functional as f

from geo_deep_learning.utils.tensors import normalization, standardization


class ScriptModel(nn.Module):
    """Script model."""

    def __init__(  # noqa: PLR0913
        self,
        model: torch.nn.Module,
        device: torch.device | None = None,
        num_classes: int = 1,
        input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
        mean: list[float] | None = None,
        std: list[float] | None = None,
        image_min: int = 0,
        image_max: int = 255,
        norm_min: float = 0.0,
        norm_max: float = 1.0,
        *,
        from_logits: bool = True,
    ) -> None:
        """Initialize the script model."""
        super().__init__()
        self.device = device or torch.device("cpu")
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.mean = torch.tensor(mean or [0.0, 0.0, 0.0]).reshape(len(mean), 1)
        self.std = torch.tensor(std or [1.0, 1.0, 1.0]).reshape(len(std), 1)
        self.image_min = int(image_min)
        self.image_max = int(image_max)
        self.norm_min = float(norm_min)
        self.norm_max = float(norm_max)

        dummy_input = torch.rand(input_shape).to(self.device)
        self.traced_model = torch.jit.trace(model.eval(), dummy_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = normalization(
            x,
            self.image_min,
            self.image_max,
            self.norm_min,
            self.norm_max,
        )
        x = standardization(x, self.mean, self.std)
        output = self.traced_model(x)
        if self.from_logits:
            if self.num_classes > 1:
                output = f.softmax(output, dim=1)
            else:
                output = f.sigmoid(output)
        return output


class SegmentationScriptModel(ScriptModel):
    """Script model for segmentation."""

    def __init__(self, model: torch.nn.Module, **kwargs: object) -> None:
        """Initialize the segmentation script model."""
        super().__init__(model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = normalization(
            x,
            self.image_min,
            self.image_max,
            self.norm_min,
            self.norm_max,
        )
        x = standardization(x, self.mean, self.std)
        output = self.traced_model(x)
        output = output[0]  # Extract main output from NamedTuple
        if self.from_logits:
            if self.num_classes > 1:
                output = f.softmax(output, dim=1)
            else:
                output = f.sigmoid(output)
        return output
