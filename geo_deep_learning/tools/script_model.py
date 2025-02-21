import torch
import torch.nn as nn
from torch.nn import functional as F
from tools.utils import normalization, standardization

class ScriptModel(nn.Module):
    def __init__(self, 
                 model,
                 device = torch.device("cpu"),
                 num_classes = 1,
                 input_shape = (1, 3, 512, 512),
                 mean = [0.0, 0.0, 0.0],
                 std = [1.0, 1.0, 1.0],
                 image_min = 0,
                 image_max = 255,
                 norm_min = 0.0,
                 norm_max = 1.0,
                 from_logits = True):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.mean = torch.tensor(mean).reshape(len(mean), 1)
        self.std = torch.tensor(std).reshape(len(std), 1)
        self.image_min = int(image_min)
        self.image_max = int(image_max)
        self.norm_min = float(norm_min)
        self.norm_max = float(norm_max)
        
        dummy_input = torch.rand(input_shape).to(self.device)
        self.traced_model = torch.jit.trace(model.eval(), dummy_input)
    
    def forward(self, x):
        x = normalization(x, self.image_min, self.image_max, self.norm_min, self.norm_max)
        x = standardization(x, self.mean, self.std)
        output = self.traced_model(x)
        if self.from_logits:
            if self.num_classes > 1:
                output = F.softmax(output, dim=1)
            else:
                output = F.sigmoid(output)
        return output

class SegmentationScriptModel(ScriptModel):
    """
    This class is used to script a model that returns a NamedTuple.
    
    for example:
    class SegmentationOutput(NamedTuple):
        out: torch.Tensor
        aux: Optional[torch.Tensor]
    """
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
    
    def forward(self, x):
        x = normalization(x, self.image_min, self.image_max, self.norm_min, self.norm_max)
        x = standardization(x, self.mean, self.std)
        output = self.traced_model(x)
        output = output[0] # Extract main output from NamedTuple
        if self.from_logits:
            if self.num_classes > 1:
                output = F.softmax(output, dim=1)
            else:
                output = F.sigmoid(output)
        return output
        