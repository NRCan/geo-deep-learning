import torch
import torch.nn as nn
from torch.nn import functional as F

class ScriptModel(nn.Module):
    def __init__(self, model, data_module, num_classes, from_logits = True, aux_head = False):
        super().__init__()
        self.model = model
        self.data_module = data_module
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.aux_head = aux_head
    def forward(self, x):
        sample = {"image": x}
        preprocessed = self.data_module._model_script_preprocess(sample)
        output = self.model(preprocessed["image"])
        if isinstance(output, dict):
            output = output['out']
        if self.from_logits:
            if self.num_classes > 1:
                output = F.softmax(output, dim=1)
            else:
                output = F.sigmoid(output)
        return output

def script_model(model, data_module, num_classes, from_logits = True):
    return ScriptModel(model, data_module, num_classes, from_logits)
        