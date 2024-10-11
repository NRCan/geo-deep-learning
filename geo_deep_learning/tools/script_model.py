import torch
import torch.nn as nn

class ScriptModel(nn.Module):
    def __init__(self, model, data_module):
        super().__init__()
        self.model = model
        self.data_module = data_module
    
    def forward(self, x):
        sample = {"image": x}
        preprocessed = self.data_module.model_script_preprocess(sample)
        return self.model(preprocessed["image"])

def script_model(model, data_module):
    print(f"data_module: {data_module}")
    return ScriptModel(model, data_module)
        