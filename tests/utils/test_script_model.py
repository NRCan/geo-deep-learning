import torch
import pytest
from utils.script_model import ScriptModel

def test_script_model():
    model = torch.nn.Linear(3, 1)
    script_model = ScriptModel(model,
                               input_shape=(1, 3),)

    input_tensor = torch.rand((1, 3))
    output = script_model.forward(input_tensor)
    
    assert output.shape == (1, 1)
    assert isinstance(output, torch.Tensor)
