import torch

class ScriptModel(torch.nn.Module):
    def __init__(self, 
                 model,
                 device = torch.device("cpu"),
                 input_shape = (1, 3, 512, 512), 
                 mean = [0.405,0.432,0.397],
                 std = [0.164,0.173,0.153],
                 min = 0,
                 max = 255,
                 scaled_min = 0.0,
                 scaled_max = 1.0):
        super().__init__()
        self.device = device
        self.mean = torch.tensor(mean).resize_(len(mean), 1)
        self.std = torch.tensor(std).resize_(len(std), 1)
        self.min = min
        self.max = max
        self.min_val = scaled_min
        self.max_val = scaled_max
        
        input_tensor = torch.rand(input_shape).to(self.device)
        self.model_scripted = torch.jit.trace(model.eval(), input_tensor)
    
    def forward(self, input):
        shape = input.shape
        B, C = shape[0], shape[1]
        input = (self.max_val - self.min_val) * (input - self.min) / (self.max -self.min) + self.min_val
        input = (input.view(B, C, -1) - self.mean) / self.std
        input = input.view(shape)
        return self.model_scripted(input.to(self.device))