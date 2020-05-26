import torch
import numpy as np
import torchvision
from torch import nn


class MyModelA(nn.Module):
    def __init__(self):
        super(MyModelA, self).__init__()
        self.fc1 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        return x
    

class MyModelB(nn.Module):
    def __init__(self):
        super(MyModelB, self).__init__()
        self.fc1 = nn.Linear(20, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        return x


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(4, 2)
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        
        # Collect the weight for each model
        depth = x1.backbone._modules['conv1'].weight.detach().numpy()
        depth_NIR = x2.backbone._modules['conv1'].weight.detach().numpy()

        #x = torch.cat((x1, x2), dim=1)
        #x = self.classifier(F.relu(x))

        new_weight = torch.cat((depth, depth_NIR), dim=1)
        x1.backbone._modules['conv1'].weight = nn.Parameter(new_weight, requires_grad=True)


        return x1

## Create models and load state_dicts    
#modelA = MyModelA()
#modelB = MyModelB()
## Load state dicts
#modelA.load_state_dict(torch.load(PATH))
#modelB.load_state_dict(torch.load(PATH))
#
#model = MyEnsemble(modelA, modelB)
#x1, x2 = torch.randn(1, 10), torch.randn(1, 20)
#output = model(x1, x2)
