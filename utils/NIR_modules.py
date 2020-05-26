import torch
import numpy as np
import copy
import torchvision
from torch import nn
import torchvision.models as models
from models import common


class NIRExtractor(nn.Module):
    def __init__(self, submodule, extracted_layer):
        super(NIRExtractor, self).__init__()
        # TODO: documentation
        self.submodule = submodule
        self.extracted_layer = extracted_layer

    def forward(self, x):
        if self.extracted_layer == 'conv1': 
            #print(list(self.submodule.children())[:4])
            modules = list(self.submodule.children())[:4]
        elif self.extracted_layer == 'inner-layer-3':                     
            modules = list(self.submodule.children())[:6]
            third_module = list(self.submodule.children())[6]
            third_module_modules = list(third_module.children())[:3]    # take the first three inner modules
            third_module = nn.Sequential(*third_module_modules)
            modules.append(third_module)
        elif self.extracted_layer == 'layer-3':                     
            modules = list(self.submodule.children())[:7]
        else:                                               # after avg-pool
            modules = list(self.submodule.children())[:9]

        self.submodule = nn.Sequential(*modules)
        x = self.submodule(x)
        return x


# model_ft = models.resnet50(pretrained=True)
# extractor = Resnet50Extractor(model_ft, extracted_layer)
# features = extractor(input_tensor)


class MyEnsemble(nn.Module):
    def __init__(self, num_channels):
        super(MyEnsemble, self).__init__()
        # TODO: documentation
        model_rgb = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, aux_loss=None)
        model_rgb.classifier = common.DeepLabHead(2048, num_channels)

        model_nir = copy.deepcopy(model_rgb)
        model_nir.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.modelB = NIRExtractor(model_nir, 'conv1')


        self.modelA = model_rgb

        #self.modelA = modelA
        #self.modelB = modelB
        #self.classifier = nn.Linear(4, 2)
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
    
        # Collect the weight for each model
        #depth = x1.backbone._modules['conv1'].weight.detach().numpy()
        #depth_NIR = x2.backbone._modules['conv1'].weight.detach().numpy()

        #x = torch.cat((x1, x2), dim=1)
        #x = self.classifier(F.relu(x))

        #new_weight = torch.cat((depth, depth_NIR), dim=1)
        #x1.backbone._modules['conv1'].weight = nn.Parameter(new_weight, requires_grad=True)

        #new_weight = torch.cat((depth, depth_NIR), dim=1)

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
