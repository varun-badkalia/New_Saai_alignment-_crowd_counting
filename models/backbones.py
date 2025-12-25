import torch
import torch.nn as nn
import torchvision.models as models

class InputStemRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight)
    def forward(self, x): return self.conv(x)

class InputStemThermal(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight)
    def forward(self, x): return self.conv(x)

def make_dilated_vgg_backbone(pretrained=True):
    """
    VGG16 with dilation in Stage 5 to maintain 1/8 spatial resolution.
    """
    vgg = models.vgg16_bn(pretrained=pretrained)
    features = list(vgg.features.children())
    
    # 1. Layers to Pool3 (downsample /2, /4, /8)
    # Skip first conv (handled by stem) -> Features[1:23] (up to pool3)
    layers_to_pool3 = nn.Sequential(*features[1:24]) 
    
    # 2. Conv4 (No pooling after this to keep /8)
    # Features[24:33] is Conv4_1 to Conv4_3 + ReLU + BN
    # We skip the MaxPool at index 33
    conv4 = nn.Sequential(*features[24:33])
    
    # 3. Dilated Conv5
    conv5_dilated = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
    )
    
    # Init dilated weights from pretrained if available
    if pretrained:
        pretrained_conv5 = list(vgg.features.children())[34:43]
        pretrained_convs = [pretrained_conv5[i] for i in [0, 3, 6]]
        dilated_convs = [conv5_dilated[i] for i in [0, 3, 6]]
        for pc, dc in zip(pretrained_convs, dilated_convs):
            dc.weight.data.copy_(pc.weight.data)
            if dc.bias is not None: dc.bias.data.copy_(pc.bias.data)
            
    return nn.Sequential(layers_to_pool3, conv4, conv5_dilated)

class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super(FeatureAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.adapter(x)