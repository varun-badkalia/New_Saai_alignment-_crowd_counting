# models/backbones.py
"""
Dilated VGG16 backbones for 8× downsampling
Now supports SEPARATE backbones for RGB and Thermal
"""
import torch
import torch.nn as nn
import torchvision.models as models


class InputStemRGB(nn.Module):
    """First conv layer for RGB input"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight)
    
    def forward(self, x):
        return self.conv(x)


class InputStemThermal(nn.Module):
    """First conv layer for Thermal input"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight)
    
    def forward(self, x):
        return self.conv(x)


def make_dilated_vgg_backbone(pretrained=True):
    """
    Creates SEPARATE dilated VGG16 backbone with 8× downsampling
    
    Architecture:
    - conv1_1, conv1_2, pool1 → /2
    - conv2_1, conv2_2, pool2 → /4
    - conv3_1, conv3_2, conv3_3, pool3 → /8
    - conv4_1, conv4_2, conv4_3 (no pool) → /8
    - conv5_1, conv5_2, conv5_3 (dilated) → /8
    
    Returns:
        nn.Module: Separate backbone (output: 512 channels, /8 downsampling)
    """
    vgg = models.vgg16_bn(pretrained=pretrained)
    features = list(vgg.features.children())
    
    # Skip first conv (we have separate stems), start from BN
    # Up to just before conv4 (index 24)
    layers_to_pool3 = nn.Sequential(*features[1:24])
    
    # Conv4 without pooling (index 24-33, but skip pool at 33)
    conv4 = nn.Sequential(*features[24:33])
    
    # Dilated conv5 (replaces original conv5 + pool5)
    conv5_dilated = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
    )
    
    # Initialize dilated conv5 with pretrained weights
    if pretrained:
        pretrained_conv5 = list(vgg.features.children())[34:43]
        pretrained_convs = [pretrained_conv5[i] for i in [0, 3, 6]]
        dilated_convs = [conv5_dilated[i] for i in [0, 3, 6]]
        
        for pc, dc in zip(pretrained_convs, dilated_convs):
            if isinstance(pc, nn.Conv2d) and isinstance(dc, nn.Conv2d):
                dc.weight.data.copy_(pc.weight.data)
                if dc.bias is not None and pc.bias is not None:
                    dc.bias.data.copy_(pc.bias.data)
    
    # Combine all layers
    full_backbone = nn.Sequential(layers_to_pool3, conv4, conv5_dilated)
    
    return full_backbone


class FeatureAdapter(nn.Module):
    """Adapter to match feature dimensions"""
    def __init__(self, in_channels, out_channels=512):
        super(FeatureAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.adapter(x)