import torch
import torch.nn as nn
import torchvision.models as models

class DilatedVGG16Backend(nn.Module):
    """
    VGG16 with dilated convolutions for 8× downsampling
    
    Architecture:
    - conv1_1, conv1_2, pool1 → /2
    - conv2_1, conv2_2, pool2 → /4  
    - conv3_1, conv3_2, conv3_3, pool3 → /8
    - conv4_1, conv4_2, conv4_3 (dilated, no pool) → /8
    - conv5_1, conv5_2, conv5_3 (dilated, no pool) → /8
    
    Output: H/8 × W/8
    """
    
    def __init__(self, pretrained=True):
        super(DilatedVGG16Backend, self).__init__()
        
        vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())
        
        # VGG16_BN structure (with indices):
        # 0: Conv2d(3, 64)      - conv1_1
        # 1: BatchNorm2d(64)
        # 2: ReLU
        # 3: Conv2d(64, 64)     - conv1_2
        # 4: BatchNorm2d(64)
        # 5: ReLU
        # 6: MaxPool2d          → /2
        # 7-13: conv2 block     → /4
        # 14-23: conv3 block    → /8
        # 24-33: conv4 block    (we'll modify this)
        # 34-43: conv5 block    (we'll modify this)
        
        # Layers up to pool3 (output: /8)
        self.features_to_pool3 = nn.Sequential(*features[:24])  # Up to just before conv4
        
        # Conv4 block WITHOUT pool4 (keep /8)
        # Original conv4: features[24:34]
        # We keep the convs but remove the final maxpool
        self.conv4 = nn.Sequential(*features[24:33])  # Conv4 layers, no pool
        
        # Conv5 block with dilation (keep /8)
        # Original pool4 would be at features[33], we skip it
        # Original conv5 block at features[34:43], we modify it
        
        # Build dilated conv5 from scratch
        self.conv5_dilated = nn.Sequential(
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
        
        # Initialize dilated conv5 with pretrained weights if available
        if pretrained:
            self._init_dilated_from_pretrained(vgg)
    
    def _init_dilated_from_pretrained(self, vgg):
        """Initialize dilated conv5 with pretrained conv5 weights"""
        pretrained_conv5 = list(vgg.features.children())[34:43]
        
        # Extract conv layers (skip BN and ReLU)
        pretrained_convs = [pretrained_conv5[i] for i in [0, 3, 6]]
        dilated_convs = [self.conv5_dilated[i] for i in [0, 3, 6]]
        
        for pretrained_conv, dilated_conv in zip(pretrained_convs, dilated_convs):
            if isinstance(pretrained_conv, nn.Conv2d) and isinstance(dilated_conv, nn.Conv2d):
                # Copy weights (adjust for dilation if needed)
                dilated_conv.weight.data.copy_(pretrained_conv.weight.data)
                if dilated_conv.bias is not None and pretrained_conv.bias is not None:
                    dilated_conv.bias.data.copy_(pretrained_conv.bias.data)
        
    def forward(self, x):
        """
        Args:
            x: [B, 64, H, W] (after first stem conv)
        Returns:
            [B, 512, H/8, W/8]
        """
        x = self.features_to_pool3(x)  # → [B, 256, H/8, W/8]
        x = self.conv4(x)              # → [B, 512, H/8, W/8] (no pool)
        x = self.conv5_dilated(x)      # → [B, 512, H/8, W/8] (dilated)
        return x


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


def make_shared_dilated_backbone(pretrained=True):
    """
    Creates shared dilated VGG backbone (skips first conv layer)
    Returns tuple: (shared_backbone, dilated_conv)
    """
    vgg = models.vgg16_bn(pretrained=pretrained)
    features = list(vgg.features.children())
    
    # Skip first conv (index 0), start from BN layer (index 1)
    # Up to just before conv4
    shared_backbone = nn.Sequential(*features[1:24])
    
    # Conv4 without pooling
    conv4 = nn.Sequential(*features[24:33])
    
    # Dilated conv5
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
    
    # Initialize from pretrained
    if pretrained:
        pretrained_conv5 = list(vgg.features.children())[34:43]
        pretrained_convs = [pretrained_conv5[i] for i in [0, 3, 6]]
        dilated_convs = [conv5_dilated[i] for i in [0, 3, 6]]
        
        for pc, dc in zip(pretrained_convs, dilated_convs):
            if isinstance(pc, nn.Conv2d) and isinstance(dc, nn.Conv2d):
                dc.weight.data.copy_(pc.weight.data)
                if dc.bias is not None and pc.bias is not None:
                    dc.bias.data.copy_(pc.bias.data)
    
    # Combine shared backbone, conv4, and dilated conv5
    full_backbone = nn.Sequential(shared_backbone, conv4, conv5_dilated)
    
    return full_backbone, None  # Return None for dilated_conv since it's integrated