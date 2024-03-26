#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 2 2023

@author: Claudine Gravel-Miguel
@description: This script defines the structure of the different layers in the UNet model as well as imports its latest pretrained weights

"""

import torch
import torch.nn as nn

from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

def double_conv(in_channels, out_channels):
    """
    Defines the steps of the double convolutions layers

        Parameters:
            in_channels (int): The number of channels of the input
            out_channels (int): The number of channels produced by the convolution

        Returns:
            Sequential (Pytorch function): A function that runs the input through the different steps sequentially
    """
    
    return nn.Sequential(
        # First convolution, using 3x3 kernel with stride 1 and padding 1
        # This changes the number of channels of the input from in_ to out_
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        # Apply batch normalization of the output
        nn.BatchNorm2d(out_channels),
        # Apply the rectify linear unit activation function on the values so that values below 0 are set to 0.
        nn.ReLU(inplace=True),
        
        # Second convolution, which does not change the input size this time
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_samp(in_channels, out_channels):
    """
    Defines the step of the upsample (which is a convolutional transposition). This is used in the decoder to return the image to its original shape.

        Parameters:
            in_channels (int): The number of channels of the input
            out_channels (int): The number of channels produced by the convolution

        Returns:
            Sequential (Pytorch function): A function that runs the input through the different steps sequentially
    """
    
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        # UNet does not use ReLU here.
    )

class UNet(nn.Module):
    """
    A Pytorch module that defines the steps of the UNet structure (encoder and decoder)

        Parameters:
            in_channels (int): The number of channels of the input
            out_channels (int): The number of channels produced by the convolution

        Returns:
            x (Pytorch tensor): The tensor of predicted segmentations
    """
    
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()

        # Encoder
        # This part uses the PyTorch sequential blocks already defined by the resnet18 structure imported from torchvision
        self.encoder = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(*(list(self.encoder.children())[0:3]))
        self.down_maxpool = nn.Sequential(*(list(self.encoder.children())[3:4]))
        self.down_block1 = nn.Sequential(*(list(self.encoder.children())[4:5]))
        self.down_block2 = nn.Sequential(*(list(self.encoder.children())[5:6]))
        self.down_block3 = nn.Sequential(*(list(self.encoder.children())[6:7]))
        # this last one is the bottom block
        self.down_block4 = nn.Sequential(*(list(self.encoder.children())[7:8]))

        # Decoder     
        # This part uses the functions created above to represent the different steps of the decoder (compare the in_ and out_channels to the drawn structure to understand better)
        self.up_transpose7 = up_samp(512, 256)
        self.up_conv7 = double_conv(512, 256)
        self.dropout = nn.Dropout(0.3) # Apply dropout layer
        
        self.up_transpose8 = up_samp(256, 128)
        self.up_conv8 = double_conv(256, 128)
        
        self.up_transpose9 = up_samp(128, 64)
        self.up_conv9 = double_conv(128, 64)
        
        self.up_transpose10 = up_samp(64, 32)
        self.up_conv10 = double_conv(64 + 32, 32)

        # Final prediction
        self.out_transpose = up_samp(32, 32)
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    # The forward function is what actually calls the structure defined above
    def forward(self, x):
        # Encoder
        stem = self.stem(x)
        down_maxpool = self.down_maxpool(stem)
        down_block1 = self.down_block1(down_maxpool)
        down_block2 = self.down_block2(down_block1)
        down_block3 = self.down_block3(down_block2)
        down_block4 = self.down_block4(down_block3)
        
        # Decoder
        x = self.up_transpose7(down_block4)
        x = torch.cat([x, down_block3], dim=1)
        x = self.up_conv7(x)
        x = self.dropout(x)  # Apply dropout to the tensor

        x = self.up_transpose8(x)
        x = torch.cat([x, down_block2], dim=1)
        x = self.up_conv8(x)

        x = self.up_transpose9(x)
        x = torch.cat([x, down_block1], dim=1)
        x = self.up_conv9(x)

        x = self.up_transpose10(x)
        x = torch.cat([x, stem], dim=1)
        x = self.up_conv10(x)

        # Final prediction
        x = self.out_transpose(x)
        x = self.out(x)
        
        return x

# Can test using the following
# image = torch.rand((1, 3, 572, 572))
# model = UNet()
# print(model(image))