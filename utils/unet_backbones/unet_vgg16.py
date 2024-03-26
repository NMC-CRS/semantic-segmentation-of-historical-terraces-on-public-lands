#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 20 2023

@author: Claudine Gravel-Miguel
@description: This script defines the structure of the different layers in the UNet model as well as imports its latest pretrained weights

"""

import torch
import torch.nn as nn

from torchvision.models import vgg16_bn
from torchvision.models import VGG16_BN_Weights

# Helper functions to simplify the code in the UNet itself
def double_conv(in_channels, out_channels):
    """
    Defines the steps of the double convolutions layers (two convolutions, each followed by batch normalization and ReLU)

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


def triple_conv(in_channels, out_channels):
    """
    Defines the steps of the triple convolutions layers (three convolutions, each followed by batch normalization and ReLU)

        Parameters:
            in_channels (int): The number of channels of the input
            out_channels (int): The number of channels produced by the convolution

        Returns:
            Sequential (Pytorch function): A function that runs the input through the different steps sequentially
    """
    
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    """
    Defines the step of the upsample (which is a convolutional transposition). This is used in the decoder to return the image to its original shape.

        Parameters:
            in_channels (int): The number of channels of the input
            out_channels (int): The number of channels produced by the convolution

        Returns:
            Sequential (Pytorch function): A function that runs the input through the different steps sequentially
    """
    
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
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
    
    def __init__(self, pretrained=True, out_channels=2):
        super(UNet, self).__init__()

        # Encoder
        # This part uses the PyTorch sequential blocks already defined by the VGG16 structure imported from torchvision
        self.encoder = vgg16_bn(weights = VGG16_BN_Weights.DEFAULT).features
        self.down_block1 = nn.Sequential(*self.encoder[:6])
        self.down_block2 = nn.Sequential(*self.encoder[6:13])
        self.down_block3 = nn.Sequential(*self.encoder[13:23])
        self.down_block4 = nn.Sequential(*self.encoder[23:33])
        # this last one is the bottom block
        self.down_block5 = nn.Sequential(*self.encoder[33:43])

        # Decoder     
        # This part uses the functions created above to represent the different steps of the decoder (compare the in_ and out_channels to the drawn structure to understand better)
        # This reverses the steps of the encoder, which is not necessary, but was a choice. We could have also simplified it to use the same decoder as the original UNet (no triple convolutions)
        self.up_trans1 = up_conv(512, 256)
        self.up_conv1 = triple_conv(256 + 512, 256)
        self.dropout = nn.Dropout(0.3) # apply dropout layer
        
        self.up_trans2 = up_conv(256, 128)
        self.up_conv2 = triple_conv(128 + 256, 128)
        
        self.up_trans3 = up_conv(128, 64)
        self.up_conv3 = double_conv(64 + 128, 64)
        
        self.up_trans4 = up_conv(64, 32)
        self.up_conv4 = double_conv(32 + 64, 32)
        
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
        
    # The forward function is what actually calls the structure defined above
    def forward(self, x):
        # Encoder
        down_block1 = self.down_block1(x)
        down_block2 = self.down_block2(down_block1)
        down_block3 = self.down_block3(down_block2)
        down_block4 = self.down_block4(down_block3)
        down_block5 = self.down_block5(down_block4)

        # Decoder
        x = self.up_trans1(down_block5)
        x = torch.cat([x, down_block4], dim=1)
        x = self.up_conv1(x)
        x = self.dropout(x)  # Apply dropout to the tensor

        x = self.up_trans2(x)
        x = torch.cat([x, down_block3], dim=1)
        x = self.up_conv2(x)

        x = self.up_trans3(x)
        x = torch.cat([x, down_block2], dim=1)
        x = self.up_conv3(x)

        x = self.up_trans4(x)
        x = torch.cat([x, down_block1], dim=1)
        x = self.up_conv4(x)

        x = self.out(x)

        return x
    
# Can test using the following
# image = torch.rand((1, 3, 256, 256))
# model = UNet()
# print(model(image))
     