# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:28:51 2024

@author: Claudine Gravel-Miguel
@description: This holds the function that identifies the parameters that are frozen and the ones that will learn. 
"""

import torch.optim as optim

def setup_parameter_learning_rate(model_structure, model):
    """
    Set up the Adam optimizer to the parameters that will learn

        Parameters:
            model_structure (str): The name of the model structure (should be UNet)
            model (Pytorch model): The model itself

        Returns:
            pretrained_optimizer (Pytorch optimizer): Pytorch optim optimizer of the frozen parameters
            new_optimizer (Pytorch optimizer): Pytorch optim optimizer of the parameters that are learning
    """
    
    if model_structure == "UNet":
        # Separate the parameters of the pretrained layers and the newly added layers
        pretrained_params = []
        new_params = []
        for name, param in model.named_parameters():
            if 'encoder' in name:  # Assuming the pretrained layers are under 'encoder'
                pretrained_params.append(param)
            else:
                new_params.append(param)
    
    print(f"\n{len(new_params)} parameters will learn, whereas {len(pretrained_params)} remain frozen.\n")

    # Define the optimizer
    # Create separate optimizer instances for each set of parameters
    pretrained_optimizer = optim.Adam(pretrained_params, lr=0)  # Set learning rate to 0 to freeze
    new_optimizer = optim.Adam(new_params, lr=1e-3)  # Use a desired learning rate for the new layers
    
    return pretrained_optimizer, new_optimizer

# THE END