import torch.nn as nn
import torch
import torch
import torch.nn.functional as F

def categorical_crossentropy(output, target):
    """
    Categorical crossentropy loss function.
    
    Args:
    - output: The output from the model (logits, unnormalized).
    - target: The ground truth labels (one-hot encoded or class indices).
    
    Returns:
    - Loss value.
    """
    return F.cross_entropy(output, target)