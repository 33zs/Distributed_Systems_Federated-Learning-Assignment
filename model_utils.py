"""
Model utilities for federated learning.
Defines the neural network architecture using PyTorch.
"""

import torch
import torch.nn as nn
from config import NUM_CLASSES, INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2


class FashionMNISTNet(nn.Module):
    """
    Neural network for Fashion-MNIST classification.

    Architecture:
        - Input: 784 neurons (28×28 flattened)
        - Hidden Layer 1: 128 neurons with ReLU
        - Hidden Layer 2: 64 neurons with ReLU
        - Output: 10 neurons with Softmax
    """

    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax here - will use CrossEntropyLoss
        return x


def create_model():
    """
    Create and return a new instance of the FashionMNIST model.

    Returns:
        FashionMNISTNet model
    """
    return FashionMNISTNet()


def get_model_weights(model):
    """
    Extract model weights as a list of tensors.

    Args:
        model: PyTorch model

    Returns:
        List of weight tensors
    """
    return [param.data.clone() for param in model.parameters()]


def set_model_weights(model, weights):
    """
    Set model weights from a list of tensors.

    Args:
        model: PyTorch model
        weights: List of weight tensors
    """
    for param, weight in zip(model.parameters(), weights):
        param.data = weight.clone()


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
