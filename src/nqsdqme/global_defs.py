import numpy as np
import torch



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_device(device=None):
    """Update global configuration"""
    global DEVICE
    if device:
        DEVICE = device


def get_device():
    """Get current configuration"""
    return DEVICE