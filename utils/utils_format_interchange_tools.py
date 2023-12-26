import numpy as np
import torch

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy array and scale its values to 0-255."""
    array = tensor.numpy().squeeze()
    return (array * 255).astype(np.uint8)

def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a tensor and scale its values from 0-255 to 0-1."""
    array = array.astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]

