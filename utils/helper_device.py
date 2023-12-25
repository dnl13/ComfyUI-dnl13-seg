import torch
import comfy.model_management
"""
def device_mapping(dedicated_device):
    device_mapping = {  
        "Auto": comfy.model_management.get_torch_device(),
        "CPU": torch.device("cpu"),
        "GPU": torch.device("cuda")
    }
    device = device_mapping.get(dedicated_device)
    return device
"""
def list_available_devices():
    """Listet alle verfügbaren Geräte auf."""
    devices = ["Auto", "CPU"]
    if torch.cuda.is_available():
        devices += [f"GPU {i}" for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        devices += ["MPS"]
    return devices

def get_device(dedicated_device):
    """Gibt das entsprechende PyTorch-Gerät basierend auf der Benutzerauswahl zurück."""
    if dedicated_device == "Auto":
        return comfy.model_management.get_torch_device()
    elif dedicated_device.startswith("GPU"):
        gpu_index = int(dedicated_device.split()[-1])
        return torch.device(f"cuda:{gpu_index}")
    elif dedicated_device == "MPS":
        return torch.device("mps")
    else:
        return torch.device("cpu")