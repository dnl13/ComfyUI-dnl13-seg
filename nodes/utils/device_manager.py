# some comfyui stuff
import comfy.model_management

####### all the device stuff
#device which is used per default by comfyui - will return "cpu" or "cuda:0" or even more "cuda:x"
device = comfy.model_management.get_torch_device()