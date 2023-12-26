import torch
from typing import Optional, Tuple
import matplotlib.cm as cm

from ...utils.helper_img_utils  import resize_image, apply_colormap, overlay_image
from ...utils.utils_format_interchange_tools import numpy_to_tensor, tensor_to_numpy

class CombineMasks:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "input_image": ("IMAGE", ),
                        "mask_1": ("MASK", ), 
                        "mask_2": ("MASK", ),
                    },
                "optional": 
                    {
                        "mask_3": ("MASK",), 
                    },
                }
        
    CATEGORY = "dnl/utilitys"
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("Combined Mask","Heatmap Mask", "BW Mask")

    FUNCTION = "combine_masks"
            
    def combine_masks(self, input_image: torch.Tensor, mask_1: torch.Tensor, mask_2: torch.Tensor, mask_3: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A method that combines two or three masks into one mask. Takes in tensors and returns the mask as a tensor, as well as the heatmap and binary mask as tensors."""

        # Combine masks
        combined_mask = mask_1 + mask_2 + mask_3 if mask_3 is not None else mask_1 + mask_2


        # Convert image and masks to numpy arrays
        image_np = tensor_to_numpy(input_image)
        heatmap = apply_colormap(combined_mask, cm.viridis)
        binary_mask = apply_colormap(combined_mask, cm.Greys_r)

        # Resize heatmap and binary mask to match the original image dimensions
        dimensions = (image_np.shape[1], image_np.shape[0])
        heatmap_resized = resize_image(heatmap, dimensions)
        binary_mask_resized = resize_image(binary_mask, dimensions)

        # Overlay the heatmap and binary mask onto the original image
        alpha_heatmap, alpha_binary = 0.5, 1
        overlay_heatmap = overlay_image(image_np, heatmap_resized, alpha_heatmap)
        overlay_binary = overlay_image(image_np, binary_mask_resized, alpha_binary)

        # Convert overlays to tensors
        image_out_heatmap = numpy_to_tensor(overlay_heatmap)
        image_out_binary = numpy_to_tensor(overlay_binary)

        return combined_mask, image_out_heatmap, image_out_binary
