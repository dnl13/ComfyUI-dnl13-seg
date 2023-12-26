
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from PIL import Image
from typing import Tuple

import matplotlib.cm as cm


from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torchvision.transforms as T

from scipy.ndimage import gaussian_filter

import folder_paths
from ...utils.helper_img_utils  import dilate_mask, resize_image, apply_colormap, overlay_image
from ...utils.utils_format_interchange_tools import numpy_to_tensor
from ...utils.helper_cmd_and_path import print_labels
dnl13 = print_labels("dnl13")

clipseg_model_dir = os.path.join(folder_paths.models_dir, "clipseg")


class ClipSegmentationProcessor:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {"required":
                    {
                        "image": ("IMAGE",),
                        "text": ("STRING", {"multiline": False}),
                        
                     },
                "optional":
                    {
                        "blur": ("FLOAT", {"min": 0, "max": 15, "step": 0.1, "default": 7}),
                        "threshold": ("FLOAT", {"min": 0, "max": 1, "step": 0.05, "default": 0.4}),
                        "dilation_factor": ("INT", {"min": 0, "max": 10, "step": 1, "default": 4}),
                    }
                }

    CATEGORY = "dnl13/vision"
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("Mask","Heatmap Mask", "BW Mask")

    FUNCTION = "segment_image"
    def segment_image(self, image: torch.Tensor, text: str, blur: float, threshold: float, dilation_factor: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a segmentation mask from an image and a text prompt using CLIPSeg.

        Args:
            image (torch.Tensor): The image to segment.
            text (str): The text prompt to use for segmentation.
            blur (float): How much to blur the segmentation mask.
            threshold (float): The threshold to use for binarizing the segmentation mask.
            dilation_factor (int): How much to dilate the segmentation mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The segmentation mask, the heatmap mask, and the binarized mask.
        """

        tensor_bw_list, image_out_heatmap_list, image_out_binary_list = [], [], []
        for item in tqdm(image, total=len(image), desc=f"{dnl13} analyizing batch", unit="img", ncols=100, colour='green'):
            # Convert the Tensor to a PIL image
            image_np = item.numpy().squeeze()  # Remove the first dimension (batch size of 1)
            # Convert the numpy array back to the original range (0-255) and data type (uint8)
            image_np = (image_np * 255).astype(np.uint8)
            # Create a PIL image from the numpy array
            i = Image.fromarray(image_np, mode="RGB")

            processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=clipseg_model_dir)
            model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=clipseg_model_dir)
            
            prompt = text
            
            input_prc = processor(text=prompt, images=i, padding="max_length", return_tensors="pt")
            
            # Predict the segemntation mask
            with torch.no_grad():
                outputs = model(**input_prc)
            
            tensor = torch.sigmoid(outputs[0]) # get the mask
            
            # Apply a threshold to the original tensor to cut off low values
            thresh = threshold
            tensor_thresholded = torch.where(tensor > thresh, tensor, torch.tensor(0, dtype=torch.float))

            # Apply Gaussian blur to the thresholded tensor
            sigma = blur
            tensor_smoothed = gaussian_filter(tensor_thresholded.numpy(), sigma=sigma)
            tensor_smoothed = torch.from_numpy(tensor_smoothed)

            # Normalize the smoothed tensor to [0, 1]
            mask_normalized = (tensor_smoothed - tensor_smoothed.min()) / (tensor_smoothed.max() - tensor_smoothed.min())

            # Dilate the normalized mask
            mask_dilated = dilate_mask(mask_normalized, dilation_factor)

            # Convert the mask to a heatmap and a binary mask
            heatmap = apply_colormap(mask_dilated, cm.viridis)
            binary_mask = apply_colormap(mask_dilated, cm.Greys_r)

            # Overlay the heatmap and binary mask on the original image
            dimensions = (image_np.shape[1], image_np.shape[0])
            heatmap_resized = resize_image(heatmap, dimensions)
            binary_mask_resized = resize_image(binary_mask, dimensions)

            alpha_heatmap, alpha_binary = 0.5, 1
            overlay_heatmap = overlay_image(image_np, heatmap_resized, alpha_heatmap)
            overlay_binary = overlay_image(image_np, binary_mask_resized, alpha_binary)

            # Convert the numpy arrays to tensors
            image_out_heatmap = numpy_to_tensor(overlay_heatmap)
            image_out_binary = numpy_to_tensor(overlay_binary)

            # Save or display the resulting binary mask
            binary_mask_image = Image.fromarray(binary_mask_resized[..., 0])

            # convert PIL image to numpy array
            tensor_bw = binary_mask_image.convert("RGB")
            tensor_bw = np.array(tensor_bw).astype(np.float32) / 255.0
            tensor_bw = torch.from_numpy(tensor_bw)[None,]
            tensor_bw = tensor_bw.squeeze(0)[..., 0]

            tensor_bw_list.append(tensor_bw)
            image_out_heatmap_list.append(image_out_heatmap)
            image_out_binary_list.append(image_out_binary)


        tensor_bw_out = torch.cat(tensor_bw_list, dim=0).to("cpu")
        image_out_heatmap_out = torch.cat(image_out_heatmap_list, dim=0).to("cpu")
        image_out_binary_out = torch.cat(image_out_binary_list, dim=0).to("cpu")
        return tensor_bw_out, image_out_heatmap_out, image_out_binary_out
