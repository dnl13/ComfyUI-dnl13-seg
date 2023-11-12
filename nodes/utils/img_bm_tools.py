###
#
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
#
from PIL import Image
from .fallback_logging import logger


def images_greenscreen(images, masks, background_color=[0, 0, 0]):
    print("images_greenscreen")
    image = "IMAGE"
    return image


def crop_to_bbox(images, masks, bboxs):
    print("crop_to_bbox")
    cropped_images = "IMAGE"
    cropped_masks = "MASKS"
    return cropped_images, cropped_masks


def tensor_to_pil(tensor):
    rgb = tensor.squeeze(0) if tensor.dim() == 4 else tensor
    rgb = rgb.permute(2, 0, 1)
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(rgb)
    return image_pil


def squeeze_tensor_for_sam(tensor):
    tensor = tensor.squeeze(0) if tensor.dim() == 4 else tensor
    tensor = tensor.permute(2, 0, 1)
    return tensor


def parse_binary_masks(binary_masks):
    """
    binary_masks = list(dict(str, any)): A list over records for masks. Each record is  a dict containing the following keys:

    ""segmentation (dict(str, any) or np.ndarray): The mask. If output_mode='binary_mask', is an array of shape HW. Otherwise, is a dictionary containing the RLE.
    bbox (list(float)): The box around the mask, in XYWH format.
    area (int): The area in pixels of the mask.
    predicted_iou (float): The model's own prediction of the mask's quality. This is filtered by the pred_iou_thresh parameter.
    point_coords (list(list(float))): The point coordinates input to the model to generate this mask.
    stability_score (float): A measure of the mask's quality. This is filtered on using the stability_score_thresh parameter.
    crop_box (list(float)): The crop of the image used to generate the mask, given in XYWH format.
    """
    segmentation_list, bbox_list, crop_box_list = [], [], []

    if len(binary_masks) > 0:
        for maske in binary_masks:

            segmentation = maske["segmentation"]
            bbox = maske["bbox"]
            crop_box = maske["crop_box"]

            segmentation_image = (segmentation * 255).astype(np.uint8)
            pil_image = Image.fromarray(segmentation_image)
            pil_image.show()

            # Transfrom to Tensor
            segmentation_tensor = torch.tensor(segmentation, dtype=torch.bool)
            bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
            crop_box_tensor = torch.tensor(crop_box, dtype=torch.float32)
            
            # Append to list
            segmentation_list.append(segmentation_tensor.unsqueeze(0))
            bbox_list.append(bbox_tensor.unsqueeze(0))
            crop_box_list.append(crop_box_tensor.unsqueeze(0))
    else:
        logger.warning(
            "The list binary_masks is empty. No elements were detected.")
        
    # Stacking, to a Batch-Tensor
    segmentation_batch = torch.cat(segmentation_list, dim=0)
    bbox_batch = torch.cat(bbox_list, dim=0)
    crop_box_batch = torch.cat(crop_box_list, dim=0)

    return segmentation_batch, bbox_batch, crop_box_batch
