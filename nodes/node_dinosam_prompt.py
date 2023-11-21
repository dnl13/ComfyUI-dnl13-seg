import torch
import numpy as np
from PIL import Image

import comfy.model_management
from ..node_fnc.node_dinosam_prompt import sam_segment, groundingdino_predict
from ..utils.collection import get_local_filepath
import os
import folder_paths

from groundingdino.util.slconfig import SLConfig
#from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict

from ..utils.grounding_dino_utils import build_model, build_groundingdino

from .node_modelloader_dinov1 import load_groundingdino_model, list_groundingdino_model


class GroundingDinoSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {"default": "arms, legs, eyes, hair, head","multiline": True}),
                "box_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
                #"text_threshold": ("FLOAT", {
                #    "default": 0.3,
                #    "step": 0.01
                #}),
                "multimask": ('BOOLEAN', {"default":False}),
                "dedicated_device": (["Auto", "CPU", "GPU"], ),
                "optimize_prompt_for_dino":('BOOLEAN', {"default":True}),
                "clean_mask_holes": ("INT", {
                    "default": 64,
                    "min": 0,
                    "step": 1
                }),
                "clean_mask_islands": ("INT", {
                    "default": 64,
                    "min": 0,
                    "step": 1
                }),
                "mask_blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1
                }),
                "mask_grow_shrink_factor": ("INT", {
                    "default": 0,
                    "min": -1024,
                    "max":1024,
                    "step": 1
                }), 
            },
        }
    CATEGORY = "dnl13"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(
            self, 
            grounding_dino_model,
            clean_mask_holes,
            clean_mask_islands, 
            mask_blur, 
            mask_grow_shrink_factor,
            sam_model, 
            image, 
            prompt, 
            box_threshold, 
            # text_threshold,
            optimize_prompt_for_dino=False,
            multimask=False, 
            dedicated_device="Auto"
            ):
        #
        #load Grounding Dino Model
        #grounding_dino_model = load_groundingdino_model(grounding_dino_model)
        #
        device_mapping = {
            "Auto": comfy.model_management.get_torch_device(),
            "CPU": torch.device("cpu"),
            "GPU": torch.device("cuda")
        }
        device = device_mapping.get(dedicated_device)
        #
        # send model to selected device 
        grounding_dino_model.to(device)
        grounding_dino_model.eval()
        sam_model.to(device)
        sam_model.eval()
        #
        # in case sam or dino dont find anything, return blank mask and original image
        img_batch, img_height, img_width, img_channel = image.shape        # get original image dimensions 
        empty_mask = torch.zeros((1, 1, img_height, img_width), dtype=torch.float32) # [B,C,H,W]
        empty_mask = empty_mask / 255.0
        #
        # empty output
        res_images = []
        res_masks = []
        #
        detection_errors = False
        #
        for item in image:
            item = Image.fromarray(np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            # run dino for prompt guides segmentation

            boxes, phrases = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                box_threshold,
                #text_threshold,
                optimize_prompt_for_dino,
                device
            )
            
            print("\033[1;32m(dnl13-seg)\033[0m > phrase(confidence):", phrases)
            
            # if nothing is detected set detection_errors
            if boxes.numel() == 0:
                detection_errors = True
                break

            # create detailed masks with SAM depending on boxes found by dino
            (images, masks) = sam_segment(
                sam_model,
                item,
                boxes,
                clean_mask_holes,
                clean_mask_islands,
                mask_blur,
                mask_grow_shrink_factor,
                multimask,
                device
            )
            # add results to output

            

            res_images.extend(images)
            res_masks.extend(masks)
        
        # if nothing was detected just send simple input image and empty mask
        if detection_errors is not False:
            print("\033[1;32m(dnl13-seg)\033[0m The tensor 'boxes' is empty. No elements were found in the image search.")
            res_images.append(image)
            res_masks.append(empty_mask)

        # generate output
        res_images = torch.cat(res_images, dim=0)
        res_masks = torch.cat(res_masks, dim=0)

        # fix for ComfyUI Mask format :MASK torch.Tensor with shape [H,W] or [B,C,H,W]
        if multimask:
            res_masks = res_masks.unsqueeze_(dim=1) 
      
        return (res_images, res_masks, )

