import torch
import numpy as np
from PIL import Image

import comfy.model_management
from ..node_fnc.node_dinosam_prompt import sam_segment, groundingdino_predict
from ..utils.collection import get_local_filepath, check_mps_device
import os
import folder_paths

from groundingdino.util.slconfig import SLConfig
#from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict

from ..utils.grounding_dino_utils import build_model, build_groundingdino

from .node_modelloader_dinov1 import load_groundingdino_model, list_groundingdino_model

import re
def sort_result_to_prompt(phrases, masks, images, prompt):
    # Split prompt by ',' or '|'
    order_tokens = re.split(r',|\|', prompt)

    # Create a mapping from order tokens to their positions
    order_mapping = {token.strip(): i for i, token in enumerate(order_tokens)}

    # Define a custom sorting key function
    def custom_key(item):
        key_match = re.match(r'([^\d]+)\((\d+\.\d+)\)', item)
        if key_match:
            key, value = key_match.groups()
            return order_mapping.get(key.strip(), float('inf')), -float(value)
        else:
            # Handle the case where the format doesn't match
            return float('inf'), float(item)

    # Sort phrases, masks, images based on the custom key
    sorted_data = sorted(zip(phrases, masks, images), key=lambda x: custom_key(x[0]))

    # Unpack the sorted data into separate lists
    sorted_phrases, sorted_masks, sorted_images = zip(*sorted_data)

    return sorted_phrases, sorted_masks, sorted_images


def resize_boxes(boxes, original_size, scale_factor):
    """
    Vergrößert die Bounding-Boxen um einen bestimmten Prozentsatz der Originalbildgröße.

    Args:
    - boxes: Liste der Bounding-Boxen im Format xyxy [(x1, y1, x2, y2), ...]
    - original_size: Tupel (width, height) der Originalbildgröße
    - scale_factor: Der Prozentsatz, um den die Boxen vergrößert werden sollen

    Returns:
    - resized_boxes: Liste der vergrößerten Bounding-Boxen im Format xyxy
    """
    resized_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        width, height = original_size

        # Berechnen Sie die Breite und Höhe der Boxen
        box_width = x2 - x1
        box_height = y2 - y1

        # Berechnen Sie den Betrag der Vergrößerung basierend auf dem scale_factor
        scale_amount = scale_factor / 100.0

        # Vergrößern Sie die Boxen
        new_width = box_width + (width * scale_amount)
        new_height = box_height + (height * scale_amount)

        # Berechnen Sie die neuen Koordinaten der vergrößerten Boxen
        new_x1 = max(0, x1 - (new_width - box_width) / 2)
        new_y1 = max(0, y1 - (new_height - box_height) / 2)
        new_x2 = min(width, x2 + (new_width - box_width) / 2)
        new_y2 = min(height, y2 + (new_height - box_height) / 2)

        resized_boxes.append((new_x1, new_y1, new_x2, new_y2))

    return resized_boxes

def interpolate_mask_positions(boxes):
    """
    Interpoliert die Maskenpositionen relativ zur Durchschnittsposition der Bounding-Boxen.

    Args:
    - boxes: Tensor der Bounding-Boxen im Format xyxy (Shape: [batch_size, 4])
    - batch_size: Die Batch-Größe

    Returns:
    - interpolated_positions: Tensor der interpolierten Maskenpositionen (Shape: [batch_size, 4])
    """
    # Hier wird die Durchschnittsposition der Bounding-Boxen berechnet
    mean_boxes = torch.mean(boxes, dim=0)

    # Hier werden die Maskenpositionen relativ zur Durchschnittsposition interpoliert
    interpolated_positions = boxes - mean_boxes

    return mean_boxes + interpolated_positions

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
                    "default": 0.35,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
                "lower_confidence_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
                "upper_confidence_threshold": ("FLOAT", {
                    "default": 1,
                    "min": 0.02,
                    "max": 1.0,
                    "step": 0.01
                }),

                "multimask": ('BOOLEAN', {"default":False}),
                "two_pass": ('BOOLEAN', {"default":False}),
                "dedicated_device": (["Auto", "CPU", "GPU"], ),
                "optimize_prompt_for_dino":('BOOLEAN', {"default":True}),
                "clean_mask_holes": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max":100000,
                    "step": 1
                }),
                "clean_mask_islands": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max":100000,
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
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("RGBA_Images", "Masks", "DINO Image Detections Debug (rgb)")

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
            upper_confidence_threshold,
            lower_confidence_threshold,
            two_pass,
            optimize_prompt_for_dino=False,
            multimask=False, 
            dedicated_device="Auto"
            ):
        #
        #load Grounding Dino Model
        #grounding_dino_model = load_groundingdino_model(grounding_dino_model)
        #
        # :TODO - if comfy.model_management.get_torch_device(), returns mps (apple silicon) switch to CPU 
        device_mapping = {
            "Auto": check_mps_device(),
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
        res_debug_images = []
        #
        detection_errors = False
        #
        # Nehme die Höhe und Breite des aktuellen Bildes
        height, width = image.size(1), image.size(2)
    
        # Erstelle ein leeres transparentes Bild mit derselben Größe wie das aktuelle Bild
        transparent_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        transparent_image_np = np.array(transparent_image)
        transparent_image_tensor = torch.from_numpy(transparent_image_np).permute(2, 0, 1).unsqueeze(0)
        transparent_image_tensor = transparent_image_tensor.permute(0, 2, 3, 1)
        transparent_maske_tensor = torch.zeros(1, height, width)

        #
        for item in image:
            item = Image.fromarray(np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            # run dino for prompt guides segmentation

            boxes, phrases, logits, debug_image = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                box_threshold,
                upper_confidence_threshold,
                lower_confidence_threshold,
                optimize_prompt_for_dino,
                device
            )


            if boxes.size(0) == 0:
            # When Dino didnt found a thing replace with empty image and empty mask
                images = [transparent_image_tensor]
                masks = [transparent_maske_tensor]
            else:
            # When Dino found a thing do SAM segmentation
                (images, masks) = sam_segment( 
                    sam_model, 
                    item, boxes, 
                    clean_mask_holes, 
                    clean_mask_islands, 
                    mask_blur, 
                    mask_grow_shrink_factor, 
                    multimask, 
                    two_pass, device 
                )
            

            # sort output according to the prompt input
            if multimask is True: 
                phrases , masks, images =  sort_result_to_prompt(phrases, masks, images , prompt)

            
            # Formated Print
            if isinstance(logits, torch.Tensor):
                logits = logits.tolist()
            formatted_phrases = [f"{phrase} ({logit:.4f})" for phrase, logit in zip(phrases, logits)]
            formatted_output = ', '.join(formatted_phrases)
            print(f"\033[1;32m(dnl13-seg)\033[0m > phrase(confidence): {formatted_output}")

            # add results to output
            res_images.extend(images)
            res_masks.extend(masks)
            res_debug_images.extend(debug_image)
        



        if not res_images:
            print("Die Liste ist leer!")
        else:
            print("Die Liste hat Elemente.")


        res_images = torch.cat(res_images, dim=0)
        res_masks = torch.cat(res_masks, dim=0)


        """

        # if nothing was detected just send simple input image and empty mask
        if detection_errors is not False:
            print("\033[1;32m(dnl13-seg)\033[0m The tensor 'boxes' is empty. No elements were found in the image search.")
            res_images.append(image)
            res_masks.append(empty_mask)
            res_debug_images.extend(image)

        try:
            # Dein Code, der den Fehler auslöst
            res_images = torch.cat(res_images, dim=0)
 
        except RuntimeError as e:
            if "Sizes of tensors must match" in str(e):
                raise Exception("No mask found for some images in the batch. Consider reducing the 'box_threshold' parameter.")
            else:
                # Andere RuntimeErrors, die hier nicht erwartet werden
                raise Exception("Another error occurred:", e)
        # generate output
        #res_images = torch.cat(res_images, dim=0)
        res_masks = torch.cat(res_masks, dim=0)
        #res_debug_images = torch.cat(debug_image, dim=0)

        # fix for ComfyUI Mask format :MASK torch.Tensor with shape [H,W] or [B,C,H,W]
        if multimask:
            res_masks = res_masks.unsqueeze_(dim=1) 


        """
      
        return (res_images, res_masks, res_debug_images, )

