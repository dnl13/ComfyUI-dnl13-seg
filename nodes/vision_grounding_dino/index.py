
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from ...utils.helper_device import list_available_devices, get_device
from ...utils.helper_cmd_and_path import print_labels
from .grounding_dino_predict import groundingdino_predict

dnl13 = print_labels("label")

class DinoSegmentationProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        available_devices = list_available_devices()
        return {
            "required": {
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {"default": "arms, legs, eyes, hair, head", "multiline": True}),
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
                "dedicated_device": (available_devices, ),
            },
        }
    CATEGORY = "dnl13/vision"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("RGBA_Images", "RGB_Images (with Background Color)",
                    "Masks", "DINO Image Detections Debug (rgb)")

    def main(
            self,
            grounding_dino_model,
            image,
            prompt,
            box_threshold,
            upper_confidence_threshold,
            lower_confidence_threshold,
            dedicated_device="Auto"
    ):

        device = get_device(dedicated_device)
        print(dedicated_device)
        #
        grounding_dino_model.to(device)
        grounding_dino_model.eval()

        #
        img_batch_size, img_height, img_width, img_channel = image.shape
        #
        res_images_rgba, res_images_rgb, res_masks, res_boxes, res_debug_images, unique_phrases = [], [], [], [], [], []
        

        if prompt.endswith(","):
            prompt = prompt[:-1]
        prompt = prompt.replace(",", " .")
        prompt = prompt.lower()
        prompt = prompt.strip()
        if not prompt.endswith("."):
            prompt = prompt + "."
        #
        prompt_list = prompt.split('.')
        prompt_list = [e.strip() for e in prompt_list]
        prompt_list.pop()
        #
        print(f"{dnl13} got prompt like this:", prompt_list)
        #
        output_mapping = {}
        for index, item in tqdm(enumerate(image), total=len(image), desc=f"{dnl13} analyizing batch", unit="img", colour='green'):
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes, phrases, logits = groundingdino_predict(grounding_dino_model, item, prompt, box_threshold, upper_confidence_threshold, lower_confidence_threshold, device)


        """
        INFO: 
        what should be returned: 

        - bboxes 
        - cropped images 
        - debug images 
        - SEGS
          - SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

        """


        return (res_images_rgba, res_images_rgb, res_masks, res_debug_images, )
