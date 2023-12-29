
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from ...utils.helper_device import list_available_devices, get_device
from ...utils.helper_cmd_and_path import print_labels
from ...utils.helper_img_utils import createDebugImage, hex_to_rgb, blend_rgba_with_background, blur_cvmask, img_combine_mask_rgba
from .grounding_dino_predict import groundingdino_predict, get_unique_phrases, prompt_to_dino_prompt, VAEencode_dnl13
from server import PromptServer

dnl13 = print_labels("dnl13")


def create_mask_from_bboxes(height, width, bboxes):
    """
    Erstellt eine binäre Maske, bei der die Bereiche innerhalb der Bounding Boxes mit 1 gefüllt sind.

    :param height: Die Höhe der Maske.
    :param width: Die Breite der Maske.
    :param bboxes: Eine Liste von Bounding Boxes, jede Box als [x_min, y_min, x_max, y_max].
    :return: Ein Tensor der Form [H,W], der die Maske repräsentiert.
    """
    # Initialisiere den Masken-Tensor mit Nullen
    mask = torch.zeros((height, width), dtype=torch.uint8)

    # Fülle die Bereiche innerhalb der Bounding Boxes
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        mask[y_min:y_max, x_min:x_max] = 1

    return mask


def adjust_bboxes(bboxes, factor, img_width, img_height):
    """
    Passt die Größe der Bounding Boxes an, indem sie vergrößert oder verkleinert werden.

    :param bboxes: Tensor von Bounding Boxes der Form [n, 4] mit [x_min, y_min, x_max, y_max].
    :param factor: Der Faktor, um den die Bounding Boxes vergrößert oder verkleinert werden sollen.
    :param img_width: Die Breite des Bildes.
    :param img_height: Die Höhe des Bildes.
    :return: Ein Tensor von angepassten Bounding Boxes der gleichen Form.
    """
    # Berechnen der Veränderung für x und y basierend auf dem Faktor
    width_factor = (bboxes[:, 2] - bboxes[:, 0]) * factor / 1024
    height_factor = (bboxes[:, 3] - bboxes[:, 1]) * factor / 1024

    # Erstelle neue Bounding Boxes
    new_bboxes = torch.stack([
        (bboxes[:, 0] - width_factor).clamp(0, img_width),
        (bboxes[:, 1] - height_factor).clamp(0, img_height),
        (bboxes[:, 2] + width_factor).clamp(0, img_width),
        (bboxes[:, 3] + height_factor).clamp(0, img_height)
    ], dim=1)

    return new_bboxes


def create_masked_tensors_for_each_frame(data_by_phrase):
    for phrase, frames in data_by_phrase.items():
        for frame_data in frames:
            # Annahme: img_ts ist der erste Eintrag in der Liste
            img_ts = frame_data['img_ts'][0]
            bboxes = frame_data['bbox'][0]

            # Maskiere das Bild für alle BBoxes
            masked_img_ts = mask_image(img_ts, bboxes)

            # Aktualisiere das maskierte Bild im frame_data
            frame_data['masked_img_ts'] = masked_img_ts

    return data_by_phrase


def extract_tensors(data_by_phrase):
    tensors_by_phrase = {}

    for phrase, frames in data_by_phrase.items():
        # Sammle alle maskierten Bilder und Logits für die aktuelle Phrase
        masked_images = [frame_data['masked_img_ts']
                         for frame_data in frames if 'masked_img_ts' in frame_data]
        bboxes = [frame_data['bbox'][0]
                  for frame_data in frames if 'bbox' in frame_data]
        logits = [frame_data['logits'][0]
                  for frame_data in frames if 'logits' in frame_data]

        # Speichere die Daten in einem Dictionary
        tensors = {}
        if masked_images:
            tensors['img'] = torch.stack(masked_images)
        if bboxes:
            tensors['bbox'] = bboxes  # Verwende eine Liste anstatt zu stapeln
        if logits:
            tensors['logits'] = logits

        tensors_by_phrase[phrase] = tensors if tensors else None

    return tensors_by_phrase


def mask_image(tensor_image, bboxes):
    # Initialisiere eine Maske mit False, die die gleiche Höhe und Breite wie das Bild hat
    height, width, _ = tensor_image.shape
    mask = torch.zeros((height, width), dtype=torch.bool)

    # Füge einen Alpha-Kanal zum Tensor hinzu, initialisiere ihn mit 255 (undurchsichtig)
    alpha_channel = torch.ones((height, width), dtype=torch.uint8) * 255

    # Erstelle eine Maske für jede BBox und kombiniere sie
    for box in bboxes:
        x_min, y_min, x_max, y_max = box.int()
        mask[y_min:y_max, x_min:x_max] = True

    # Setze den Alpha-Kanal auf 0 (transparent) für alle Pixel außerhalb der Masken
    alpha_channel[~mask] = 0

    # Kombiniere die ursprünglichen Bildkanäle mit dem Alpha-Kanal
    masked_tensor_image = torch.cat(
        (tensor_image, alpha_channel.unsqueeze(2)), dim=2)

    return masked_tensor_image


def collect_data_for_phrases(output_mapping, unique_phrases):
    data_by_phrase = {phrase: [] for phrase in unique_phrases}

    for frame in output_mapping:
        for phrase in unique_phrases:
            if phrase in frame:
                # Füge das komplette Daten-Dictionary für die Phrase hinzu
                data_by_phrase[phrase].append(frame[phrase])

    return data_by_phrase


class DinoSegmentationProcessor:

    """
    DYNAMIC OUTPUTS EXPERIMENT
    For later use if I have figured out how to connect js dynamic outputs to python return_types

    dynamic_outputs = {
        "default":[
            { "name":"RGBA_Images", "type" :"IMAGE"},
            { "name":"RGB_Images (with Background Color)", "type" :"IMAGE"},
            { "name":"combined Masks", "type" :"MASK"},
            { "name":"DINO Image Detections Debug (rgb)", "type" :"IMAGE"},
        ],
        "detected":[]
    }

    def get_output_names( dynamic_outputs ):
        current_list = []
        for name in dynamic_outputs["default"]:
            current_list.append(name["name"])

        if dynamic_outputs["detected"]:
            for name in dynamic_outputs["default"]:
                print(name["name"])
                current_list.append(name["name"])
        return tuple(current_list)

    def get_output_types( dynamic_outputs ):
        current_list = []
        for type in dynamic_outputs["default"]:
            current_list.append(type["type"])
        if dynamic_outputs["detected"]:
            for type in dynamic_outputs["default"]:
                print(type["type"])
                current_list.append(type["type"])
        return tuple(current_list)
    """

    @classmethod
    def INPUT_TYPES(cls):
        available_devices = list_available_devices()
        return {
            "required": {
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
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
                "prompt": ("STRING", {"default": "arms, legs, eyes, hair, head", "multiline": True}),
                "background_color": ("STRING", {"default": "#00FF00", "multiline": False}),
                "mask_grow_shrink_factor": ("INT", {
                    "default": 0,
                    "min": -1024,
                    "max": 1024,
                    "step": 1
                }),
                "mask_blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1
                }),
                "latent_grow_mask": ("INT", {
                    "default": 16,
                    "min": 0,
                    "step": 1
                }),
                "mask_invert":('BOOLEAN', {"default":False}),
            },
            "optional": {"vae": ("VAE", ), },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }
    CATEGORY = "dnl13/vision"
    FUNCTION = "main"

    """
    DYNAMIC OUTPUTS EXPERIMENT
    for later use 
    RETURN_TYPES = get_output_types(dynamic_outputs)
    RETURN_NAMES = get_output_names(dynamic_outputs)
    """
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE",
                    "LATENT", "BBOX", "DINO_DETECTION_COLLECTION")
    RETURN_NAMES = ("RGBA_Images", "RGB_Images (with Background Color)", "Combined Masks",
                    "DINO Image Detections Debug (rgb)", "inpaint Latent", "BBOX" ,"Dino Collection")

    def main(
            self,
            grounding_dino_model=None,
            image=None,
            prompt="",
            box_threshold=0.35,
            upper_confidence_threshold=1,
            lower_confidence_threshold=0.01,
            unique_id=None,
            background_color="#00FF00",
            dedicated_device="Auto",
            mask_grow_shrink_factor=0,
            mask_blur=0,
            latent_grow_mask=16,
            mask_invert=False,
            vae=None,
    ):
        """
        DYNAMIC OUTPUTS EXPERIMENT

        # Reset Ouputs 
        self.dynamic_outputs["detected"] = []
        """

        #
        device = get_device(dedicated_device)
        #
        grounding_dino_model.to(device)
        grounding_dino_model.eval()

        #
        img_batch_size, img_height, img_width, img_channel = image.shape
        #
        bg_color_rgb = hex_to_rgb(background_color)
        bg_color_tensor = torch.tensor(
            bg_color_rgb, dtype=torch.float32) / 255.0
        bg_color_tensor = bg_color_tensor.view(1, 1, 1, 3)

        #
        prompt, prompt_list = prompt_to_dino_prompt(prompt)

        # Erstelle ein leere BBox
        empty_box = torch.zeros((1, 4)) .to(device)

        # Erstelle ein leere Logits Tensor
        empty_logits = torch.zeros((1, 1)) .to(device)

        #
        print(f"{dnl13} got prompt like this:", prompt_list)
        #

        res_bboxes, res_segs = [], []
        res_debug_images = []
        res_cropped_img_rgb, res_img_rgba = [], []
        res_latent_inpaint, res_masks = [], []

        res_cropped_img_fullsize_pil = []

        output_mapping = [None] * len(image)
        for index, item in tqdm(enumerate(image), total=len(image), desc=f"{dnl13} analyzing batch", unit="img", ncols=100, colour='green'):
            item_dino = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            bbox, phrases, logits = groundingdino_predict(
                grounding_dino_model, item_dino, prompt, box_threshold, upper_confidence_threshold, lower_confidence_threshold, device)

            bbox = adjust_bboxes(bbox, mask_grow_shrink_factor, img_width, img_height)
            res_bboxes.append(bbox)

            phrase_boxes = {}
            for p in phrases:
                phrase_boxes[p] = {'bbox': [bbox], 'logits': [logits], "phrase": [phrases], "img_pil": [item_dino], "img_ts": [item]}
            output_mapping[index] = phrase_boxes


            ############### Do Output Stacks

            # Mask 
            mask = create_mask_from_bboxes(img_height, img_width, bbox).float()
            if mask_invert is not False:
                mask = 1 - mask
            mask_np = np.expand_dims(mask.detach().cpu().numpy(), axis=-1)
            mask_np_blurred = np.expand_dims(blur_cvmask( mask_np, mask_blur ), axis=-1)
            mask = torch.tensor(mask_np_blurred).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # MASK shape [B,C,H,W]
            res_masks.append(mask.unsqueeze(0))

            # Image RGBA 
            masked_image_rgba = torch.tensor(img_combine_mask_rgba(item, mask_np_blurred)).unsqueeze(0)
            res_img_rgba.append(masked_image_rgba)

            # Image with Background RGB
            masked_image_with_bg = blend_rgba_with_background(masked_image_rgba.to(device), bg_color_tensor.to(device))
            res_cropped_img_rgb.append(masked_image_with_bg)

        if vae is not None:
            for index, item in tqdm(enumerate(image), total=len(image), desc=f"{dnl13} generate Latent", unit="img", ncols=100, colour='#FFC0CB'):
                encoded_tensor = VAEencode_dnl13(vae=vae, image=item.unsqueeze(0), mask=res_masks[index].squeeze(0), latent_grow_mask=latent_grow_mask, device=device)
                res_latent_inpaint.append(encoded_tensor)

            # Sammeln der Tensoren
            samples_list = [d["samples"] for d in res_latent_inpaint]
            noise_mask_list = [d["noise_mask"] for d in res_latent_inpaint]

            # Verkettung der Tensoren entlang der Batch-Dimension
            concatenated_samples = torch.cat(samples_list, dim=0).to("cpu")
            concatenated_noise_masks = torch.cat(noise_mask_list, dim=0).to("cpu")

            # Erstellen eines neuen Wörterbuchs mit den kombinierten Tensoren
            final_dict = { "samples": concatenated_samples, "noise_mask": concatenated_noise_masks }
            res_latent_inpaint = final_dict
        else:

            empty_latent = torch.zeros([img_batch_size, 4, img_height // 8, img_width // 8], device=device)
            res_latent_inpaint = {"samples": empty_latent}




        #-------------------------------------------- Dino Collection
        # Einzigartige Phrases finden
        unique_phrases = get_unique_phrases(output_mapping, prompt_list)
        print(f"{dnl13} found following unique_phrases in prompt:", unique_phrases)

        output_mapping_phrase_sorted = collect_data_for_phrases(
            output_mapping, unique_phrases)
        output_mapping_phrase_sorted = create_masked_tensors_for_each_frame(
            output_mapping_phrase_sorted)
        tensors_by_phrase = extract_tensors(output_mapping_phrase_sorted)

        # Ausgabe zum Testen

        for index, output_item in enumerate(output_mapping):
            # Nicht vorhandene boxen und logits mit leeren auffüllen
            for phrase in unique_phrases:
                if phrase not in output_item:
                    output_item[phrase] = {
                        'box': [empty_box],
                        'logits': [empty_logits]
                    }

            # Nehme nur das erste Element (oder einen bestimmten Schlüssel) aus output_item
            first_key = next(iter(output_item))
            item = output_item[first_key]

            img = item["img_pil"][0]
            bbox = item["bbox"][0]  # Erstes Element von bbox
            phrase_list = item["phrase"][0]  # Erstes Element von phrase
            logits = item["logits"][0]  # Erstes Element von logits

            # Erstelle Debug-Bild für das erste Element
            debug_image = createDebugImage(
                img, bbox, phrase_list[0], logits, False)
            res_debug_images.append(debug_image)

        """
        DYNAMIC OUTPUTS EXPERIMENT

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

        






        res_bboxes = torch.cat( res_bboxes, dim=0).to("cpu")
        res_images_rgba = torch.cat(res_img_rgba, dim=0).to("cpu")
        res_images_rgb = torch.cat(res_cropped_img_rgb, dim=0).to("cpu")
        res_masks = torch.cat(res_masks, dim=0).to("cpu")
        res_debug_images = torch.cat(res_debug_images, dim=0).to("cpu")

        res_dino_collection = [output_mapping]

        return (res_images_rgba, res_images_rgb, res_masks, res_debug_images, res_latent_inpaint,res_bboxes, res_dino_collection, )
