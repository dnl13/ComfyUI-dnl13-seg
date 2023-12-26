import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from ...utils.helper_device import list_available_devices, get_device
from ...utils.helper_cmd import print_labels
from ...utils.helper_img_utils import hex_to_rgb, blend_rgba_with_background, createDebugImage
from ...utils.comfy_vae_encode_inpaint import VAEencode
from ..vision_grounding_dino.grounding_dino_predict import groundingdino_predict
from ..vision_segment_anything.sam_auto_segmentation import sam_segment

#Print Label
dnl13 = print_labels("dnl13")

"""
TODO: 
  - besseres output_mapping handling
  - performance ab ~75 batch_size bricht ein
  - clipseq für besser sam_hints ?  (tests needed)
"""
 
class LazyMaskSegmentation:
    @classmethod
    def INPUT_TYPES(cls):
        available_devices = list_available_devices()
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {"default": "arms, legs, eyes, hair, head", "multiline": True}),
                "bbox_threshold": ("FLOAT", {
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

                "multimask": ('BOOLEAN', {"default": False}),

                "two_pass": ('BOOLEAN', {"default": False}),

                "sam_contrasts_helper": ("FLOAT", {
                    "default": 1.25,
                    "min": 0,
                    "max": 3.0,
                    "step": 0.01
                }),
                "sam_brightness_helper": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "sam_hint_threshold_helper": ("FLOAT", {
                    "default": 0.00,
                    "min": -1.00,
                    "max": 1.00,
                    "step": 0.001
                }),
                "sam_helper_show": ('BOOLEAN', {"default": False}),

                "dedicated_device": (available_devices, ),

                "clean_mask_holes": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 100000,
                    "step": 1
                }),

                "clean_mask_islands": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 100000,
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
                    "max": 1024,
                    "step": 1
                }),

                "background_color": ("STRING", {"default": "#00FF00", "multiline": False}),
                
            },
            "optional":{
                "vae": ("VAE", ),
            }
        }
    CATEGORY = "dnl13"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE", "LATENT",)
    RETURN_NAMES = ("RGBA_Images", "RGB_Images (with Background Color)", "Masks", "DINO Image Detections Debug (rgb)", "Latent for Inapaint (VAE required)")

    def main(
        self,
        vae=None, # Optional
        sam_model=None,
        grounding_dino_model=None,
        image=None,
        prompt="arms, legs, eyes, hair, head",
        bbox_threshold=0.35,
        lower_confidence_threshold=0.01,
        upper_confidence_threshold=1,
        multimask=False,
        two_pass=False,
        sam_contrasts_helper=1.25,
        sam_brightness_helper=0,
        sam_hint_threshold_helper=0.00,
        sam_helper_show=False,
        dedicated_device="Auto",
        clean_mask_holes=64,
        clean_mask_islands=64,
        mask_blur=0,
        mask_grow_shrink_factor=0,
        background_color="#00FF00"
    ):
        # Zu ausführendes Gerät auswählen und setzen
        device = get_device(dedicated_device)

        # Bild Tensor informationen
        img_batch, img_height, img_width, img_channel = image.shape
               
        # Konvertiere den Hex-Farbcode direkt in einen normalisierten Tensor
        bg_color_rgb = hex_to_rgb(background_color)
        bg_color_rgb_normalized = torch.tensor([value / 255.0 for value in bg_color_rgb], dtype=torch.float32)
        bg_color_tensor = torch.tensor([value / 255.0 for value in bg_color_rgb], dtype=torch.float32).to(device)
        bg_color_tensor = bg_color_tensor.view(1, 1, 1, 3).expand(1, img_height, img_width, 3)
        
        # Erstelle ein transparentes Bild 
        transparent_image = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        transparent_image_np = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        transparent_image_tensor = torch.zeros(1, img_height, img_width, 4, dtype=torch.float32)
        
        # Erstelle ein leere Maske 
        transparent_maske_tensor = torch.zeros( 1, img_height, img_width).to(device)

        # Erstelle ein leere BBox
        empty_box = torch.zeros((1, 4)) .to(device)

        # Erstelle ein leere Logits Tensor
        empty_logits = torch.zeros((1, 1)) .to(device)
        
        # Eingabe Prompt für Verarbeitung vorbereiten 
        prompt = prompt.strip().lower()
        prompt_list = [p.strip() for p in prompt.split(',') if p.strip()]

        # Den User über unser Prompt informieren: 
        print(f"{dnl13} got prompt like this:", prompt_list)

        # Nötige Listen initialisieren die für den Prozzess benötigt werden
        res_images_rgba, res_images_rgb, res_masks = [], [], []
        res_boxes, res_debug_images = [], []
        sam_hint_threshold_points, sam_hint_threshold_labels = [], []

        # Sammel Dict für Ouptun gennerierung
        output_mapping = {}

        # Modelle auf Geräte verschieben und in evaluierung schalten
        grounding_dino_model.to(device)
        grounding_dino_model.eval()
        sam_model.to(device)
        sam_model.eval()

        for index, item in tqdm(enumerate(image), total=len(image), desc=f"{dnl13} analyizing batch", unit="img", ncols=100, colour='green'):
            item = Image.fromarray(np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes, phrases, logits = groundingdino_predict(grounding_dino_model, item, prompt, bbox_threshold, upper_confidence_threshold, lower_confidence_threshold, device)           

            phrase_boxes = {}
            for p in phrases:
                phrase_boxes[p] = {'box': [], 'mask': [], 'image': [], 'logits':[]}

            for i, phrase in enumerate(phrases):
                sam_masks, masked_image, sam_grid_points, sam_grid_labels  = sam_segment(sam_model, item, boxes[i], clean_mask_holes, clean_mask_islands, mask_blur, mask_grow_shrink_factor, two_pass, sam_contrasts_helper, sam_brightness_helper,sam_hint_threshold_helper, sam_helper_show, device)
                masked_image = masked_image.unsqueeze(0)
                phrase_boxes[phrase]['box'].append(boxes[i].to(device))
                phrase_boxes[phrase]['mask'].append(sam_masks.to(device))
                phrase_boxes[phrase]['image'].append(masked_image.to(device))
                phrase_boxes[phrase]['logits'].append(logits[i].to(device))
                sam_hint_threshold_points.append(sam_grid_points)
                sam_hint_threshold_labels.append(sam_grid_labels)

            output_mapping[index] = phrase_boxes

            tensor_image_formated = createDebugImage( 
                item, 
                boxes, 
                phrases, 
                logits, 
                toggle_sam_debug=sam_helper_show, 
                sam_contrasts_helper=sam_contrasts_helper, 
                sam_brightness_helper=sam_brightness_helper, 
                sam_hint_grid_points=sam_hint_threshold_points,
                sam_hint_grid_labels=sam_hint_threshold_labels
                )
            res_debug_images.extend(tensor_image_formated)
  

        phrase_mask_count = defaultdict(lambda: defaultdict(int))
        for index, data in output_mapping.items():
            for phrase, values in data.items():
                num_masks = len(values['mask'])
                phrase_mask_count[phrase][index] = num_masks
        max_masks_per_phrase = {
            phrase: {'frame': max(counts, key=counts.get), 'num_masks': max(counts.values())}
            for phrase, counts in phrase_mask_count.items()
        }
        max_found_masks = 0
        for info in max_masks_per_phrase.values():
            if info['num_masks'] > max_found_masks:
                max_found_masks = info['num_masks']
        print(f"{dnl13} maximum masks per frame depending on prompt detection:", max_found_masks)

        # Erstelle eine Liste mit allen Phrasen aus dem output_mapping
        all_phrases = set()
        for index, data in output_mapping.items():
            all_phrases.update(data.keys())

        # Durchlaufe die prompt_list und überprüfe, ob die Phrasen im output_mapping oder als Teilphrasen vorhanden sind
        for phrase in prompt_list:
            # Überprüfe, ob die Phrase im output_mapping oder als Teilphrase vorhanden ist
            found = False
            for existing_phrase in all_phrases:
                if phrase == existing_phrase or phrase in existing_phrase or existing_phrase in phrase:
                    unique_phrases.append(existing_phrase)
                    found = True
                    break

            # Falls die Phrase nicht gefunden wurde, füge sie zur eindeutigen Phrasenliste hinzu
            if not found:
                unique_phrases.append(phrase)

        # Entferne doppelte Einträge und behalte die Reihenfolge bei
        unique_phrases = list(dict.fromkeys(unique_phrases))


        # Durchlaufe das output_mapping und füge fehlende Phrasen hinzu, dupliziere Elemente und sortiere dann
        for index, data in output_mapping.items():
            # Füge fehlende Phrasen hinzu
            for phrase in unique_phrases:
                if phrase not in data:
                    data[phrase] = {
                        'mask': [transparent_maske_tensor],
                        'image': [transparent_image_tensor],
                        'box': [empty_box],
                        'logits': [empty_logits]
                    }

            # Stelle sicher, dass jede Phrase die gleiche Anzahl an Elementen hat
            for phrase, items in data.items():
                current_masks = len(items['mask'])
                if current_masks < max_found_masks:
                    # Berechne, wie viele Duplikate benötigt werden
                    duplicates_needed = max_found_masks - current_masks

                    # Dupliziere die vorhandenen Daten
                    items['mask'].extend(items['mask'][:1] * duplicates_needed)
                    items['image'].extend(items['image'][:1] * duplicates_needed)
                    items['box'].extend(items['box'][:1] * duplicates_needed)
                    items['logits'].extend(items['logits'][:1] * duplicates_needed)

            # Sortiere die Phrasen gemäß der Reihenfolge in unique_phrases
            data_sorted = {phrase: data[phrase] for phrase in unique_phrases if phrase in data}
            output_mapping[index] = data_sorted

        print(f"{dnl13} found following unique_phrases in prompt:", unique_phrases)

        for index in sorted(output_mapping.keys()):

            if multimask:
                # Füge alle Masken und Bilder separat hinzu, wenn multimask aktiviert ist
                for phrase in unique_phrases:
                    if phrase in output_mapping[index]:
                        res_images_rgba.extend(output_mapping[index][phrase]['image'])
                        res_masks.extend(output_mapping[index][phrase]['mask'])

                        # Erstellen Sie hier den Hintergrund-Tensor
                        for img_rgba in output_mapping[index][phrase]['image']:
                            rgb_image_tensor = blend_rgba_with_background(img_rgba.to(device), bg_color_tensor.to(device))
                            res_images_rgb.append(rgb_image_tensor)  # Nur RGB-Kanäle hinzufügen

            else:
                # Kombiniere Masken und Bilder für das aktuelle Frame
                combined_mask = None
                combined_image = None

                for phrase in unique_phrases:
                    if phrase in output_mapping[index]:
                        # Kombiniere Masken
                        for mask in output_mapping[index][phrase]['mask']:
                            if combined_mask is None:
                                combined_mask = mask
                            else:
                                combined_mask = torch.max(combined_mask, mask)

                        # Kombiniere Bilder mit Max-Pooling
                        for img in output_mapping[index][phrase]['image']:
                            if combined_image is None:
                                combined_image = img
                            else:
                                combined_image = torch.max(combined_image, img)

                # Füge das kombinierte Bild und die kombinierte Maske hinzu
                if combined_image is not None:
                    res_images_rgba.append(combined_image)

                    rgb_image_tensor = blend_rgba_with_background(combined_image.to(device), bg_color_tensor.to(device))
                    res_images_rgb.append(rgb_image_tensor)  # Nur RGB-Kanäle hinzufügen

                if combined_mask is not None:
                    res_masks.append(combined_mask)

                if vae is not None:
                    if mask_grow_shrink_factor <= 6:
                        mask_grow_shrink_factor = 6 
                    let_test = VAEencode( vae=vae, pixels=image, mask=res_masks[0], device=device, grow_mask_by=mask_grow_shrink_factor)
                    res_inpaint_latent.append(let_test)

        #VAEencode( vae, pixels, mask, grow_mask_by=6))
        if vae is not None and multimask is False:
            res_inpaint_latent = res_inpaint_latent[0]
        else: 
            empty_latent = torch.zeros([img_batch, 4, img_height // 8, img_width // 8], device=device)
            res_inpaint_latent= {"samples":empty_latent}
        
        res_images_rgba = torch.cat(res_images_rgba, dim=0).to("cpu")
        res_images_rgb = torch.cat(res_images_rgb, dim=0).to("cpu")
        res_masks = torch.cat(res_masks, dim=0).to("cpu")

        return (res_images_rgba, res_images_rgb, res_masks, res_debug_images, res_inpaint_latent, )
