import torch
import math
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from ..node_fnc.node_dinosam_prompt import sam_segment, groundingdino_predict, sam_segment_new, enhance_edges
from ..utils.collection import check_mps_device, print_labels
from ..utils.image_processing import hex_to_rgb
import re
from collections import defaultdict
import comfy.model_management
from tqdm import tqdm

dnl13 = print_labels("label")

"""
from Impact Pack:

what are SEG
SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

"""

def VAEencode(vae, pixels, mask, device, grow_mask_by=6):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        #grow mask by a few pixels to keep things seamless in latent space
        if grow_mask_by == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)
            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.to(device).round(), kernel_tensor.to(device), padding=padding), 0, 1)

        m = (1.0 - mask.round()).squeeze(1)
        pixels = pixels.to(device)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        t = vae.encode(pixels)
        #return t
        return {"samples":t, "noise_mask": (mask_erosion[:,:,:x,:y].round())}

def createDebugImage( image, dino_bbox, dino_pharses, dino_logits, toggle_sam_debug, sam_contrasts_helper, sam_brightness_helper, sam_hint_grid_points, sam_hint_grid_labels):
    width, height = image.size
    debug_image = image.copy().convert('RGB')
    debug_image_np = np.array(debug_image)

    if toggle_sam_debug:
        # enhance_edges()
        debug_image_np = enhance_edges(debug_image_np, alpha=sam_contrasts_helper, beta=sam_brightness_helper, edge_alpha=1.0)
        debug_image = Image.fromarray(debug_image_np)

        if sam_hint_grid_points != [None] and sam_hint_grid_points != [None]: 

            # Erstellen einer transparenten Ebene
            transparent_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(debug_image)

            for index, bbox in enumerate(dino_bbox):
                x_min, y_min, x_max, y_max = map(int, bbox)
                box_width = x_max - x_min
                box_height = y_max - y_min

                # Definieren Sie die Schrittgrößen für die Grid-Erstellung
                x_step = max(3, int(box_width / 20))
                y_step = max(3, int(box_height / 20))

                # Erstellen Sie Punkte innerhalb der Bounding Box
                grid_points = []
                for y in range(y_min, y_max, y_step):
                    for x in range(x_min, x_max, x_step):
                        grid_points.append((x, y))
                    if sam_hint_grid_labels[index] is not None:
                        for point, label in zip(grid_points, sam_hint_grid_labels[index]):
                            #print( f" {point} : {label}" )
                            x, y = point
                            if label > 0:
                                color = (0, 255, 0, 128)  # Halbtransparentes Grün
                            else:
                                color = (255, 0, 0, 128)  # Halbtransparentes Rot
                            # Zeichnen eines halbtransparenten Kreises
                            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
            # Kombinieren der transparenten Ebene mit dem Originalbild
            image_with_transparent_layer = Image.alpha_composite(debug_image.convert('RGBA'), transparent_layer)
            debug_image = image_with_transparent_layer
        # Anzeigen des Bildes
        #image_with_transparent_layer.show()
    else:
        debug_image = Image.fromarray(debug_image_np)

    draw = ImageDraw.Draw(debug_image)
    text_padding = 2.5  # Anpassbares Padding
    for bbox, conf, phrase in zip(dino_bbox, dino_logits, dino_pharses):
        bbox = tuple(map(int, bbox))
        draw.rectangle(bbox, outline="red", width=2)
        
        # Erstelle das Rechteck für den Hintergrund des Textes mit Padding
        text_bg_x1 = bbox[0]
        text_bg_y1 = bbox[1]
        text_bg_x2 = bbox[0] + 120  # Beispielbreite des Hintergrunds
        text_bg_y2 = bbox[1] + 20   # Beispielhöhe des Hintergrunds
        draw.rectangle(((text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2)), fill="red")  # Hintergrund für den Text

        # Textkoordinaten mit Padding
        text_x = bbox[0] + text_padding
        text_y = bbox[1] + text_padding

        draw.text((text_x, text_y), f"{phrase}: {conf:.2f}", fill="white")  # Weißer Text auf rotem Hintergrund

    transform = transforms.ToTensor()
    tensor_image = transform(debug_image)
    tensor_image_expanded = tensor_image.unsqueeze(0)
    tensor_image_formated = tensor_image_expanded.permute(0, 2, 3, 1)
    
    return tensor_image_formated



def blend_rgba_with_background(rgba_tensor, bg_color_tensor):
    """
    Kombiniert ein RGBA-Bild mit einem RGB-Hintergrund und gibt ein RGB-Bild zurück.

    :param rgba_tensor: Ein RGBA-Bildtensor der Form [1, Höhe, Breite, 4].
    :param bg_color_tensor: Ein RGB-Hintergrundtensor der Form [3, Höhe, Breite].
    :return: Ein RGB-Bildtensor der Form [1, Höhe, Breite, 3].
    """
    # Extrahieren Sie den RGB-Teil und den Alpha-Kanal des Bildes
    rgb = rgba_tensor[:, :, :, :3]
    alpha = rgba_tensor[:, :, :, 3:4].squeeze(0)  # Entfernen Sie die Batch-Dimension

    # Führen Sie das Alpha-Blending durch
    blended_rgb = alpha * rgb + (1 - alpha) * bg_color_tensor

    return blended_rgb


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
                "sam_hint_threshold_helper":("FLOAT", {
                    "default": 0.00,
                    "min": -1.00,
                    "max": 1.00,
                    "step": 0.001
                }),
                "sam_helper_show":('BOOLEAN', {"default":False}),
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
                "background_color": ("STRING", {"default": "#00FF00","multiline": False}),
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
            sam_contrasts_helper, 
            sam_brightness_helper,
            sam_hint_threshold_helper,
            sam_helper_show,
            background_color,
            optimize_prompt_for_dino=False,
            multimask=False, 
            dedicated_device="Auto",
            vae=None,
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
        grounding_dino_model.to(device)
        grounding_dino_model.eval()
        sam_model.to(device)
        sam_model.eval()
        #
        img_batch, img_height, img_width, img_channel = image.shape       
        empty_mask = torch.zeros((1, 1, img_height, img_width), dtype=torch.float32) # [B,C,H,W]
        empty_mask = empty_mask / 255.0
        #
        res_images_rgba , res_images_rgb ,res_masks , res_boxes, res_debug_images, unique_phrases, res_inpaint_latent = [],[],[],[],[],[],[]

        sam_hint_threshold_points, sam_hint_threshold_labels = [], []
        #
        bg_color_rgb = hex_to_rgb(background_color)
        bg_color_rgb_normalized = torch.tensor([value / 255.0 for value in bg_color_rgb], dtype=torch.float32)

        #
        height, width = image.size(1), image.size(2)
        #
        transparent_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        transparent_image_np = np.array(transparent_image)
        transparent_image_tensor = torch.from_numpy(transparent_image_np).permute(2, 0, 1).unsqueeze(0)
        transparent_image_tensor = transparent_image_tensor.permute(0, 2, 3, 1).to(device)
        transparent_maske_tensor = torch.zeros(1, height, width).to(device)
        empty_box = torch.zeros((1, 4)) .to(device)
        empty_logits = torch.zeros((1, 1)) .to(device)
        #
        bg_color_tensor = bg_color_rgb_normalized.clone().detach()
        bg_color_tensor = bg_color_tensor.view(1, 1, 1, 3)  # Form anpassen für Broadcasting
        bg_color_tensor = bg_color_tensor.expand(1, height, width, 3).to(device) 
        #
        if optimize_prompt_for_dino is not False:
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
        for index, item in tqdm(enumerate(image), total=len(image), desc=f"{dnl13} analyizing batch", unit="img", ncols=100, colour='green'):
            item = Image.fromarray(np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes, phrases, logits = groundingdino_predict(grounding_dino_model, item, prompt, box_threshold, upper_confidence_threshold, lower_confidence_threshold, device)           

            phrase_boxes = {}
            for p in phrases:
                phrase_boxes[p] = {'box': [], 'mask': [], 'image': [], 'logits':[]}

            for i, phrase in enumerate(phrases):
                sam_masks, masked_image, sam_grid_points, sam_grid_labels  = sam_segment_new(sam_model, item, boxes[i], clean_mask_holes, clean_mask_islands, mask_blur, mask_grow_shrink_factor, two_pass, sam_contrasts_helper, sam_brightness_helper,sam_hint_threshold_helper, sam_helper_show, device)
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
                #sam_hint_threshold_helper=sam_hint_threshold_helper,
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
        if vae is not None:
            res_inpaint_latent = res_inpaint_latent[0]
        else: 
            empty_latent = torch.zeros([img_batch, 4, img_height // 8, img_width // 8], device=device)
            res_inpaint_latent= {"samples":empty_latent}
        
        res_images_rgba = torch.cat(res_images_rgba, dim=0).to("cpu")
        res_images_rgb = torch.cat(res_images_rgb, dim=0).to("cpu")
        res_masks = torch.cat(res_masks, dim=0).to("cpu")



        return (res_images_rgba, res_images_rgb, res_masks, res_debug_images, res_inpaint_latent, )
