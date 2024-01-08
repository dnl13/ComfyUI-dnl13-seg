import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from ...utils.helper_device import list_available_devices, get_device
from ...utils.helper_cmd_and_path import print_labels
from ...utils.helper_img_utils import createDebugImageTensor, hex_to_rgb, blend_rgba_with_background, blur_cvmask, img_combine_mask_rgba
# , get_unique_phrases
from .grounding_dino_predict import groundingdino_predict, prompt_to_dino_prompt, VAEencode_dnl13, dino_image_preperation
from server import PromptServer
import comfy.model_management
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


def get_unique_phrases(data, prompt_list):
    # Sammle alle Phrasen aus 'data'
    all_phrases = set(entry['phrase'] for entry in data)

    # Überprüfe die prompt_list gegen die gesammelten Phrasen
    unique_phrases = []
    for phrase in prompt_list:
        found = False
        for existing_phrase in all_phrases:
            if phrase == existing_phrase or phrase in existing_phrase or existing_phrase in phrase:
                unique_phrases.append(existing_phrase)
                found = True
                break
        if not found:
            unique_phrases.append(phrase)

    # Entferne doppelte Einträge und behalte die Reihenfolge bei
    unique_phrases = list(dict.fromkeys(unique_phrases))
    return unique_phrases


class DinoSegmentationProcessor:
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
                "mask_invert": ('BOOLEAN', {"default": False}),
            },
            "optional": {"vae": ("VAE", ), },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }
    CATEGORY = "dnl13/vision"
    FUNCTION = "main"

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE", "LATENT", "BBOX", "DINO_DETECTION_COLLECTION")
    RETURN_NAMES = ("RGBA_Images", "RGB_Images (with Background Color)", "Combined Masks","DINO Image Detections Debug (rgb)", "inpaint Latent", "BBOX", "Dino Collection")

    def main(self, grounding_dino_model=None, image=None, prompt="", box_threshold=0.35, upper_confidence_threshold=1, lower_confidence_threshold=0.01, unique_id=None, background_color="#00FF00", dedicated_device="Auto", mask_grow_shrink_factor=0, mask_blur=0, latent_grow_mask=16, mask_invert=False, vae=None):
        #
        torch.cuda.empty_cache()
        #
        device = get_device(dedicated_device)
        #
        grounding_dino_model.to(device)
        #grounding_dino_model.eval()
        #
        img_batch_size, img_height, img_width, img_channel = image.shape
        #
        bg_color_rgb = hex_to_rgb(background_color)
        bg_color_tensor = torch.tensor(bg_color_rgb, dtype=torch.float32) / 255.0
        bg_color_tensor = bg_color_tensor.view(1, 1, 1, 3)

        #
        prompt, prompt_list = prompt_to_dino_prompt(prompt)

        # Erstelle ein leere Elemente
        alpha_channel = torch.ones((1, 1, 1, 1), dtype=torch.float32)  # Alpha-Wert von 1 für volle Deckkraft
        bg_color_tensor_with_alpha = torch.cat((bg_color_tensor, alpha_channel), dim=3)
        empty_img_rgba = torch.zeros((1, img_height, img_width, 4), dtype=torch.float32)

        empty_box = torch.zeros((1,4)) .to(device)
        empty_logits = torch.zeros((1)) .to(device)
        empty_mask = torch.zeros([1, 1, img_height, img_width], device=device)
        empty_mask_batched = torch.zeros([img_batch_size, 1, img_height, img_width], device=device)
        empty_latent = torch.zeros([1, 4, img_height // 8, img_width // 8], device=device)
        empty_latent_batched = torch.zeros([img_batch_size, 4, img_height // 8, img_width // 8], device=device)
        #
        print(f"{dnl13} got prompt like this:", prompt_list)
        #

        res_bboxes, res_segs = [], []
        res_latent_inpaint, res_masks = [], []

        # ------------------------------------------ Dino Detection and Data Preperation
        data = []
        data_by_frame, data_by_phrase = [], []
        for index, item in tqdm(enumerate(image), total=len(image), desc=f"{dnl13} analyzing batch", unit="img", ncols=100, colour='green'):
            bbox, phrases, logits = groundingdino_predict(grounding_dino_model, item, prompt, box_threshold, upper_confidence_threshold, lower_confidence_threshold, device) 
            frame_data = {
                "frame": index,
                "phrase": phrases,
                'bbox': bbox,
                "logits": logits
            }
            data_by_frame.append(frame_data)
            for i, p in enumerate(phrases):
                phrase_data = {
                    "frame": index,
                    "phrase": p,
                    'bbox': bbox[i].unsqueeze(0),
                    "logits": logits[i]
                }
                data_by_phrase.append(phrase_data)

        # Einzigartige Phrases finden
        unique_phrases = get_unique_phrases(data, prompt_list)
        print(f"{dnl13} found following unique phrases in prompt:", unique_phrases)

        # ---------------------------------------------- Generate normal Img/Masks-Outputs
        res_bboxes_ts = torch.Tensor()
        res_masks_ts = torch.empty(img_batch_size, 1, img_height, img_width)
        res_images_rgba_ts = torch.empty(img_batch_size, img_height, img_width, 4)
        res_images_rgb_ts = torch.empty(img_batch_size, img_height, img_width, img_channel)
        res_debug_images_ts = torch.empty(img_batch_size, img_height, img_width, img_channel)

        for index, item in tqdm(enumerate(image), total=len(image), desc=f"{dnl13} generate images and masks", unit="img", ncols=100, colour='blue'):

            # BBOX
            bbox = adjust_bboxes(data_by_frame[index]["bbox"], mask_grow_shrink_factor, img_width, img_height)
            res_bboxes.append(bbox.to("cpu"))
            
            # Mask
            mask = create_mask_from_bboxes(img_height, img_width, bbox).float()
            if mask_invert is not False:
                mask = 1 - mask
            mask_np = np.expand_dims(mask.detach().cpu().numpy(), axis=-1)
            mask_np_blurred = np.expand_dims( blur_cvmask(mask_np, mask_blur), axis=-1)
            mask = torch.tensor(mask_np_blurred).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # MASK shape [B,C,H,W]
            mask = mask.unsqueeze(0)
            res_masks_ts[index] = mask.to("cpu")

            # Image RGBA
            masked_image_rgba = torch.tensor(img_combine_mask_rgba(item, mask_np_blurred)).unsqueeze(0)
            res_images_rgba_ts[index] = masked_image_rgba.to("cpu")

            # Image with Background RGB
            masked_image_with_bg = blend_rgba_with_background(masked_image_rgba.to(device), bg_color_tensor.to(device))
            res_images_rgb_ts[index] = masked_image_with_bg.to("cpu")

            # Debug Image
            debug_image = createDebugImageTensor( item,  bbox, data_by_frame[index]["phrase"], data_by_frame[index]["logits"])
            res_debug_images_ts[index] = debug_image.to("cpu")

            # For Dino_Collection-Pipe
            for i, p in enumerate(data_by_frame[index]["phrase"]):
                phrase_mask = create_mask_from_bboxes(img_height, img_width, [bbox[i]]).float()
                if mask_invert is not False:
                    phrase_mask = 1 - phrase_mask
                phrase_mask_np = np.expand_dims(phrase_mask.detach().cpu().numpy(), axis=-1)
                phrase_mask_np_blurred = np.expand_dims(blur_cvmask(phrase_mask_np, mask_blur), axis=-1)
                phrase_mask = torch.tensor(phrase_mask_np_blurred).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # MASK shape [B,C,H,W]
                phrase_mask = phrase_mask #.unsqueeze(0)

                current_img_rgba = torch.tensor(img_combine_mask_rgba(item, phrase_mask_np_blurred)).unsqueeze(0)
                phrase_data = {
                    "frame": index,
                    "phrase": p,
                    'bbox': bbox[i].unsqueeze(0),
                    "mask": phrase_mask,
                    "img_rgba": current_img_rgba,
                    "img_rgb": blend_rgba_with_background(current_img_rgba.to(device), bg_color_tensor.to(device)),
                    "img_ts": item.unsqueeze(0),
                    "logits": logits[i], # NOTE: Sometimes bugged? Oo
                    "latent": {"samples": empty_latent}
                }
                data.append(phrase_data)

        # -------------------------------------------- Generate Latents for Inpaint
        if vae is not None:
            # we do this just for nicer progressbars 🤣
            _ = vae.encode((image[0].unsqueeze(0) / 255.0).to(device)) 
            #
            for index, item in tqdm(enumerate(image), total=len(image), desc=f"{dnl13} generate Latent", unit="img", ncols=100, colour='#FFC0CB'):
                encoded_tensor = VAEencode_dnl13(vae=vae, image=item.unsqueeze(0), mask=res_masks_ts[index], latent_grow_mask=latent_grow_mask, device=device)
                res_latent_inpaint.append(encoded_tensor)

                # Finde alle Einträge in 'data', die zur aktuellen Frame-Nummer passen
                matching_entries = [d for d in data if d["frame"] == index]
                for entry in matching_entries:
                    entry["latent"] = VAEencode_dnl13(vae=vae, image=item.unsqueeze(0), mask=entry["mask"], latent_grow_mask=latent_grow_mask, device=device)
    	            
            # Sammeln der Tensoren
            samples_list = [d["samples"] for d in res_latent_inpaint]
            noise_mask_list = [d["noise_mask"] for d in res_latent_inpaint]

            # Verkettung der Tensoren entlang der Batch-Dimension
            concatenated_samples = torch.cat(samples_list, dim=0).to(device) # TODO: Abklären ob Latent auf CPU/DEFAULT/GPU vorhanden sein soll
            concatenated_noise_masks = torch.cat(noise_mask_list, dim=0).to(device)

            # Erstellen eines neuen Wörterbuchs mit den kombinierten Tensoren
            final_dict = {"samples": concatenated_samples,"noise_mask": concatenated_noise_masks}
            res_latent_inpaint = final_dict
        else:
            for index, item in enumerate(image):
                matching_entries = [d for d in data if d["frame"] == index]
                for entry in matching_entries:
                    entry["latent"] =  {"samples":empty_latent}
            res_latent_inpaint = {"samples": empty_latent_batched}

        # -------------------------------------------- Dino Collection       
        def dummy_img_rgb(frame_nummer):
            return image[frame_nummer].unsqueeze(0)  # Ersetze dies durch deine Logik zur Erzeugung von Dummy-img_rgb

        def dummy_img_ts(frame_nummer):
            return image[frame_nummer].unsqueeze(0) # Ersetze dies durch deine Logik zur Erzeugung von Dummy-img_ts

        dummy_daten = {
            'bbox': empty_box.to(device),
            'mask': empty_mask,
            'img_rgba':  empty_img_rgba,
            'img_rgb': dummy_img_rgb,
            'img_ts': dummy_img_ts,
            'logits': empty_logits,
            'latent': {"samples": empty_latent, "noise_mask": empty_mask}
        }
        # Sammeln aller einzigartigen Frame-Nummern
        alle_frames = sorted({item["frame"] for item in data})

        # Die Werte, die organisiert werden sollen
        schluessel = [
            'bbox', 
            'mask', 
            'img_rgba', 
            'img_rgb', 
            'img_ts', 
            #'logits', 
            'latent'
        ]

        # Initialisieren des organisierten Daten-Dictionaries
        organisierte_daten = {phrase: {key: [] for key in schluessel} for phrase in unique_phrases}
        
        # Durchlaufen aller tatsächlichen Frames und Phrasen
        for frame_nummer in alle_frames:
            for phrase in unique_phrases:
                # Überprüfen, ob Bilder für die Phrase vorhanden sind
                bilder_vorhanden = any(item["frame"] == frame_nummer and item["phrase"] == phrase for item in data)
                
                for key in schluessel:
                    if bilder_vorhanden:
                        wert = next(item[key] for item in data if item["frame"] == frame_nummer and item["phrase"] == phrase)
                    else:
                        dummy_wert = dummy_daten[key]
                        wert = dummy_wert(frame_nummer) if callable(dummy_wert) else dummy_wert
                    organisierte_daten[phrase][key].append((frame_nummer, wert))

        # Ausgabe der organisierten Daten
        for phrase, werte_dict in organisierte_daten.items():
            for key, werte_liste in werte_dict.items():
                tensor_liste = [wert for frame_nummer, wert in werte_liste if isinstance(wert, torch.Tensor)]
                if tensor_liste:
                    tensor_liste = [ten.to(device) for ten in tensor_liste]
                    zusammengeführte_tensoren = torch.cat(tensor_liste, dim=0).to("cpu")
                    werte_dict[key] = zusammengeführte_tensoren
                else:
                    if vae is not None:
                        samples_list = []
                        noise_mask_list = []
                        for frame_nummer, wert in werte_liste:
                            if not isinstance(wert, torch.Tensor):
                                s = wert["samples"].to(device)
                                n = wert["noise_mask"].to(device)
                                samples_list.append(s)
                                noise_mask_list.append(n)
                        verkettete_samples = torch.cat(samples_list, dim=0).to("cpu")
                        verkettete_noise_masks = torch.cat(noise_mask_list, dim=0).to("cpu")
                        werte_dict[key] = {"samples":verkettete_samples, 'noise_mask':verkettete_noise_masks}
                    else:
                        werte_dict[key] = {"samples": empty_latent_batched}


        res_bboxes_ts = torch.cat(res_bboxes, dim=0).to("cpu")
        del res_bboxes


        dino_collection_header = {
            "prompt":prompt_list, # Array
            "image": image, # passed input tensor
            "device": device, 
        }
        res_dino_collection = {
            "header":dino_collection_header,
            "data": organisierte_daten,
        }
        

        # Löschen des Modells
        del grounding_dino_model
        del vae
        # CUDA-Cache freigeben
        torch.cuda.empty_cache()
        

        #res_dino_collection=None
        return (res_images_rgba_ts, res_images_rgb_ts, res_masks_ts, res_debug_images_ts, res_latent_inpaint, res_bboxes_ts.tolist(), res_dino_collection, )
        return (res_images_rgba, res_images_rgb, res_masks, res_debug_images, res_latent_inpaint, res_bboxes.tolist(), res_dino_collection, )

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


