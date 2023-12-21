import torch
import numpy as np
from PIL import Image
from ..node_fnc.node_dinosam_prompt import sam_segment, groundingdino_predict, sam_segment_new
from ..utils.collection import get_local_filepath, check_mps_device
import re
from collections import defaultdict


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
        grounding_dino_model.to(device)
        grounding_dino_model.eval()
        sam_model.to(device)
        sam_model.eval()
        #
        img_batch, img_height, img_width, img_channel = image.shape       
        empty_mask = torch.zeros((1, 1, img_height, img_width), dtype=torch.float32) # [B,C,H,W]
        empty_mask = empty_mask / 255.0
        #
        res_images , res_masks , res_boxes, res_debug_images, unique_phrases = [],[],[],[],[]
        #
        height, width = image.size(1), image.size(2)
        #
        transparent_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        transparent_image_np = np.array(transparent_image)
        transparent_image_tensor = torch.from_numpy(transparent_image_np).permute(2, 0, 1).unsqueeze(0)
        transparent_image_tensor = transparent_image_tensor.permute(0, 2, 3, 1)
        transparent_maske_tensor = torch.zeros(1, height, width)
        empty_box = torch.zeros((1, 4)) 
        empty_logits = torch.zeros((1, 1)) 
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
        print("got prompt like this:", prompt_list)
        #
        #
        #      
        output_mapping = {}
        for index, item in enumerate(image):
            item = Image.fromarray(np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes, phrases, logits, debug_image = groundingdino_predict(grounding_dino_model, item, prompt, box_threshold, upper_confidence_threshold, lower_confidence_threshold, device)
            res_debug_images.extend(debug_image)

            phrase_boxes = {}
            for p in phrases:
                phrase_boxes[p] = {'box': [], 'mask': [], 'image': [], 'logits':[]}

            for i, phrase in enumerate(phrases):
                sam_masks, masked_image = sam_segment_new(sam_model, item, boxes[i], clean_mask_holes, clean_mask_islands, mask_blur, mask_grow_shrink_factor, two_pass, device)
                masked_image = masked_image.unsqueeze(0)
                phrase_boxes[phrase]['box'].append(boxes[i])
                phrase_boxes[phrase]['mask'].append(sam_masks)
                phrase_boxes[phrase]['image'].append(masked_image)
                phrase_boxes[phrase]['logits'].append(logits[i])

            output_mapping[index] = phrase_boxes

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
        print("maximum masks per frame depending on prompt detection:", max_found_masks)

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

        print("found following unique_phrases in prompt:", unique_phrases)

        for index in sorted(output_mapping.keys()):
            if multimask:
                # Füge alle Masken und Bilder separat hinzu, wenn multimask aktiviert ist
                for phrase in unique_phrases:
                    if phrase in output_mapping[index]:
                        res_images.extend(output_mapping[index][phrase]['image'])
                        res_masks.extend(output_mapping[index][phrase]['mask'])
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
                    res_images.append(combined_image)
                if combined_mask is not None:
                    res_masks.append(combined_mask)
            
        res_images = torch.cat(res_images, dim=0)
        res_masks = torch.cat(res_masks, dim=0)




      
        return (res_images, res_masks, res_debug_images, )
    
        """
        TODO: 
        - sortiere phrase wie im prompt. beachte aber das es zu einfachen ähnlichkeiten kann "left feet" = 'feet"
        - checke ob phrase der gleichen anzahl an prompts entsprechen. sonst mit leeren masken und bildern auffüllen

        - stelle fest wo wieviel masken und boxen drin sind um ggfl auch aufzufüllen
        - 
        """
        """
        # Vorübergehende Speicherung der Anzahl der erkannten Phrasen pro Bild
        phrase_count = {phrase: phrases.count(phrase) for phrase in set(phrases)}

        output_mapping[index] = {pl: {'masks': [], 'images': []} for pl in prompt_list}


        # Iteration durch die Phrasen in der finalen Liste
        for pl in prompt_list:
            # Sicherstellen, dass die richtige Anzahl von Masken und Bildern gespeichert wird
            count = phrase_count.get(pl, 0)

            for _ in range(count):
                box = phrase_boxes.get(pl)  # Die Box für die aktuelle Phrase
                for b in box: 
                    sam_masks, masked_image = sam_segment_new(sam_model, item, b, clean_mask_holes, clean_mask_islands, mask_blur, mask_grow_shrink_factor, two_pass, device)
                    masked_image = masked_image.unsqueeze(0)
                    output_mapping[index][pl]['masks'].extend(sam_masks)
                    output_mapping[index][pl]['images'].extend(masked_image)
        """

        """
        for index, data in output_mapping.items():
            print("Frame", index)
            for phrase, values in data.items():
                print("phrase",phrase)
                print("values['masks']",len(values['masks']))

        find_max_len = []
        for index, data in output_mapping.items():
            for key, value in data.items():
                find_max_len.append(len(value))
        max_len = max(find_max_len)
       

        #max_masks_per_phrase = get_max_masks_per_phrase(output_mapping)
        #print(max(get_max_masks_per_phrase))
    
        output_mapping = process_masks(output_mapping, max_len, transparent_image_tensor, transparent_maske_tensor)
         """



        """
        for i in sorted(output_mapping.keys()):  # Sortieren nach Index
            for phrase in prompt_list:  # Reihenfolge gemäß prompt_list beachten
                if phrase in output_mapping[i]:
                    values = output_mapping[i][phrase]
                    res_masks.extend(values['masks'])
                    res_images.extend(values['images'])



        phrases_set = set()
        for index, data in output_mapping.items():
            for phrase in data.keys():
                phrases_set.add(phrase)                
        """

        """
        sam_masks, masked_image = sam_segment_new(sam_model, item, boxes, clean_mask_holes, clean_mask_islands, mask_blur, mask_grow_shrink_factor, two_pass, device)
        masked_image = masked_image.unsqueeze(0)
        """

        """
        # Formated Print
        if isinstance(logits, torch.Tensor):
            logits = logits.tolist()
        formatted_phrases = [f"{phrase} ({logit:.4f})" for phrase, logit in zip(phrases, logits)]
        formatted_output = ', '.join(formatted_phrases)
        #print(f"\033[1;32m(dnl13-seg)\033[0m > phrase(confidence): {formatted_output}")
        """


        #print("------------ res_images ----------- ")
        #for it in res_images:
        #    print(it.shape)

 
