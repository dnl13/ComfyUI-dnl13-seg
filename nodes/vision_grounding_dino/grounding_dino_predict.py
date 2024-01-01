import math
import torch
import numpy as np
from PIL import Image, ImageDraw

import torchvision.transforms.v2 as v2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from ...libs.groundingdino.util.inference import Model, predict

def get_unique_phrases(mapping_item, prompt_list):
    # Erstelle eine Liste mit allen Phrasen aus dem output_mapping
    unique_phrases = []
    all_phrases = set()
    for index, data in enumerate(mapping_item):
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
    return unique_phrases

def prompt_to_dino_prompt(prompt):
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
    return prompt, prompt_list



def groundingdino_predict(dino_model,image, prompt,box_threshold,upper_confidence_threshold, lower_confidence_threshold, device ):

    dino_model = dino_model.to(device)
    image = image.convert("RGB")

    # Manuelle Skalierung auf eine maximale Seitenlänge von 800 Pixel
    max_size = 800
    width, height = image.size
    aspect_ratio = width / height

    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)
        else:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)

        resized_image = image.resize((new_width, new_height))
    else:
        resized_image = image

    image_np_tmp = np.asarray(resized_image)
    image_np = np.copy(image_np_tmp)
    # Note: this is just a mess an led to missarable detection. 
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    image_transformed, _ = transform(image_np, None)
    image_transformed_new = image_transformed.clone().detach().requires_grad_(True)

    # image_transformed in ein NumPy-Array umwandeln
    image_transformed_np = image_transformed_new.cpu().numpy()
    image_transformed_np = (image_transformed_np * 255).astype(np.uint8).transpose(1, 2, 0)

    with torch.no_grad():
        boxes, logits, phrases = predict(dino_model, image_transformed.to(device), caption=prompt, text_threshold=box_threshold, box_threshold=box_threshold)

    boxes_np = boxes.numpy()
    logits_np = logits.numpy()

    # Filtere basierend auf einer Bedingung
    condition = np.logical_and(lower_confidence_threshold < logits_np, logits_np < upper_confidence_threshold)
    filtered_boxes_np = boxes_np[condition]
    filtered_logits_np = logits_np[condition]
    filtered_phrases = [phrase for i, phrase in enumerate(phrases) if condition[i]]

    tensor_xyxy_np = Model.post_process_result(source_h=height, source_w=width, boxes=boxes, logits=logits).xyxy
    filtered_tensor_xyxy_np = tensor_xyxy_np[condition]

    # Konvertiere NumPy-Arrays zurück zu Tensoren
    filtered_boxes = torch.tensor(filtered_boxes_np)
    filtered_logits = torch.tensor(filtered_logits_np)
    filtered_tensor_xyxy = torch.tensor(filtered_tensor_xyxy_np)


    return filtered_tensor_xyxy, filtered_phrases, filtered_logits

def VAEencode_dnl13(vae, image, mask, device, latent_grow_mask=16):
        x = (image.shape[1] // 8) * 8
        y = (image.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(image.shape[1], image.shape[2]), mode="bilinear")
        mask = mask.to(device)
        pixels = image.clone()
        pixels = pixels.to(device)
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        #grow mask by a few pixels to keep things seamless in latent space
        if latent_grow_mask == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, latent_grow_mask, latent_grow_mask), device=mask.device)
            padding = math.ceil((latent_grow_mask - 1) / 2)
            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        t = vae.encode(pixels.to(device))

        return {"samples":t, "noise_mask": (mask_erosion[:,:,:x,:y].round())}
