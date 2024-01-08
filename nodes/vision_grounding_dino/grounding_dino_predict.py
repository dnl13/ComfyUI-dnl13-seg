import math
import torch
import numpy as np
from PIL import Image, ImageDraw

import torchvision.transforms.v2 as v2
import torchvision.transforms as transforms
import torch.nn.functional as F
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


def dino_image_preperation(image):
    max_size = 800
    height, width, channels = image.shape
    if channels == 4:
        image = image[:, :, :3]
    aspect_ratio = width / height

    new_height = height
    new_width =  width

    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)
        else:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)

    image = image.permute(2, 0, 1)

    transform = v2.Compose(
        [
            v2.Resize((new_height, new_width)),
            #v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            #v2.ToDtype(torch.uint8, scale=True),
        ]
    )
    image_transformed, _ = transform(image, None)
    image_transformed_new = image_transformed #image_transformed.clone().detach().requires_grad_(True)
    #image_transformed_new = image_transformed_new.permute(1,2, 0)
    #print("!",image_transformed_new)
    #print("!",type(image_transformed_new))
    #print("!",image_transformed_new.shape)
    # image_transformed in ein NumPy-Array umwandeln
    #image_transformed_np = image_transformed_new.cpu().numpy()
    #image_transformed_np = image_transformed_np * 255
    #image_transformed_np = image_transformed_np.astype(np.uint8)
    #image_transformed_np = image_transformed_np.transpose(1, 2, 0)

    return image_transformed_new

def groundingdino_predict(dino_model, image, prompt, box_threshold, upper_confidence_threshold, lower_confidence_threshold, device):
    dino_model = dino_model.to(device)
    height, width, _ = image.shape
    image = dino_image_preperation(image)
    with torch.no_grad():
        boxes, logits, phrases = predict(
                                    dino_model, 
                                    image.to(device), 
                                    caption=prompt, 
                                    text_threshold=box_threshold, 
                                    box_threshold=box_threshold
                                )
    del image
    boxes_np = boxes.numpy()
    logits_np = logits.numpy()

    # Filtere basierend auf einer Bedingung
    #condition = np.logical_and(lower_confidence_threshold < logits_np, logits_np < upper_confidence_threshold)
    #filtered_boxes_np = boxes_np[condition]
    #filtered_logits_np = logits_np[condition]
    #filtered_phrases = [phrase for i, phrase in enumerate(phrases) if condition[i]]


    # Logische Operation direkt auf den Tensoren
    condition = (lower_confidence_threshold < logits) & (logits < upper_confidence_threshold)

    # Filtere basierend auf der Bedingung
    filtered_boxes = boxes[condition]
    filtered_logits = logits[condition]
    filtered_phrases = [phrase for i, phrase in enumerate(phrases) if condition[i]]

    tensor_xyxy_np = Model.post_process_result(
                                                source_h=height, 
                                                source_w=width, 
                                                boxes=boxes, 
                                                logits=logits
                                            ).xyxy
    #print(type(tensor_xyxy_np[condition]))# numpy.ndarray
    tensor_xyxy = torch.from_numpy(tensor_xyxy_np)
    
    filtered_tensor_xyxy = tensor_xyxy[condition] 

    # Konvertiere NumPy-Arrays zurück zu Tensoren
    #filtered_boxes = torch.tensor(filtered_boxes_np)
    #filtered_logits = torch.tensor(filtered_logits_np)
    #filtered_tensor_xyxy = torch.tensor(filtered_tensor_xyxy_np)

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
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

    # grow mask by a few pixels to keep things seamless in latent space
    if latent_grow_mask == 0:
        mask_erosion = mask
    else:
        kernel_tensor = torch.ones((1, 1, latent_grow_mask, latent_grow_mask), device=mask.device)
        padding = math.ceil((latent_grow_mask - 1) / 2)
        mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

    m = (1.0 - mask.round()).squeeze(1)
    for i in range(3):
        pixels[:, :, :, i] -= 0.5
        pixels[:, :, :, i] *= m
        pixels[:, :, :, i] += 0.5
    t = vae.encode(pixels.to(device))

    return {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}
