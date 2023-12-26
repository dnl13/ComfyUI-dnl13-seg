import torch
import numpy as np
from PIL import Image, ImageDraw

import torchvision.transforms.v2 as v2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from ...libs.groundingdino.util.inference import Model, predict


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

