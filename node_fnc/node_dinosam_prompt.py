import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2

from ..utils.collection import to_tensor 
#from ..libs.groundingdino.datasets.transforms import T
#import torchvision.transforms.v2 as T
import torchvision.transforms.v2 as v2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from ..libs.groundingdino.util.utils import get_phrases_from_posmap
from ..libs.groundingdino.util.inference import Model, predict
from ..libs.sam_hq.predictorHQ import SamPredictor as SamPredictorHQ
from ..libs.sam_hq.predictor import SamPredictor 
#from segment_anything import SamPredictor
from segment_anything.utils.amg import  remove_small_regions,build_point_grid, batched_mask_to_box,uncrop_points

from ..utils.image_processing import mask2cv, shrink_grow_mskcv, blur_mskcv, img_combine_mask_rgba , split_image_mask
from ..utils.collection import split_captions

def load_dino_image(image_pil):
    transform = v2.Compose([
        v2.RandomResize(800, max_size=1333),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

"""
def get_grounding_output(
        model, 
        image, 
        caption, 
        lower_threshold,
        upper_threshold,    
        optimize_prompt_for_dino, 
        device, 
        with_logits=True
        ):
    
    # NOTE: not clear for what we can use text_threshold
    text_threshold = lower_threshold 

    #check if we want prompt optmiziation = replace sd ","" with dino "." 
    if optimize_prompt_for_dino is not False:
        if caption.endswith(","):
          caption = caption[:-1]
        caption = caption.replace(",", ".")

    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)


    if "|" in caption:
        captions = split_captions(caption)
        all_boxes = []
        for caption in captions:
            with torch.no_grad():
                outputs = model(image[None], captions=[caption])
            
            logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
            boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
            logits.shape[0]

            # filter output
            logits_filt = logits.clone()
            boxes_filt = boxes.clone()
            filt_mask = (logits_filt.max(dim=1)[0] > lower_threshold) & (logits_filt.max(dim=1)[0] < upper_threshold)
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
            #logits_filt.shape[0]

            # get phrase
            tokenlizer = model.tokenizer
            tokenized = tokenlizer(caption)
            # build pred
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
                pred_phrase = get_phrases_from_posmap(text_threshold, tokenized, tokenlizer)
                if with_logits:
                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)

        # Concatenate all the boxes along the 0 dimension and return.
        #boxes_filt_concat = torch.cat(all_boxes, dim=0)
        #return boxes_filt_concat.to(device)
    else:

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = (logits_filt.max(dim=1)[0] > lower_threshold) & (logits_filt.max(dim=1)[0] < upper_threshold)
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)




    return boxes_filt.to(device), pred_phrases
"""




def resize_mask(
        self, ref_mask: np.ndarray, longest_side: int = 256
    ) -> tuple[np.ndarray, int, int]:
        """
        Resize an image to have its longest side equal to the specified value.

        Args:
            ref_mask (np.ndarray): The image to be resized.
            longest_side (int, optional): The length of the longest side after resizing. Default is 256.

        Returns:
            tuple[np.ndarray, int, int]: The resized image and its new height and width.
        """
        height, width = ref_mask.shape[:2]
        if height > width:
            new_height = longest_side
            new_width = int(width * (new_height / height))
        else:
            new_width = longest_side
            new_height = int(height * (new_width / width))

        return (
            cv2.resize(
                ref_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST
            ),
            new_height,
            new_width,
        )

def pad_mask(
        ref_mask: np.ndarray,
        new_height: int,
        new_width: int,
        pad_all_sides: bool = False,
    ) -> np.ndarray:
        """
        Add padding to an image to make it square.

        Args:
            ref_mask (np.ndarray): The image to be padded.
            new_height (int): The height of the image after resizing.
            new_width (int): The width of the image after resizing.
            pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

        Returns:
            np.ndarray: The padded image.
        """
        pad_height = 256 - new_height
        pad_width = 256 - new_width
        if pad_all_sides:
            padding = (
                (pad_height // 2, pad_height - pad_height // 2),
                (pad_width // 2, pad_width - pad_width // 2),
            )
        else:
            padding = ((0, pad_height), (0, pad_width))

        # Padding value defaults to '0' when the `np.pad`` mode is set to 'constant'.
        return np.pad(ref_mask, padding, mode="constant")

def reference_to_sam_mask( ref_mask: np.ndarray, threshold: int = 127, pad_all_sides: bool = False ) -> np.ndarray:
        """
        Convert a grayscale mask to a binary mask, resize it to have its longest side equal to 256, and add padding to make it square.

        Args:
            ref_mask (np.ndarray): The grayscale mask to be processed.
            threshold (int, optional): The threshold value for the binarization. Default is 127.
            pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

        Returns:
            np.ndarray: The processed binary mask.
        """

        # Convert a grayscale mask to a binary mask.
        # Values over the threshold are set to 1, values below are set to -1.
        ref_mask = np.clip((ref_mask > threshold) * 2 - 1, -1, 1)

        # Resize to have the longest side 256.
        resized_mask, new_height, new_width = resize_mask(ref_mask)

        # Add padding to make it square.
        square_mask = pad_mask(resized_mask, new_height, new_width, pad_all_sides)

        # Expand SAM mask's dimensions to 1xHxW (1x256x256).
        return np.expand_dims(square_mask, axis=0)


def denormalize_bbox(bbox, image_width, image_height):
    cx, cy, w, h = bbox

    # Entnormalisiere Width und Height
    w *= image_width
    h *= image_height

    # Entnormalisiere cx und cy
    cx = cx * image_width + 0.5  # 0.5, weil cxcywh-Format zentriert ist
    cy = cy * image_height + 0.5

    # Berechne xmin, ymin, xmax, ymax
    xmin = int(cx - w / 2)
    ymin = int(cy - h / 2)
    xmax = int(cx + w / 2)
    ymax = int(cy + h / 2)

    return xmin, ymin, xmax, ymax


def xyxy_to_cxcywh(xyxy):
    xmin, ymin, xmax, ymax = xyxy
    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + w / 2
    cy = ymin + h / 2
    return cx, cy, w, h


def resize_boxes(boxes_tensor, image_width, image_height, scale_factor=0.2):
    # Extrahiere die Werte aus dem Tensor
    cx, cy, w, h = boxes_tensor[0]

    # Berechne die Änderungen in Breite und Höhe um den Skalierungsfaktor
    delta_w = w * scale_factor / 2
    delta_h = h * scale_factor / 2

    # Neues Zentrum der Box nach der Skalierung
    new_cx = cx
    new_cy = cy

    # Berechne die neuen Werte für Breite und Höhe
    new_width = w + 2 * delta_w
    new_height = h + 2 * delta_h

    # Überprüfe, ob die skalierte Box die Grenzen des Originalbildes überschreitet
    max_width = image_width - cx
    max_height = image_height - cy

    if new_width > max_width:
        new_width = max_width

    if new_height > max_height:
        new_height = max_height

    # Erstelle die neuen Boxen mit den veränderten Werten
    new_boxes_tensor = torch.tensor([[new_cx, new_cy, new_width, new_height]])

    return new_boxes_tensor



def crop_image(image_transformed, box):
    xmin, ymin, xmax, ymax = box

    # Berechne die Breite und Höhe des Bildausschnitts
    width = xmax - xmin
    height = ymax - ymin

    # Schneide den Bildausschnitt aus dem transformierten Bild aus
    cropped_image = TF.crop(image_transformed, ymin, xmin, height, width)

    return cropped_image

# Passe die neuen Boxen auf die ursprünglichen Bildabmessungen an
def scale_boxes(new_boxes, original_box):
    # Extrahiere die Werte aus dem ursprünglichen Boxen-Tensor
    cx, cy, w, h = original_box[0]

    # Passe die neuen Boxen auf die ursprünglichen Dimensionen an
    scaled_boxes = new_boxes * torch.tensor([[w, h, w, h]])

    # Zentriere die Boxen an der ursprünglichen Position
    scaled_boxes[:, :2] += torch.tensor([[cx - w / 2, cy - h / 2]])

    return scaled_boxes

def groundingdino_predict(dino_model,image, prompt,box_threshold,upper_confidence_threshold, lower_confidence_threshold, device ):
    
    #dino_image = load_dino_image(image.convert("RGB")) 
    #dino_image = image.convert("RGB")
    dino_model = dino_model.to(device)
    #dino_image = dino_image.to(device)

    #check if we want prompt optmiziation = replace sd ","" with dino "." 
    """
    if optimize_prompt_for_dino is not False:
        if prompt.endswith(","):
          prompt = prompt[:-1]
        prompt = prompt.replace(",", ".")

    prompt = prompt.lower()
    prompt = prompt.strip()
    if not prompt.endswith("."):
        prompt = prompt + "."
    """


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
            #v2.RandomResize(min_size=800, max_size=1333),
            #v2.Resize(800),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            #
            #v2.RandomAdjustSharpness(sharpness_factor=2),
            #v2.RandomAutocontrast(),
            #
            #v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
            #v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image_np, None)
    image_transformed_new = image_transformed.clone().detach().requires_grad_(True)

    # image_transformed in ein NumPy-Array umwandeln
    image_transformed_np = image_transformed_new.cpu().numpy()
    image_transformed_np = (image_transformed_np * 255).astype(np.uint8).transpose(1, 2, 0)
    pil_image_transformed = Image.fromarray(image_transformed_np)
    #pil_image_transformed.show() #debug image which is going into the dino model

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

    # draw Bounding Boxes to image for user debug
    duplicate_image = image.copy()
    draw = ImageDraw.Draw(duplicate_image)
    text_padding = 2.5  # Anpassbares Padding
    for bbox, conf, phrase in zip(filtered_tensor_xyxy, filtered_logits, filtered_phrases):
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


    # if we need to debug lets see what we got
    #image.show()
    cropped_images = []
    for box in filtered_boxes:
        # Denormalisiere die Box, falls nötig
        box = denormalize_bbox(box, width, height)  # Stelle sicher, dass du die korrekten Bildabmessungen hast

        # Schneide das Bild anhand der Box aus
        cropped_img = crop_image(image, box)

        #cropped_img.show()
        # Füge das ausgeschnittene Bild der Liste hinzu
        cropped_images.append(cropped_img)
        


    

    # convert Image to Tensor
    transform = transforms.ToTensor()
    tensor_image = transform(duplicate_image)
    tensor_image_expanded = tensor_image.unsqueeze(0)
    tensor_image_formated = tensor_image_expanded.permute(0, 2, 3, 1)

    return filtered_tensor_xyxy, filtered_phrases, filtered_logits, tensor_image_formated





def sam_segment_new(
    sam_model,
    image,
    boxes,
    clean_mask_holes,
    clean_mask_islands,
    mask_blur,
    mask_grow_shrink_factor,
    two_pass,
    device
):  
    if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
        sam_is_hq = True
        predictor = SamPredictorHQ(sam_model)
    else:
        sam_is_hq = False
        predictor = SamPredictor(sam_model)
    if boxes.shape[0] == 0:
        return None
    
    #predictor = SamPredictorHQ(sam_model)
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)

    transformed_boxes = predictor.transform.apply_boxes_torch( boxes, image_np.shape[:2]).to(device)

    """
    predictor.predict_torch Returns:
    (np.ndarray): The output masks in CxHxW format, where C is the number of masks, and (H, W) is the original image size.
    (np.ndarray): An array of length C containing the model's predictions for the quality of each mask.
    (np.ndarray): An array of shape CxHxW, where C is the number of masks and H=W=256. These low resolution logits can be passed to a subsequent iteration as mask input.
    """

    if sam_is_hq is True:
        if two_pass is True: 
            _, _ , pre_logits = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input = None, multimask_output=True, return_logits=True, hq_token_only=False)
            pre_logits = torch.mean(pre_logits, dim=1, keepdim=True)
            masks, quality , logits = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input = pre_logits, multimask_output=False, hq_token_only=True)
        else:
            masks,quality , logits = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False, hq_token_only=True)
    else:
        # :NOTE - https://github.com/facebookresearch/segment-anything/issues/169 segement_anything got wrong floats instead of boolean mask_input=
        if two_pass is True: 
            tmpmasks, _ , pre_logits = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False)
            masks, quality , logits = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=pre_logits, multimask_output=False)

        else:
            masks, quality , logits = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False)


    #masks = masks.squeeze(0)
    # Finde den Index der Maske mit dem höchsten Qualitätswert
    #index_highest_quality = torch.argmax(quality)
    #mask_highest_quality = masks[index_highest_quality]
    #print("masks.shape",masks.shape)
    combined_mask = torch.sum(masks, dim=0)
    #print("combined_mask.shape",combined_mask.shape)
    mask_np =  combined_mask.permute( 1, 2, 0).cpu().numpy()# H.W.C
    # postproccess mask 
    mask_np, _ = remove_small_regions(mask=mask_np,area_thresh=clean_mask_holes,mode="holes" )
    mask_np, _ = remove_small_regions(mask=mask_np,area_thresh=clean_mask_islands,mode="islands" )
    
    msk_cv2 = mask2cv(mask_np)
    msk_cv2 = shrink_grow_mskcv(msk_cv2, mask_grow_shrink_factor)
    msk_cv2_blurred = blur_mskcv(msk_cv2, mask_blur)

    # fix if mask gets wrong dimensions
    if msk_cv2_blurred.ndim < 3 or msk_cv2_blurred.shape[-1] != 1:
        msk_cv2_blurred = np.expand_dims(msk_cv2_blurred, axis=-1)
    image_with_alpha = img_combine_mask_rgba(image_np_rgb , msk_cv2_blurred)
    _, msk = split_image_mask(image_with_alpha,device)

    image_with_alpha_tensor = to_tensor(image_with_alpha)
    image_with_alpha_tensor = image_with_alpha_tensor.permute(1, 2, 0)

    mask_ts = to_tensor(image_with_alpha)
    mask_ts = mask_ts.unsqueeze(0)
    mask_ts = mask_ts.permute(0, 2, 3, 1) 

    return msk, image_with_alpha_tensor

def sam_segment(
    sam_model,
    image,
    boxes,
    clean_mask_holes,
    clean_mask_islands,
    mask_blur,
    mask_grow_shrink_factor,
    multimask,
    two_pass,
    device
):  

    if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
        sam_is_hq = True
        predictor = SamPredictorHQ(sam_model)
    else:
        sam_is_hq = False
        predictor = SamPredictor(sam_model)
    if boxes.shape[0] == 0:
        return None
    
    #predictor = SamPredictorHQ(sam_model)
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)


    transformed_boxes = predictor.transform.apply_boxes_torch( boxes, image_np.shape[:2]).to(device)

    """
    predictor.predict_torch Returns:
    (np.ndarray): The output masks in CxHxW format, where C is the number of masks, and (H, W) is the original image size.
    (np.ndarray): An array of length C containing the model's predictions for the quality of each mask.
    (np.ndarray): An array of shape CxHxW, where C is the number of masks and H=W=256. These low resolution logits can be passed to a subsequent iteration as mask input.
    """

    if sam_is_hq is True:
        if two_pass is True: 
            _, _ , pre_logits = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input = None, multimask_output=True, return_logits=True, hq_token_only=False)
            pre_logits = torch.mean(pre_logits, dim=1, keepdim=True)
            masks, _ , _ = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input = pre_logits, multimask_output=False, hq_token_only=True)
        else:
            masks, _ , _ = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False, hq_token_only=True)
    else:
        # :NOTE - https://github.com/facebookresearch/segment-anything/issues/169 segement_anything got wrong floats instead of boolean mask_input=
        if two_pass is True: 
            tmpmasks, _ , logits = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False)
            masks, _ , _ = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=logits, multimask_output=False)

        else:
            masks, _ , logits = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False)

    """
    Removes small disconnected regions and holes in masks, then reruns
    box NMS to remove any new duplicates.

    Edits mask_data in place.

    Requires open-cv as a dependency.
    """

    if multimask is not False:
        output_images, output_masks = [], []
        for batch_index in range(masks.size(0)):
            # convert mask
            mask_np =  masks[batch_index].permute( 1, 2, 0).cpu().numpy()# H.W.C

            # postproccess mask 
            mask_np, _ = remove_small_regions(mask=mask_np,area_thresh=clean_mask_holes,mode="holes" )
            mask_np, _ = remove_small_regions(mask=mask_np,area_thresh=clean_mask_islands,mode="islands" )
            
            msk_cv2 = mask2cv(mask_np)
            msk_cv2 = shrink_grow_mskcv(msk_cv2, mask_grow_shrink_factor)
            msk_cv2_blurred = blur_mskcv(msk_cv2, mask_blur)

            # fix if mask gets wrong dimensions
            if msk_cv2_blurred.ndim < 3 or msk_cv2_blurred.shape[-1] != 1:
                msk_cv2_blurred = np.expand_dims(msk_cv2_blurred, axis=-1)

            # image proccessing
            #image_with_alpha = Image.fromarray(np.concatenate((image_np_rgb, msk_cv2_blurred), axis=2).astype(np.uint8), 'RGBA')
            image_with_alpha = img_combine_mask_rgba(image_np_rgb , msk_cv2_blurred)
            _, msk = split_image_mask(image_with_alpha,device)


            image_with_alpha_tensor = to_tensor(image_with_alpha).unsqueeze(0)
            image_with_alpha_tensor = image_with_alpha_tensor.permute(0, 2, 3, 1)

            output_images.append(image_with_alpha_tensor)
            output_masks.append(msk)
                        
        return (output_images, output_masks)
    else:
        
        output_images, output_masks = [], []
        
        #masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        masks = masks.sum(dim=0, keepdim=True)

        # workaround for better combined masks 
        masks_tmp = torch.squeeze(masks, dim=(0, 1))
        summed_mask_np = masks_tmp.cpu().numpy()
        pil_tmp_masks_image = Image.fromarray((summed_mask_np * 255).astype(np.uint8), mode='L')
        masks = torch.from_numpy(np.array(pil_tmp_masks_image) / 255.0).unsqueeze(0).unsqueeze(0)

        height, width = masks.shape[2], masks.shape[3]
        masks = masks.squeeze(0)

        # convert mask
        mask_np =  masks.permute( 1, 2, 0).cpu().numpy().astype(bool) # H.W.C

        # postproccess mask 
        mask_np, _ = remove_small_regions(mask=mask_np,area_thresh=clean_mask_holes,mode="holes" )
        mask_np, _ = remove_small_regions(mask=mask_np,area_thresh=clean_mask_islands,mode="islands" )
        
        msk_cv2 = mask2cv(mask_np)
        msk_cv2 = shrink_grow_mskcv(msk_cv2, mask_grow_shrink_factor)
        msk_cv2_blurred = blur_mskcv(msk_cv2, mask_blur)

        # fix if mask gets wrong dimensions
        if msk_cv2_blurred.ndim < 3 or msk_cv2_blurred.shape[-1] != 1:
            msk_cv2_blurred = np.expand_dims(msk_cv2_blurred, axis=-1)
  
        #image_with_alpha = Image.fromarray(np.concatenate((image_np_rgb, msk_cv2_blurred), axis=2).astype(np.uint8), 'RGBA')
        image_with_alpha = img_combine_mask_rgba(image_np_rgb , msk_cv2_blurred)
        _, msk = split_image_mask(image_with_alpha,device)

        rgb_ts = to_tensor(image_with_alpha)
        rgb_ts = rgb_ts.unsqueeze(0)
        rgb_ts = rgb_ts.permute(0, 2, 3, 1)    

        output_images.append(rgb_ts)
        output_masks.append(msk)

        return output_images, output_masks
    