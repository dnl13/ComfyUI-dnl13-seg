import torch
import numpy as np
from PIL import Image
import cv2

from ..utils.collection import to_tensor 
import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap
from segment_anything_hq import SamPredictor as SamPredictorHQ
from segment_anything import SamPredictor
from segment_anything_hq.utils.amg import  remove_small_regions,build_point_grid, batched_mask_to_box,uncrop_points

from ..utils.image_processing import mask2cv, shrink_grow_mskcv, blur_mskcv, img_combine_mask_rgba , split_image_mask
from ..utils.collection import split_captions

def load_dino_image(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def get_grounding_output(
        model, 
        image, 
        caption, 
        box_threshold,       
        optimize_prompt_for_dino, 
        device, 
        #text_threshold, 
        with_logits=True
        ):
    
    # NOTE: not clear for what we can use text_threshold
    text_threshold = box_threshold 

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
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
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
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
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

def groundingdino_predict(
    dino_model,
    image,
    prompt,
    box_threshold,
    #text_threshold,
    optimize_prompt_for_dino,
    device
):
    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt, pred_phrases = get_grounding_output(
        dino_model, 
        dino_image, 
        prompt, 
        box_threshold, 
        #text_threshold, 
        optimize_prompt_for_dino, 
        device
        )

    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(device)
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt.to(device), pred_phrases


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
            masks, _ , _ = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input = pre_logits, multimask_output=False, hq_token_only=True)
        else:
            masks, _ , _ = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False, hq_token_only=True)
    else:
        # :NOTE - https://github.com/facebookresearch/segment-anything/issues/169 segement_anything got wrong floats instead of boolean mask_input=
        #if two_pass is True: 
        #    pre_masks, _ , _ = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False)
        #    masks, _ , _ = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False)
        #else:
        #    masks, _ , _ = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False)
        masks, _ , _ = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False)
        
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
        height, width = masks.shape[2], masks.shape[3]
        masks = masks.squeeze(0)

        # convert mask
        mask_np =  masks.permute( 1, 2, 0).cpu().numpy()# H.W.C

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
    