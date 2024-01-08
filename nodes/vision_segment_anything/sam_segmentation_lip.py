import torch
import numpy as np
from PIL import Image

from segment_anything.utils.amg import  remove_small_regions,build_point_grid, batched_mask_to_box,uncrop_points

from ...libs.sam_hq.automatic import SamAutomaticMaskGenerator
from ...libs.sam_hq.predictorHQ import SamPredictor as SamPredictorHQ
from ...libs.sam_hq.predictor import SamPredictor 

from ...utils.helper_img_utils import enhance_edges, make_2d_mask, mask2cv, adjust_cvmask_size, blur_cvmask, split_image_mask, img_combine_mask_rgba
#

from torchvision.transforms.v2 import Compose, ToImage, ToDtype
to_tensor = Compose([ToImage(), ToDtype(torch.float32, scale=True)])



def gen_detection_hints_from_bbox_area(mask, threshold, bbox):
    """
    Generiert Erkennungshinweise innerhalb einer gegebenen Bounding Box basierend auf der Maske.

    :param mask: Ein 2D-Tensor, der die Maske des Bildes darstellt.
    :param threshold: Ein Schwellenwert, um zu entscheiden, ob ein Punkt als positiv oder negativ betrachtet wird.
    :param bbox: Ein Tensor oder Array mit den Koordinaten der Bounding Box [x_min, y_min, x_max, y_max].
    :return: Zwei Listen, eine für Punkte und eine für deren Labels.
    """
    logit_values, plabs, points = [], [], []
    mask = make_2d_mask(mask)

    x_min, y_min, x_max, y_max = map(int, bbox)
    box_width = x_max - x_min
    box_height = y_max - y_min
    y_step = max(3, int(box_height / 20))
    x_step = max(3, int(box_width / 20))

    for i in range(y_min, y_max, y_step):
        for j in range(x_min, x_max, x_step):
            mask_i = int((i - y_min) * mask.shape[0] / box_height)
            mask_j = int((j - x_min) * mask.shape[1] / box_width)
            logit_values.append(mask[mask_i, mask_j].cpu())

    logit_values = np.array(logit_values)
    mean = np.mean(logit_values)
    std = np.std(logit_values)
    standardized_values = (logit_values - mean) / std
    tanh_values = np.tanh(standardized_values)
    rounded_values = np.round(tanh_values, 5)
    rounded_values.tolist() 
    for i in range(y_min, y_max, y_step):
        for j in range(x_min, x_max, x_step):
            index = ((i - y_min) // y_step) * (box_width // x_step) + ((j - x_min) // x_step)
            if rounded_values[index] > threshold:
                points.append((j, i))
                plabs.append(1)
            else:
                points.append((j, i))
                plabs.append(0)
    return points, plabs


import numpy as np
import torch

def split_image_mask_np(image_array, device):
    """
    Trennt ein NumPy-Array in seine RGB-Komponenten und eine Alpha-Maske.

    Das NumPy-Array sollte im RGBA- oder RGB-Format vorliegen. Das RGB-Bild und die
    Alpha-Maske (falls vorhanden) werden normalisiert und in PyTorch-Tensoren konvertiert. 
    Falls keine Alpha-Maske vorhanden ist, wird eine leere Maske erstellt.

    Args:
        image_array (numpy.ndarray): Das NumPy-Array des Bildes im RGBA- oder RGB-Format.
        device (torch.device): Das Gerät, auf dem die Berechnung ausgeführt wird (z.B. 'cpu' oder 'cuda').

    Returns:
        tuple: Ein Tuple, das zwei Elemente enthält:
               - Ein Tensor, der das RGB-Bild repräsentiert.
               - Ein Tensor, der die Alpha-Maske repräsentiert (oder eine leere Maske, falls keine Alpha-Maske vorhanden ist).
    """

    # Überprüfe, ob das Array vier Kanäle (RGBA) hat; wenn ja, trenne RGB und Alpha
    if image_array.shape[-1] == 4:
        # RGB-Komponenten und Alpha-Kanal trennen
        image_rgb = image_array[..., :3]
        mask = image_array[..., 3]
    else:
        # Nur RGB, keine Alpha-Komponente
        image_rgb = image_array
        mask = np.zeros(image_array.shape[:2], dtype=np.float32)  # Erstelle eine leere Maske

    # Konvertiere NumPy-Arrays in Tensoren
    image_rgb = torch.from_numpy(image_rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

    return (image_rgb, mask)




def sam_auto_segmentationHQ(sam_model, image, points_per_side, min_mask_region_area, stability_score_thresh, box_nms_thresh,crop_n_layers,crop_nms_thresh, crop_overlap_ratio,device):

    """
    SamAutomaticMaskGenerator possible options:

    model (Sam): The SAM model to use for mask prediction.
    points_per_side (int or None): The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit point sampling.
    points_per_batch (int): Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.
    pred_iou_thresh (float): A filtering threshold in [0,1], using the model's predicted mask quality.
    stability_score_thresh (float): A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
    stability_score_offset (float): The amount to shift the cutoff when calculated the stability score.
    box_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
    crop_n_layers (int): If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
    crop_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.
    crop_overlap_ratio (float): Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.
    crop_n_points_downscale_factor (int): The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
    point_grids (list(np.ndarray) or None): A list over explicit grids  of points used for sampling, normalized to [0,1]. The nth grid in the  list is used in the nth crop layer. Exclusive with points_per_side.
    min_mask_region_area (int): If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.
    output_mode (str): The form masks are returned in. Can be 'binary_mask', 'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools. For large resolutions, 'binary_mask' may consume large amounts of memory.
    """

    image_rgb = image.convert("RGB") # to RGB in any case 4 dimension are not supported
    image_np = np.array(image_rgb) # image (np.ndarray): The image to generate masks for, in HWC uint8 format.

    sam_is_hq = hasattr(sam_model, 'model_name') and 'hq' in getattr(sam_model, 'model_name', '').lower()

    # use other proccess for HQ
    if sam_is_hq is True:

        sam = SamAutomaticMaskGenerator(
            sam_model,
            stability_score_thresh=stability_score_thresh, 
            points_per_side=points_per_side, 
            min_mask_region_area=min_mask_region_area,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers, 
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,

        ) # build detector

        binary_masks_list = SamAutomaticMaskGenerator.generate(sam, image_np, multimask_output = False) # generate binary_masks
    else:
        sam = SamAutomaticMaskGenerator(sam_model, points_per_side=points_per_side , crop_n_layers=1, min_mask_region_area=min_mask_region_area) # build detector
        binary_masks_list = SamAutomaticMaskGenerator.generate(sam, image_np) # generate binary_masks
    #
    response_images, response_masks = [],[] # prepare output
    #
    for binary_mask in binary_masks_list:

        # possible binary_masks_list responses: 
        """
        area = binary_mask['area']
        bbox = binary_mask['bbox']
        predicted_iou = binary_mask['predicted_iou']
        point_coords = binary_mask['point_coords']
        stability_score = binary_mask['stability_score']
        crop_box = binary_mask['crop_box']
        """
        #
        segmentation = binary_mask['segmentation']
        # Mask
        binary_mask_np = np.array(segmentation)
        binary_mask_uint8 = binary_mask_np.astype(np.uint8) * 255
        binary_mask_tensor = torch.from_numpy(binary_mask_uint8).float().unsqueeze(0).unsqueeze(0) # do tensor [B,C,H,W]
        response_masks.append(binary_mask_tensor)
        # Images
        image_rgb_copy = image_rgb.copy()
        #
        background = Image.new("RGBA", image_rgb_copy.size, (0, 0, 0, 255))
        binary_mask_pil = Image.fromarray(binary_mask_uint8) # to PIL 
        image_rgb_copy.putalpha(binary_mask_pil) # apply Mask
        background.paste(image_rgb_copy, (0, 0), binary_mask_pil)
        image_rgb_copy = background
        #
        rgb_tensor = to_tensor(image_rgb_copy).unsqueeze(0)  # add batch_size 1
        rgb_tensor = rgb_tensor.permute(0, 2, 3, 1) # [B,C,H,W] to [B,H,W,C]
        response_images.append(rgb_tensor)

    return response_images, response_masks





def sam_segment(
    sam_model,
    image,
    boxes,
    clean_mask_holes,
    clean_mask_islands,
    mask_blur,
    mask_grow_shrink_factor,
    two_pass,
    sam_contrasts_helper,
    sam_brightness_helper,
    sam_hint_threshold_helper,
    sam_helper_show,
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
    
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    image_np_rgb = image_np[..., :3]
    #
    sam_grid_points, sam_grid_labels = None, None
    #
    sam_input_image = enhance_edges(image_np_rgb, alpha=sam_contrasts_helper, beta=sam_brightness_helper, edge_alpha=1.0) # versuche um die Erkennung zu verbessern
    #
    #if sam_helper_show: 
    #    image_np_rgb = sam_input_image
    predictor.set_image(sam_input_image)

    transformed_boxes = predictor.transform.apply_boxes_torch( boxes, image_np.shape[:2]).to(device)

    """
    predictor.predict_torch Returns:
    (np.ndarray): The output masks in CxHxW format, where C is the number of masks, and (H, W) is the original image size.
    (np.ndarray): An array of length C containing the model's predictions for the quality of each mask.
    (np.ndarray): An array of shape CxHxW, where C is the number of masks and H=W=256. These low resolution logits can be passed to a subsequent iteration as mask input.
    """

    # :NOTE - https://github.com/facebookresearch/segment-anything/issues/169 segement_anything got wrong floats instead of boolean mask_input=
       
    if sam_is_hq is True:
        if two_pass is True: 
            # first pass
            pre_masks, _ , pre_logits = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input = None, multimask_output=True, return_logits=True, hq_token_only=False)
            
             # Zugriff auf die erste Box, falls mehrere Boxen vorhanden
            if sam_hint_threshold_helper != 0.0 :
                combined_pre_mask = torch.max(pre_masks, dim=1)[0]
                #detection_points, detection_labels = gen_detection_hints_from_mask_area( combined_pre_mask, sam_hint_threshold_helper, height, width)
                detection_points, detection_labels = gen_detection_hints_from_bbox_area(combined_pre_mask, sam_hint_threshold_helper, transformed_boxes[0])
                sam_grid_points, sam_grid_labels = detection_points, detection_labels
                # Konvertieren Sie Listen in Tensoren
                detection_points_tensor = torch.tensor(detection_points, dtype=torch.float32).to(device)
                detection_labels_tensor = torch.tensor(detection_labels, dtype=torch.float32).to(device)
                B = 1  # Bei einer einzelnen Bildvorhersage
                N = detection_points_tensor.shape[0]
                detection_points_tensor = detection_points_tensor.view(B, N, 2)
                detection_labels_tensor = detection_labels_tensor.view(B, N)
                
            else:
                detection_points_tensor = None
                detection_labels_tensor = None
            # second pass
            pre_logits = torch.mean(pre_logits, dim=1, keepdim=True)
            masks, quality , logits = predictor.predict_torch( point_coords=detection_points_tensor, point_labels=detection_labels_tensor, boxes=transformed_boxes, mask_input = pre_logits, multimask_output=False, hq_token_only=True)

        else:
            masks, quality , logits = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False, hq_token_only=True)
    else:
        if two_pass is True: 
            # first pass
            pre_masks, _ , pre_logits = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False)
            
            if sam_hint_threshold_helper  != 0.0 :
                combined_pre_mask = torch.max(pre_masks, dim=1)[0]
                #detection_points, detection_labels = gen_detection_hints_from_mask_area( combined_pre_mask, sam_hint_threshold_helper, height, width)
                detection_points, detection_labels = gen_detection_hints_from_bbox_area(combined_pre_mask, sam_hint_threshold_helper, transformed_boxes[0])
                sam_grid_points, sam_grid_labels = detection_points, detection_labels
                # Konvertieren Sie Listen in Tensoren
                detection_points_tensor = torch.tensor(detection_points, dtype=torch.float32).to(device)
                detection_labels_tensor = torch.tensor(detection_labels, dtype=torch.float32).to(device)
                B = 1  # Bei einer einzelnen Bildvorhersage
                N = detection_points_tensor.shape[0]
                detection_points_tensor = detection_points_tensor.view(B, N, 2)
                detection_labels_tensor = detection_labels_tensor.view(B, N)
            else:
                detection_points_tensor = None
                detection_labels_tensor = None
            # second pass
            masks, quality , logits = predictor.predict_torch( point_coords=detection_points_tensor, point_labels=detection_labels_tensor, boxes=transformed_boxes, mask_input=pre_logits, multimask_output=False)
        else:
            masks, quality , logits = predictor.predict_torch( point_coords=None, point_labels=None, boxes=transformed_boxes, mask_input=None, multimask_output=False)


    # Finde den Index der Maske mit dem höchsten Qualitätswert
    combined_mask = torch.sum(masks, dim=0)
    mask_np =  combined_mask.permute( 1, 2, 0).cpu().numpy()# H.W.C
    # postproccess mask 
    mask_np, _ = remove_small_regions(mask=mask_np,area_thresh=clean_mask_holes,mode="holes" )
    mask_np, _ = remove_small_regions(mask=mask_np,area_thresh=clean_mask_islands,mode="islands" )
    
    msk_cv2 = mask2cv(mask_np)
    msk_cv2 = adjust_cvmask_size(msk_cv2, mask_grow_shrink_factor)
    msk_cv2_blurred = blur_cvmask(msk_cv2, mask_blur)

    # fix if mask gets wrong dimensions
    if msk_cv2_blurred.ndim < 3 or msk_cv2_blurred.shape[-1] != 1:
        msk_cv2_blurred = np.expand_dims(msk_cv2_blurred, axis=-1)
    image_with_alpha = img_combine_mask_rgba(image_np_rgb , msk_cv2_blurred)
    _, msk = split_image_mask_np(image_with_alpha,device)

    image_with_alpha_tensor = to_tensor(image_with_alpha)
    image_with_alpha_tensor = image_with_alpha_tensor.permute(1, 2, 0)

    mask_ts = to_tensor(image_with_alpha)
    mask_ts = mask_ts.unsqueeze(0)
    mask_ts = mask_ts.permute(0, 2, 3, 1) 

    #sam_grid_points, sam_grid_labels = detection_points_tensor, detection_labels_tensor
    return msk, image_with_alpha_tensor, sam_grid_points, sam_grid_labels
