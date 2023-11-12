import torch
import numpy as np
##
from .sam_hq.segment_anything import SamAutomaticMaskGenerator
from .utils.img_bm_tools import parse_binary_masks, images_greenscreen, crop_to_bbox, tensor_to_pil
from .sam_hq.segment_anything.utils.amg import mask_to_rle_pytorch
##

from .utils.fallback_logging import logger
from PIL import Image
from io import BytesIO
##
import matplotlib.pyplot as plt
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


class dnl13_AutomaticSegmentation:
    version = '1.0.0'

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_input": ("IMAGE", {}),
                "sam_model": ('SAM_MODEL', {}),
                "points_per_batch": ("INT", {"min": -1024, "max": 1024, "default": None, "step": 1}),
                "pred_iou_thresh": ("FLOAT", {"min": 0.01, "max": 1, "default": 0.88, "step": 0.01}),# mask qualtiy
                "stability_score_thresh": ("FLOAT", {"min": 0.01, "max": 1, "default": 0.95, "step": 0.01}),# detection threshold
                "stability_score_offset": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "box_nms_thresh": ("FLOAT", {"default": 0.7, "step": 0.1}),
                "crop_n_layers": ("INT", {"default": 0, "step": 0.1}),
                "crop_nms_thresh": ("FLOAT", {"default": 0.7, "step": 0.1}),
                "crop_overlap_ratio": ("FLOAT", {"default": 512 / 1500}),
                "crop_n_points_downscale_factor": ("INT", {"default": 1, "step": 1}),
                "min_mask_region_area": ("INT", {"default": 0, "step": 1}),
            },
            "optional": {
                "points_per_side": ("INT", {"min": 1, "max": 7680, "default": None, "step": 1}),
                # Optional[List[np.ndarray]]
                "point_grids": ("LIST_NP", {"default": None}),
                "background_color_rgb": ("RGB", {"default": (0, 0, 0)}),
            },
            "hidden": {"dnl13NnodeVersion": dnl13_AutomaticSegmentation.version},
        }
    RETURN_TYPES = ("IMAGE", "MASK", "BBOX", "IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK", "BBOX", "crop_image", "crop_mask")
    FUNCTION = "main"
    OUTPUT_NODE = True

    CATEGORY = "dnl13"

    @staticmethod
    def main(image_input=None, background_color_rgb=(0, 0, 0), sam_model=None, points_per_side=None, points_per_batch=None, pred_iou_thresh=0.88, stability_score_thresh=0.95, stability_score_offset=1.0, box_nms_thresh=0.7, crop_n_layers=0, crop_nms_thresh=0.7, crop_overlap_ratio=512 / 1500, crop_n_points_downscale_factor=1, point_grids=None, min_mask_region_area=0):
        # default values
        # points_per_side=32
        # points_per_batch=64

        # NOTE: Dropdown Button for model size selection tiny,sb,l,h etc. or model loader node

        # NOTE :maybe cpu switch

        # you have to choose between points_per_side
        if not points_per_side and not points_per_batch:
            points_per_side = 32
            points_per_batch = 64
            logger.warning("Please provide values for either points_per_side or points_per_batch; both cannot be None.")
            logger.warning("setting defaults: points_per_side=32")
            logger.warning("setting defaults: points_per_batch=64")

        # Sam image format: image (np.ndarray): The image to generate masks for, in HWC uint8 format.
        predictor_image_input_np = image_input.squeeze(0).numpy().astype(np.uint8)
        image_input_np_rgb_clipped = np.clip(predictor_image_input_np, 0, 255)



        #sam = SamAutomaticMaskGenerator(sam_model, points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh, stability_score_offset,
        #                                box_nms_thresh, crop_n_layers, crop_nms_thresh, crop_overlap_ratio, crop_n_points_downscale_factor, point_grids, min_mask_region_area)
        
        sam = SamAutomaticMaskGenerator(sam_model, points_per_side, pred_iou_thresh, stability_score_thresh, crop_n_layers, crop_n_points_downscale_factor, min_mask_region_area)



        # generate masks
        binary_masks = SamAutomaticMaskGenerator.generate( sam, image_input_np_rgb_clipped, False)
        

        print("image_input.shape1", image_input.shape)
        test = image_input.squeeze(0) if image_input.dim() == 4 else image_input
        test = tensor_to_pil(test)

        plt.figure(figsize=(20,20))
        plt.imshow(test)
        show_anns(binary_masks)
        plt.axis('on')
        plt.show() 

        # Matplotlib-Plot in ein PIL-Bild umwandeln
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_pil = Image.open(buf)
        img_pil.show()


        print("plt",plt)
        print("plt.type",type(plt))
        
        #mask_list, bboxs_list, crop_box_list  = parse_binary_masks(binary_masks) 

        #mask_rle = mask_to_rle_pytorch(mask_list)
        #print("mask_rle",mask_rle)

        binary_masks = SamAutomaticMaskGenerator.postprocess_small_regions(mask_rle,min_mask_region_area,crop_nms_thresh)




        detected_images = images_greenscreen(images, masks, background_color_rgb)  # TODO
        detected_cropped_images, detected_cropped_masks = crop_to_bbox( images, masks, bboxs)  # TODO

        return detected_images, masks, bboxs, detected_cropped_images, detected_cropped_masks,
