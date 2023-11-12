import torch
from PIL import Image
##
from .sam_hq.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from .utils.img_bm_tools import parse_binary_masks, images_greenscreen, crop_to_bbox, tensor_to_pil
##
from .utils.fallback_logging import logger



class dnl13_AutomaticSegmentation:
    version = '1.0.0'

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
              "image_input" : ("IMAGE", {}),
              "sam_model": ('SAM_MODEL', {}),
              "points_per_batch": ("INT", {"min": -1024, "max": 1024, "default": None, "step": 1}),
              "pred_iou_thresh": ("FLOAT", {"min": -16, "max": 16, "default": 0.88, "step": 0.01}),
              "stability_score_thresh": ("FLOAT", {"default": 0.95, "step": 0.01}),
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
                "background_color_rgb": ("RGB", {"default": (0,0,0)}), 
            },
            "hidden": {"dnl13NnodeVersion": dnl13_AutomaticSegmentation.version},
        }
    RETURN_TYPES = ("IMAGE", "MASK", "BBOX", "IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE","MASK","BBOX","crop_image", "crop_mask")
    FUNCTION = "main"
    OUTPUT_NODE = True

    CATEGORY = "dnl13"

    @staticmethod
    def main(image_input=None, background_color_rgb=(0,0,0) , sam_model=None, points_per_side=None, points_per_batch=None, pred_iou_thresh=0.88, stability_score_thresh=0.95, stability_score_offset=1.0, box_nms_thresh=0.7, crop_n_layers=0, crop_nms_thresh=0.7, crop_overlap_ratio=512 / 1500, crop_n_points_downscale_factor=1, point_grids=None, min_mask_region_area=0):
        #### default values 
        # points_per_side=32
        # points_per_batch=64

        # you have to choose between points_per_side
        if not points_per_side and not points_per_batch:
            points_per_side=32
            points_per_batch = 64
            logger.warning("Please provide values for either points_per_side or points_per_batch; both cannot be None.")
            logger.warning("setting defaults: points_per_side=32")
            logger.warning("setting defaults: points_per_batch=64")
        
        # get
        image_pil = tensor_to_pil(image_input)

 
    # TODO: Dropdown Button for model size selection tiny,sb,l,h etc. or model loader node

        # NOTE :maybe cpu switch

        
        # collet binary_masks with bboxes etc
        sam = SamAutomaticMaskGenerator(sam_model, points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh, stability_score_offset,
                                                 box_nms_thresh, crop_n_layers, crop_nms_thresh, crop_overlap_ratio, crop_n_points_downscale_factor, point_grids, min_mask_region_area)
        binary_masks = SamAutomaticMaskGenerator.generate(sam, image_pil, multimask_output=False)
        
        # prepare output
        images, masks, bboxs = parse_binary_masks(binary_masks)  # TODO 
        detected_images = images_greenscreen(images, masks , background_color_rgb) # TODO 
        detected_cropped_images, detected_cropped_masks = crop_to_bbox(images, masks, bboxs) # TODO 

        return detected_images, masks, bboxs, detected_cropped_images, detected_cropped_masks,
