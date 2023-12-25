

class LazyMaskSegmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {"default": "arms, legs, eyes, hair, head", "multiline": True}),
                "bbox_threshold": ("FLOAT", {
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

                "multimask": ('BOOLEAN', {"default": False}),

                "two_pass": ('BOOLEAN', {"default": False}),

                "sam_contrasts_helper": ("FLOAT", {
                    "default": 1.25,
                    "min": 0,
                    "max": 3.0,
                    "step": 0.01
                }),
                "sam_brightness_helper": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "sam_hint_threshold_helper": ("FLOAT", {
                    "default": 0.00,
                    "min": -1.00,
                    "max": 1.00,
                    "step": 0.001
                }),
                "sam_helper_show": ('BOOLEAN', {"default": False}),

                "dedicated_device": (["Auto", "CPU", "GPU"], ),

                "clean_mask_holes": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 100000,
                    "step": 1
                }),

                "clean_mask_islands": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 100000,
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
                    "max": 1024,
                    "step": 1
                }),

                "background_color": ("STRING", {"default": "#00FF00", "multiline": False}),
            },
        }
    CATEGORY = "dnl13"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("RGBA_Images", "RGB_Images (with Background Color)",
                    "Masks", "DINO Image Detections Debug (rgb)")

    def main(
        self,
        sam_model=None,
        grounding_dino_model=None,
        image=None,
        prompt="arms, legs, eyes, hair, head",
        bbox_threshold=0.35,
        lower_confidence_threshold=0.01,
        upper_confidence_threshold=1,
        multimask=False,
        two_pass=False,
        sam_contrasts_helper=1.25,
        sam_brightness_helper=0,
        sam_hint_threshold_helper=0.00,
        sam_helper_show=False,
        dedicated_device="Auto",
        clean_mask_holes=64,
        clean_mask_islands=64,
        mask_blur=0,
        mask_grow_shrink_factor=0,
        background_color="#00FF00"
    ):
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

        res_images_rgba, res_images_rgb, res_masks, res_debug_images = None, None, None, None
        return (res_images_rgba, res_images_rgb, res_masks, res_debug_images, )
