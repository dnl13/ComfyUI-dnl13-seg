from ...utils.helper_device import list_available_devices, get_device


class SAMSegmentationProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        available_devices = list_available_devices()
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),

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

                "dedicated_device": (available_devices, ),

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
            "optional":{
                "bw_mask": ('IMAGE', {}),
                "bbox": ('BBOX', {}),
            }
        }
    CATEGORY = "dnl13/vision"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )

    def main(self, sam_model, use_cpu=False):
        # Zu ausführendes Gerät auswählen und setzen
        device = get_device(dedicated_device)
        
        return (True, )