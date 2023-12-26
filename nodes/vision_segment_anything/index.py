


class SAMSegmentationProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ({}, ),
            },
        }
    CATEGORY = "dnl13/vision"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )

    def main(self, model_name, use_cpu=False):
        
        return (True, )