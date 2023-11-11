import segment_anything 

print(SamPredictor);

class dnl13_TEST:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "whatEver": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                },
                "hidden": {"dnl13NnodeVersion": dnl13_TEST.version},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "main"
    OUTPUT_NODE = True

    CATEGORY = "utils"

    @staticmethod
    def main(whatEver):
        return whatEver,