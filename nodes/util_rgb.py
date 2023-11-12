##
class dnl13_RGB:
    version = '1.0.0'

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
              "red": ("INT", {"min": 0, "max": 255, "default": 0, "step": 1}),
              "green": ("INT", {"min": 0, "max": 255, "default": 0, "step": 1}),
              "blue": ("INT", {"min": 0, "max": 255, "default": 0, "step": 1}),
            },
            "hidden": {"dnl13NnodeVersion": dnl13_RGB.version},
        }
    RETURN_TYPES = ("RGB", "TUPLE")
    RETURN_NAMES = ("RGB","TUPLE",)
    FUNCTION = "main"
    OUTPUT_NODE = True

    CATEGORY = "dnl13"

    @staticmethod
    def main(red, green, blue):
      return (red, green, blue)
       