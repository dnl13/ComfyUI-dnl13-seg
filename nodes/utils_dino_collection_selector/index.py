
class DinoCollectionPromptSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dino_collection": ('DINO_DETECTION_COLLECTION', {}),
                "prompt_selector": ("INT", {
                    "default": 1,
                    "min": 1,
                    "step": 1
                }),
            }
        }
    CATEGORY = "dnl13/utilitys"
    FUNCTION = "main"
    
    RETURN_NAMES = ( "image_rgba","image_rgb", "cropped image", "image mask", "bbox", "latent", "passes dino collection")
    RETURN_TYPES = ( "IMAGE", "IMAGE", "IMAGE", "MASK","BBOX", "LATENT", "DINO_DETECTION_COLLECTION")


    def main(self, dino_collection, prompt_selector):
        print("-------------------- DinoCollectionSelector --------------------")
        print(len(dino_collection))
        print("prompt_selector", prompt_selector)

        image_rgba = None 
        image_rgb = None 
        cropped_image = None
        bbox = None
        latent = None 
        dino_collection = None


        return (image_rgba, image_rgb, cropped_image,bbox,latent, dino_collection, )