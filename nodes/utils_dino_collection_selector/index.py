
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
        #print(len(dino_collection))
        #print(type(dino_collection))
        #print(dino_collection)
        #print("prompt_selector", prompt_selector)

        # Erstellen einer Liste der Schlüssel (Phrasen)
        schluessel_liste = list(dino_collection.keys())

        # Zugriff auf Daten über den Index
        phrase = schluessel_liste[prompt_selector-1]
        daten_fuer_phrase = dino_collection[phrase]

        image_rgba = daten_fuer_phrase['img_rgba']
        image_rgb = daten_fuer_phrase['img_rgb'] 
        cropped_image = None
        image_mask = daten_fuer_phrase['mask']
        bbox = daten_fuer_phrase['bbox']
        latent = daten_fuer_phrase['latent'] 
        dino_collection = dino_collection


        return (image_rgba, image_rgb, cropped_image, image_mask, bbox, latent, dino_collection, )