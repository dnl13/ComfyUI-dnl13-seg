from ...utils.helper_img_utils import crop_tensor_by_bbox_batch


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

    RETURN_NAMES = ("image_rgba", "image_rgb", "cropped image","image mask", "bbox", "latent", "passes dino collection")
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "BBOX","LATENT", "DINO_DETECTION_COLLECTION")

    def main(self, dino_collection, prompt_selector):
        if prompt_selector < 1 or prompt_selector > len(dino_collection):
            raise ValueError(f"The index you selected for the prompt does not exist. "
                                f"Only {len(dino_collection)} prompts were identified. "
                                    f"Please choose an index within this range.")

        for key, phrase in dino_collection.items():
            phrase["cropped"] = crop_tensor_by_bbox_batch(
                phrase["img_ts"], phrase["bbox"])

        # Erstellen einer Liste der Schlüssel (Phrasen)
        schluessel_liste = list(dino_collection.keys())

        # Zugriff auf Daten über den Index
        phrase = schluessel_liste[prompt_selector-1]
        daten_fuer_phrase = dino_collection[phrase]

        image_rgba = daten_fuer_phrase['img_rgba']
        image_rgb = daten_fuer_phrase['img_rgb']
        cropped_image = daten_fuer_phrase["cropped"]
        image_mask = daten_fuer_phrase['mask']
        bbox = daten_fuer_phrase['bbox'].tolist()
        latent = daten_fuer_phrase['latent']
        dino_collection = dino_collection,

        output = [image_rgba, image_rgb, cropped_image,
                  image_mask, bbox, latent, dino_collection]
        return tuple(output)
