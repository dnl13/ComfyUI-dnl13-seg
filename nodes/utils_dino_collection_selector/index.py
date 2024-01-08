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

    RETURN_NAMES = ("image_rgba", "image_rgb", "image mask",
                    "bbox", "latent", "dino collection filtered")
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "BBOX",
                    "LATENT", "DINO_DETECTION_COLLECTION")

    def main(self, dino_collection, prompt_selector):
        prompt_selector = prompt_selector - 1 # make it machine conform
        if prompt_selector < 0 or prompt_selector > len(dino_collection):
            raise ValueError(f"The index you selected for the prompt does not exist. "
                             f"Only {len(dino_collection)} prompts were identified. "
                             f"Please choose an index within this range.")
        
        # Erstellen einer Liste der Schlüssel (Phrasen)
        schluessel_liste = list(dino_collection['data'].keys())
        # Zugriff auf Daten über den Index
        phrase = schluessel_liste[prompt_selector]
        daten_fuer_phrase = dino_collection['data'][phrase]

        # Anpassen des headers, um nur die ausgewählte Phrase zu behalten
        angepasster_header = {
            'prompt': [phrase],  # Aktualisieren der Prompt-Liste auf nur die ausgewählte Phrase
            'image': dino_collection['header']['image'],  # Beibehalten der Image-Informationen
            'device': dino_collection['header']['device']  # Beibehalten des Device, falls vorhanden
        }

        # Filtern der dino_collection, um nur die Daten für die ausgewählte Phrase zu behalten
        gefilterte_dino_collection = {
            'header': angepasster_header,  # Beibehalten der Header-Informationen
            'data': {phrase: daten_fuer_phrase}  # Nur Daten für die ausgewählte Phrase
        }

        image_rgba = daten_fuer_phrase['img_rgba']
        image_rgb = daten_fuer_phrase['img_rgb']
        image_mask = daten_fuer_phrase['mask']
        bbox = daten_fuer_phrase['bbox'].tolist()
        latent = daten_fuer_phrase['latent']
        dino_collection = gefilterte_dino_collection,

        output = [image_rgba, image_rgb, image_mask, bbox, latent, dino_collection]
        return tuple(output)
