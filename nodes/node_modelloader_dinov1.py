from ..node_fnc.node_modelloader_dino import list_groundingdino_model, load_groundingdino_model

class GroundingDinoModelLoaderV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_groundingdino_model(), ),
            },
        }
    CATEGORY = "dnl13/model_loaders"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )

    def main(self, model_name):
        dino_model = load_groundingdino_model(model_name)
        return (dino_model, )
