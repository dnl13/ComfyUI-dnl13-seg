
# Konfiguration für die Zuordnung von Unterordnern zu Klassen
import os
import importlib
NODES_CONFIG = {

    "model_loader_dino": ['GroundingDinoModelLoaderV1'],
    "model_loader_sam": ['SAMModelLoader'],
    "universal_lazy_segmentation": ['Lazy_Mask_Segmentation'],
    "utils_batch_selector": ['BatchSelector'],
    "utils_greenscreen": ['Greenscreen Generator'],
    "vision_clip_segementation": ['ClipSegmentationProcessor'],
    "vision_grounding_dino": ['DinoSegmentationProcessor'],
    "vision_segment_anything": ['SAM_Mask_Processor'],
}

# Basispfad für den 'nodes'-Ordner
nodes_path = os.path.join(os.path.dirname(__file__), 'nodes')

# NODE_CLASS_MAPPINGS initialisieren
NODE_CLASS_MAPPINGS = {}

# Dynamischer Import basierend auf der Konfiguration
for node_folder, classes in NODES_CONFIG.items():
    module_path = f'.nodes.{node_folder}'
    module = importlib.import_module(module_path, package=__name__)
    for class_name in classes:
        class_obj = getattr(module, class_name)
        NODE_CLASS_MAPPINGS[f"{class_name} (dnl13)"] = class_obj
