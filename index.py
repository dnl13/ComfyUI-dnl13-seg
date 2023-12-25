import os
import importlib

from .nodes.model_loader_sam import SAMModelLoader

# Konfiguration für die Zuordnung von Unterordnern zu Klassen
NODES_CONFIG = {
    "model_loader_dino": ['GroundingDinoModelLoaderV1'],
    "model_loader_sam": ['SAMModelLoader'],
    "universal_lazy_segmentation": ['Lazy_Mask_Segmentation'],
    "utils_batch_selector": ['BatchSelector'],
    "utils_greenscreen": ['GreenscreenGenerator'],
    "vision_clip_segementation": ['ClipSegmentationProcessor'],
    "vision_grounding_dino": ['DinoSegmentationProcessor'],
    "vision_segment_anything": ['SAM_Mask_Processor'],
}

# NODE_CLASS_MAPPINGS initialisieren
NODE_CLASS_MAPPINGS = {}

# Dynamischer Import basierend auf der Konfiguration
for node_folder, classes in NODES_CONFIG.items():
    for class_name in classes:
        try:
            # Vollständiger Modulpfad, relativ zum aktuellen Paket
            module_path = f'.nodes.{node_folder}'
            # Importieren des Moduls
            module = importlib.import_module(module_path, package=__package__)
            # Zugriff auf die Klasse und Zuordnung
            class_obj = getattr(module, class_name)
            NODE_CLASS_MAPPINGS[f"{class_name} (dnl13)"] = class_obj
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"Klasse {class_name} im Modul {module_path} nicht gefunden. Überspringe diese Node.")
            continue
