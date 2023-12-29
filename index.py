import os
import importlib

from .nodes.model_loader_sam import SAMModelLoader
from .nodes.universal_lazy_segmentation import LazyMaskSegmentation
from .nodes.vision_clip_segementation import ClipSegmentationProcessor
from .utils.helper_cmd_and_path import print_labels

dnl13 = print_labels("dnl13")

"""
TODO:
    working states:
    [x] GroundingDinoModelLoaderV1
    [x] SAMModelLoader
    [ ] LazyMaskSegmentation
    [ ] BatchSelector
    [ ] GreenscreenGenerator
    [x] ClipSegmentationProcessor
    [ ] DinoSegmentationProcessor
    [ ] SAM_Mask_Processor
"""
# Konfiguration für die Zuordnung von Unterordnern zu Klassen
NODES_CONFIG = {
    "model_loader_dino": ['GroundingDinoModelLoaderV1'],            # Done
    "model_loader_sam": ['SAMModelLoader'],                         # Done
    "universal_lazy_segmentation": ['LazyMaskSegmentation'],        # 
    "utils_dino_collection_selector": ['DinoCollectionPromptSelector'],
    "utils_batch_selector": ['BatchSelector'],                      # Done
    "utils_combine_masks": ['CombineMasks'],
    "utils_dropshadow": ['DropShadow'],
    "utils_greenscreen": ['GreenscreenGenerator'],

    "utils_resize_masks_directional": ['ResizeMaskDirectional'],
    "vision_clip_segementation": ['ClipSegmentationProcessor'],     # Done
    "vision_grounding_dino": ['DinoSegmentationProcessor'],         # Done?!?
    "vision_segment_anything": ['SAMSegmentationProcessor'],
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
            print(f"{dnl13} Klasse {class_name} im Modul {module_path} nicht gefunden. Überspringe diese Node.")
            continue
