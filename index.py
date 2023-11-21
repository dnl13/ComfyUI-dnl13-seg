
"""
Use filenames prefixes like:
pr_ for proccessing or
util_ for ulitity Nodes
"""

from .nodes.node_modelloader_dinov1 import GroundingDinoModelLoaderV1
from .nodes.node_modelloader_sam import SAMModelLoader
from .nodes.node_sam_autoseg import SAMAutoSegment
from .nodes.node_dinosam_prompt import GroundingDinoSAMSegment
from .nodes.node_batch_selector import BatchSelector
from .nodes.node_combine_images_by_mask import CombineImagesByMask


NODE_CLASS_MAPPINGS = {
    "Automatic Segmentation (dnl13)": SAMAutoSegment,
    "Mask with prompt (dnl13)": GroundingDinoSAMSegment,
    "Dinov1 Model Loader (dnl13)": GroundingDinoModelLoaderV1,
    "SAM Model Loader (dnl13)": SAMModelLoader,
    'BatchSelector (dnl13)': BatchSelector,
    'Combine Images By Mask (dnl13)': CombineImagesByMask,
    # Nodes to come
    #"RGB (dnl13)": SAMAutoSegment,
}