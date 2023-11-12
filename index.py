
"""
Use filenames prefixes like:
pr_ for proccessing or
util_ for ulitity Nodes
"""

from .nodes.pr_automatic_segmentation import dnl13_AutomaticSegmentation
from .nodes.util_rgb import dnl13_RGB

NODE_CLASS_MAPPINGS = {
    "Automatic Segmentation (dnl13)": dnl13_AutomaticSegmentation,
    "RGB (dnl13)": dnl13_RGB,
}