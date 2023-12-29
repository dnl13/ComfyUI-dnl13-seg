"""
@author: dnl13
@title: dnl13 Computer Vision Suite
@nickname: dnl13-cvs
@description: This extension offers various nodes for computer Vision and image segmentation and object detection, and more.
"""

import os
import sys

#sys.path.append(os.path.join(os.path.dirname(__file__), "libs"))
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Nodes
from .index import NODE_CLASS_MAPPINGS

#WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS']
