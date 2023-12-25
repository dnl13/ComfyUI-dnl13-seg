import os
import sys

#sys.path.append(os.path.join(os.path.dirname(__file__), "libs"))
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Nodes
from .index import NODE_CLASS_MAPPINGS

# for later use
# WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS']
