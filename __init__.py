# In the __init__.py file of your main directory
#import sys
#import os

# Get the current path to the directory of the __init__.py file
#current_path = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (project directory) to sys.path
#project_path = os.path.dirname(current_path)
#sys.path.append(project_path)

# Import Nodes
from .index import NODE_CLASS_MAPPINGS

# for later use
# WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS']



# - WAS Dictionary
MANIFEST = {
    "name": "DNL13 Nodes",  # The title that will be displayed on Node Class menu,. and Node Class view
    "version": (0, 1, 0),  # Version of the custom_node or sub module
    "author": "Daniel Herfen",  # Author or organization of the custom_node or sub module
    "project": "https://github.com/dnl13/comfy_mtb",  # The address that the `name` value will link to on Node Class Views
    "description": "testing segmantations",
}