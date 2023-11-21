# In the __init__.py file of your main directory
#import sys
#import os

# Get the current path to the directory of the __init__.py file
#current_path = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (project directory) to sys.path
#project_path = os.path.dirname(current_path)
#sys.path.append(project_path)

import os
import sys

# Füge das Verzeichnis des Moduls zum sys.path hinzu
#sys.path.append(os.path.join(os.path.dirname(__file__), "nodes"))
sys.path.append(os.path.join(os.path.dirname(__file__), "libs"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Import Nodes
from .index import NODE_CLASS_MAPPINGS

# for later use
# WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS']



