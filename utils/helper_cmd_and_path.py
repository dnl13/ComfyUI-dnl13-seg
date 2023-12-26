import os
import folder_paths
from torch.hub import download_url_to_file
from urllib.parse import urlparse

def print_labels(to_return):
    list = {
        "dnl13":"\033[92m(dnl13-seg)\033[0m",
    }
    return list[to_return]

def get_local_filepath(url, dirname, local_file_name=None):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)
    destination = os.path.join(dirname, local_file_name)
    if not os.path.exists(destination):
        download_url_to_file(url, destination)
    return destination
