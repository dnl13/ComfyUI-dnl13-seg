# some fedaults
import os

# some comfyzu defaults
import folder_paths


#some torch
import torch
from torch.hub import download_url_to_file

####### all the file stuff
# model information taken from the repo and their huggingface page
informations_about_models = {
  "tiny":{
    "id":"vit_tiny",
    "filename":"sam_hq_vit_tiny.pth",
    "url":"https://huggingface.co/lkeab/hq-sam/raw/main/sam_hq_vit_tiny.pth"
  },
  "base":{
    "id":"vit_b",
    "filename":"sam_hq_vit_b.pth",
    "url":"https://huggingface.co/lkeab/hq-sam/raw/main/sam_hq_vit_b.pth"
  },
  "large":{
    "id":"vit_l",
    "filename":"sam_hq_vit_l.pth",
    "url":"https://huggingface.co/lkeab/hq-sam/raw/main/sam_hq_vit_l.pth"
  },
  "huge":{
    "id":"vit_h",
    "filename":"sam_hq_vit_h.pth",
    "url":"https://huggingface.co/lkeab/hq-sam/raw/main/sam_hq_vit_h.pth"
  }
}

#model location in comfyui models folder
sam_model_dirctory_path = os.path.join(folder_paths.models_dir, "sams")

# if there is no sams models folder - create one
if not os.path.exists(sam_model_dirctory_path):
    os.makedirs(sam_model_dirctory_path)


def get_model_location(modelSize="base"):
    model_info = informations_about_models[modelSize]
    model_path_and_filename = os.path.join(sam_model_dirctory_path, model_info["filename"])
    if not os.path.exists(model_path_and_filename):
        downloadModel(modelSize)
    return model_path_and_filename
    
def downloadModel(modelSize="base"):
  model_path_and_filename = get_model_location(modelSize)
  download_url_to_file(informations_about_models[modelSize]["url"], model_path_and_filename)
