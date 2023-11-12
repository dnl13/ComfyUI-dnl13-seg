from torchvision import transforms
from PIL import Image

def parse_binary_masks(binary_masks):
  
  # TODO: parse_binary_masks and return images, masks, bboxs

  print("parse_binary_masks")

  images = "Tensor BATCH"
  images = "Tensor BATCH"
  bboxs = "LIST"
  return images_masked, masks_fullsize, bboxs
  

def images_greenscreen(images, masks , background_color=[0, 0, 0]):
  print("images_greenscreen")
  image = "IMAGE"
  return image


def crop_to_bbox(images, masks, bboxs):
  print("crop_to_bbox")
  cropped_images="IMAGE"
  cropped_masks="MASKS"
  return cropped_images, cropped_masks 



def tensor_to_pil(tensor):
    rgb = tensor.squeeze(0) if tensor.dim() == 4 else tensor
    rgb = rgb.permute(2, 0, 1)
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(rgb)
    return image_pil