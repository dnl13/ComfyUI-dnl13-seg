import torch
import numpy as np
from PIL import Image
import cv2


def mask2cv(mask_np):
  """
  Requires:
    input mask as # H.W.C np.array
  """
  msk_cv2 = mask_np.astype(np.float32)
  msk_cv2 = msk_cv2 * 255
  return msk_cv2


def shrink_grow_mskcv(mask, factor):
  """
  shrinks or grow a mask by factor in the rande of -x to +x 
  """
  if factor > 0:
      # Dilation (Grow)
      mask = cv2.dilate(mask, kernel=np.ones((3, 3), np.uint8), iterations=abs(factor))
  elif factor < 0:
      # Erosion (Shrink)
      mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=abs(factor))
  else:
      pass
  return mask


def blur_mskcv(mask, blur_factor):
  """
  blures a mask 
  """
  if blur_factor % 2 == 0:  
    blur_factor += 1  

  blured_mask = cv2.GaussianBlur(mask, (blur_factor, blur_factor), 15)
  return blured_mask

def img_combine_mask_rgba(image_np_rgb,msk_cv2_blurred):
   image_with_alpha = Image.fromarray(np.concatenate((image_np_rgb, msk_cv2_blurred), axis=2).astype(np.uint8), 'RGBA')
   return image_with_alpha


def split_image_mask(image, device):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device=device)
    return (image_rgb, mask)