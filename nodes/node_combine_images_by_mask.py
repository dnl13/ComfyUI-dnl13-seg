import numpy as np
import torch
from PIL import Image
from ..utils.collection import is_rgba_tensor

class CombineImagesByMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ('IMAGE', {}),
                "background": ('IMAGE', {}),
                "mask": ('MASK', {}),
                
            }
        }
    CATEGORY = "dnl13/utilitys"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)

    def main(self, foreground, background, mask):


        if mask.size(0) > 1 and len(mask.size()) == 4:
            raise AssertionError("Multimask is NOT SUPPORTED!")

        if len(mask.size()) == 4:
            # remove batch_size from tensor
            mask = mask.squeeze(dim=0)

        if is_rgba_tensor(foreground):
            foreground = foreground[..., :3]
        if is_rgba_tensor(background):
            background = background[..., :3]
        mask_bhwc = mask.permute(1,2,0)

        foreground_pil = Image.fromarray((foreground * 255).to(torch.uint8).squeeze().cpu().numpy(), 'RGB')
        background_pil = Image.fromarray((background * 255).to(torch.uint8).squeeze().cpu().numpy(), 'RGB')
        mask_pil = Image.fromarray((mask_bhwc.squeeze().cpu().numpy() * 255).astype(np.uint8), 'L')
        result_pil = Image.composite(foreground_pil, background_pil, mask_pil)


        result_tensor = torch.tensor(np.array(result_pil), dtype=torch.float32) / 255.0
        result_tensor = result_tensor.unsqueeze(0)


        return result_tensor, 