# Some Notes

Trying to utilize https://github.com/SysCV/sam-hq
and  https://github.com/facebookresearch/dinov2


## 🚧 TODO
    - 🏗 clean installation of segment-anything with HQ models based on SAM_HQ
    - 🏗 automatic mask detection with segment-anything
    - 🚧 default detection with segment-anything and GroundingDino Dinov1
    - 🚧 try Dinov2 for video consistancy
    - 🚧 optimize mask generation (feather, shift mask, blur, etc)
    - 🚧 add a button/link to documentation at the bottom of complex nodes
    - 🚧 dont forget the documentation
    
## Documents
### Processing Nodes 

<details open>
<summary><strong>Automatic Segmentation</strong></summary>

<blockquote><br>

![Automatic Segmentation](https://raw.githubusercontent.com/dnl13/ComfyUI-dnl13-seg/main/doc_assets/img/nodes/automatic_segmentation.jpg)

<h4>Utilize Automatic Segmentation with SAM (segment-anything)</h4>
Autodetect elements in images and return images as possible greenscreen footage, the element-detected mask in full size of the fed image, a cropped version of the image where the element was detected, also with a separated mask, and a bbox list to later use the detected information in other workflow processes. 

##### Arguments:

- `model (SAM_MODEL)` The SAM model to use for mask prediction.

- `points_per_side (int or None)` 
    The number of points to be sampled along one side of the image. The total number of points is points_per_side. If None, 'point_grids' must provide explicit point sampling.

- `points_per_batch(INT)`
    Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.
        
- `pred_iou_thresh(FLOAT)` 
    A filtering threshold in [0,1], using the model's predicted mask quality.

- `stability_score_thresh(FLOAT)` 
    A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.

- `stability_score_offset(FLOAT)` 
    The amount to shift the cutoff when calculated the stability score.

- `box_nms_thresh(FLOAT)` 
    The box IoU cutoff used by non-maximal suppression to filter duplicate masks.

- `crop_n_layers(int)` 
    If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.

- `crop_nms_thresh(FLOAT)` 
    The box IoU cutoff used by non-maximal
    suppression to filter duplicate masks between different crops.

- `crop_overlap_ratio(FLOAT)`
    Sets the degree to which crops overlap.
    In the first crop layer, crops will overlap by this fraction of
    the image length. Later layers with more crops scale down this overlap.

- `crop_n_points_downscale_factor(int)` 
    The number of points-per-side
    sampled in layer n is scaled down by crop_n_points_downscale_factor**n.

- `point_grids(list(np.ndarray) or None)` 
    A list over explicit grids
    of points used for sampling, normalized to [0,1]. The nth grid in the list is used in the nth crop layer. Exclusive with points_per_side.

- `min_mask_region_area(int)`
    If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.
<!--
- `output_mode(str)` 
    The form masks are returned in. Can be 'binary_mask' 'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
    For large resolutions, 'binary_mask' may consume large amounts of memory.
-->
</blockquote>
</details>


<hr>

### Utility Nodes

<details open>
  <summary><strong>RGB</strong></summary>
<blockquote>

### RGB
TODO: Docs for RGB
Returns a RGB tuple `(R,G,B)`

#### Arguments:

`None`
</blockquote>
</details>