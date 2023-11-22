## Why?

After discovering [@storyicon](https://github.com/storyicon) implementation [here](https://github.com/storyicon/comfyui_segment_anything/) of Segment Anything, I realized its potential as a powerful tool for ComfyUI if implemented correctly. I delved into the SAM and Dino models. 

The following is my own adaptation of [sam_hq](https://github.com/SysCV/sam-hq) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).



# IMPORTANT!
This is still in a very early development phase. I plan to conduct thorough research to maximize its potential. These nodes will only utilize the **SAM_HQ models**.
<hr>


### References: 
- inspired by https://github.com/storyicon/comfyui_segment_anything/
- included https://github.com/SysCV/sam-hq
- ~~https://github.com/facebookresearch/dinov2~~


## 🚧 TODO
- ✔ Clean installation of Segment Anything with HQ models based on SAM_HQ
- ✔ Automatic mask detection with Segment Anything
- ✔ Default detection with Segment Anything and GroundingDino Dinov1
- ✔ Optimize mask generation (feather, shift mask, blur, etc)
- 🚧 Integration of SEGS for better interoperability with, among others, Impact Pack.
- 🚧 Bounding box (bbox) output for detected areas, including images and masks of the bbox.
- 🚧 Optional ability to output Dino "masks" separately.
- 🚧 Possibly use SAM to create masks within bounding boxes.
- 🚧 Add a button/link to documentation at the bottom of complex nodes
- 🚧 Try [Dinov2](https://github.com/facebookresearch/dinov2) for video consistency and depth maps
- 🚧 Don't forget the documentation

    
<hr>

## 🏗 Install 

***be shure to install requirements.txt***

##### ComfyUI Windows Portable Version:
<small>Start your terminal where your `run_nvidia_gpu.bat` or `run_cpu.bat` are located.</small>
```
python_embeded\python.exe -m pip install -r ".\ComfyUI\custom_nodes\ComfyUI-dnl13-seg\requirements.txt"
```

## 📜 Documentation


Due to the fact that the nodes are still in development and subject to change at any time, I encourage you to share your experiences, tips, and tricks in the discussions forum. Once it becomes apparent that the nodes have reached a stable state with minimal changes, I'll be happy to compile our collective knowledge into a wiki for a comprehensive guide on using the nodes. So please don't hesitate to join the discussions.

Feel free to jump into the discussions and share your insights!

### Processing Nodes 

<details>
<summary><strong>Automatic Segmentation</strong></summary>

<blockquote><br>

<h4>Utilize Automatic Segmentation with SAM (segment-anything)</h4>
Autodetect elements in images and return images as possible greenscreen footage, the element-detected mask in full size of the fed image, a cropped version of the image where the element was detected, also with a separated mask, and a bbox list to later use the detected 
information in other workflow processes. 
<br><br>
TODO: read this: https://github.com/facebookresearch/segment-anything/issues/185

#### Arguments:

*Every item marked with (+) has been implemented, while those marked with (-) have been removed after testing. (discussion needed) indicates that we should discuss the relevance of these items.*


```
Automatic Segmentations possible options:

(+) model (Sam): The SAM model to use for mask prediction.

(+) points_per_side (int or None): The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit point sampling.

(-) points_per_batch (int): Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.

(-) pred_iou_thresh (float): A filtering threshold in [0,1], using the model's predicted mask quality.

(+) stability_score_thresh (float): A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.

(-) stability_score_offset (float): The amount to shift the cutoff when calculated the stability score.

(+) box_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks.

(discussion needed)(+) crop_n_layers (int): If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.  

(discussion needed)(+) crop_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops. 

(discussion needed)(+) crop_overlap_ratio (float): Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.

(discussion needed)(-) crop_n_points_downscale_factor (int): The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.       

(discussion needed)(-) point_grids (list(np.ndarray) or None): A list over explicit grids  of points used for sampling, normalized to [0,1]. The nth grid in the  list is used in the nth crop layer. Exclusive with points_per_side.

(+) min_mask_region_area (int): If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.
```
</blockquote>
</details>

<details>
<summary><strong>Mask with prompt</strong></summary>

<blockquote><br>

#### `box_threshold` 
marks the threashold at which confidence the image features are filtered. 
lowering the threashold will result in more image feature. 
***but be aware!! the lower the number the more vram will be consumed***


#### `multimask`
When activated, the node will give you multiple mask and images stacked on the batch_size of the tensor. 
To make a selection later on please use the `BatchSelector-Node` until a selector inside this node is missing. 

#### `clean_mask_holes`  `clean_mask_island` 
`clean_mask_holes` and `clean_mask_island` can take on very large values,
as this seems to reflect the pixel density of the mask. 64 as default value is mainly used to remove small parts of the mask. 
**A value of 0 will therefore not make any corrections to the mask.**


</blockquote>
</details>

<hr>


## ❤ THANK YOU!

First and foremost, I want to express my gratitude to everyone who has contributed to these fantastic tools like ComfyUI and SAM_HQ. Special thanks to [storyicon](https://github.com/storyicon) for their initial implementation, which inspired me to create this repository. These are exceptionally well-crafted works, and I salute the creators. 

I also want to thank everyone who has contributed to the project, whether through pull requests, reporting issues, or simply testing, as your involvement propels the development forward.

I am relatively new here and still gaining experience with GitHub and open-source projects. Therefore, please bear with me if the repository is not yet optimal for you. 
If something is not right, incorrect, or there's a way I can do better, please tell me.

And if you enjoy or find the repo useful, and it brings you joy or success, I would appreciate it if you could consider buying me a coffee on ☕ [Ko-Fi](https://ko-fi.com/dnl13). Many thanks! 💖

## Credits:

- Special thanks to  [SAM_HQ](https://github.com/SysCV/sam-hq), upon which a significant portion of this code is dependent.
- Thanks [SAM](https://github.com/facebookresearch/segment-anything), [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) for their public code and released models.
- and also [GroundingDino](https://github.com/IDEA-Research/GroundingDINO) for their great work
- And, of course, [ComfyUI](https://github.com/comfyanonymous/ComfyUI), because without this project, this repository wouldn't exist.