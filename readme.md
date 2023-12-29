
# IMPORTANT!
This is still in a very early development phase. I plan to conduct thorough research to maximize its potential. 

These nodes will ~~only~~ utilize the ~~**SAM_HQ models**~~ S**AM_HQ an default Metas SAM Models** <small>*maybe later Mobile-SAM and ONNX-Segmentation Models</small>

<hr>





### Development State:
working states of planed Nodes

#### Model Nodes:
-  ✔ GroundingDino Model Loader V1
-  ✔ SAM Model Loader

#### Universal Nodes:
-  🚧 Lazy Mask Segmentation

#### Vision Nodes:
-  ✔ Clip Segmentation Processor
-  🏗 Dino Segmentation Processor
-  🚧 SAM Mask Processor

#### Utility Nodes:
-  🚧 Dino Collection Selector (dnl13s own Pipe System for Mask Selection 🙄)
-  🚧 Batch Selector
-  🚧 Greenscreen Generator
-  🚧 Directional Dropshadow
-  🚧 Resize Mask Directional 
-  🚧 Crop Image ( because nobody was expecting batches and multiple bbox per batch 🙄)
-  🚧 Uncrop Image 

<hr>

## 🚧 TODO

- 🚧 requirements.txt
- 🚧 Try [Dinov2](https://github.com/facebookresearch/dinov2) for video consistency and depth maps
- 🚧 Documentation

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

Feel free to jump into the [discussions](https://github.com/dnl13/ComfyUI-dnl13-seg/discussions) and share your insights!




### References: 
- inspired by https://github.com/storyicon/comfyui_segment_anything/
- inspired CLIPSeg
- included https://github.com/SysCV/sam-hq
- ~~https://github.com/facebookresearch/dinov2~~


<hr>


## ❤ THANK YOU!

First and foremost, I want to express my gratitude to everyone who has contributed to these fantastic tools like ComfyUI and SAM_HQ. Special thanks to [storyicon](https://github.com/storyicon) for their initial implementation, which inspired me to create this repository. These are exceptionally well-crafted works, and I salute the creators. 

I also want to thank everyone who has contributed to the project, whether through pull requests, reporting issues, or simply testing, as your involvement propels the development forward.

I am relatively new here and still gaining experience with GitHub and open-source projects. Therefore, please bear with me if the repository is not yet optimal for you. 
If something is not right, incorrect, or there's a way I can do better, please tell me.

And if you enjoy or find the repo useful, and it brings you joy or success, I would appreciate it if you could consider buying me a coffee on ☕ [Ko-Fi](https://ko-fi.com/dnl13). Many thanks! 💖


## Why?

After discovering [@storyicon](https://github.com/storyicon) implementation [here](https://github.com/storyicon/comfyui_segment_anything/) of Segment Anything, I realized its potential as a powerful tool for ComfyUI if implemented correctly. I delved into the SAM and Dino models. 

The following is my own adaptation of [sam_hq](https://github.com/SysCV/sam-hq) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).


## Credits:

- Special thanks to  [SAM_HQ](https://github.com/SysCV/sam-hq), upon which a significant portion of this code is dependent.
- Thanks [SAM](https://github.com/facebookresearch/segment-anything), [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) for their public code and released models.
- and also [GroundingDino](https://github.com/IDEA-Research/GroundingDINO) for their great work
- And, of course, [ComfyUI](https://github.com/comfyanonymous/ComfyUI), because without this project, this repository wouldn't exist.