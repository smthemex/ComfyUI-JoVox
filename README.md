# ComfyUI-JoVox
[JoVox](https://github.com/yuyujunjun/JoVox): Joint Hierarchical Flow Matching for Lightweight Voxel Generation


## 1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI-JoVox
```
## 2. Models

[modelscope](https://modelscope.cn/models/smthem/JoVox-64)
[huggingface](https://huggingface.co/xiangjuntang/JoVox-64)
```
ComfyUI/models/
├── diffusion_models/JoVox_int8_convrot.safetensors      # or JoVox.safetensors
├── clip_vision/dino_v3_vit_l.safetensors 
```

# 3 .Example
![](https://github.com/smthemexComfyUI-JoVox/blob/main/example_workflows/example.png)

## 4. Citation

```
@article{tang2026jovox,
    title   = {JoVox: Joint Hierarchical Flow Matching for Lightweight Voxel Generation},
    author  = {Tang, Xiangjun and Chen, Jianqi and Wonka, Peter},
    journal = {arXiv preprint arXiv:2611.10690},
    year    = {2026}
}
```