"""ComfyUI nodes: load JoVox, sample 64^3 voxels from upstream DINO conditioning."""

import logging
import os
import time

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import Types

from .jovox.sampler import keys_to_voxel, sample
from .jovox.weights import load_jovox

_CACHE = {}


def _log(msg):
    logging.info(f'[JoVox] {msg}')


def _get_model(path: str):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    key = (path, mtime)
    model = _CACHE.get(key)
    if model is None:
        for old in list(_CACHE):
            if old[0] == path:
                del _CACHE[old]
        model = load_jovox(path, device='cpu')
        _CACHE[key] = model
    return model


def _extract_cond(conditioning, name: str) -> torch.Tensor:
    if not conditioning:
        raise ValueError(f'JoVox: {name} conditioning is empty')
    entry = conditioning[0]
    cond = entry[0] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else entry
    if isinstance(cond, dict):
        raise TypeError(
            f'JoVox: {name} conditioning carries packed extras instead of a token tensor; '
            'connect the Trellis2/Pixal3D Conditioning node output directly'
        )
    if not torch.is_tensor(cond):
        raise TypeError(f'JoVox: {name} conditioning entry is {type(cond)}, expected a tensor')
    if cond.ndim == 2:
        cond = cond.unsqueeze(0)
    if cond.ndim != 3:
        raise ValueError(f'JoVox: {name} conditioning must be [B,T,D], got {tuple(cond.shape)}')
    if cond.shape[-1] != 1024:
        raise ValueError(
            f'JoVox: {name} conditioning has {cond.shape[-1]} channels, expected 1024 (DINOv3 ViT-L/16). '
            'Use a Trellis2 Conditioning or Pixal3D Conditioning node with dino_v3_vit_l clip vision.'
        )
    return cond.float()


class JoVoxModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "JoVox-64 safetensors, int8/convrot quantized or fp32/bf16. "
                               "The int8 file runs through Comfy Kitchen int8 kernels when supported.",
                }),
            },
        }

    RETURN_TYPES = ("JOVOX_MODEL",)
    RETURN_NAMES = ("jovox",)
    FUNCTION = "load"
    CATEGORY = "loaders/trellis"
    DESCRIPTION = "Load the JoVox 64^3 voxel generator (hierarchical discrete flow)."

    def load(self, model_name):
        path = folder_paths.get_full_path("diffusion_models", model_name)
        if path is None:
            raise FileNotFoundError(f'JoVox checkpoint not found: {model_name}')
        model = _get_model(path)
        return (model,)


class JoVoxVoxelSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "jovox": ("JOVOX_MODEL",),
                "positive": ("CONDITIONING", {
                    "tooltip": "Upstream DINOv3 conditioning (Trellis2 Conditioning / Pixal3D Conditioning).",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "steps": ("INT", {"default": 72, "min": 36, "max": 108, "step": 1,
                                  "tooltip": "Grid steps (training range 36..108). 72 is the released default."}),
                "cfg": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.05,
                                  "tooltip": "JoVox logit guidance. 1.0 disables CFG (negative branch unused)."}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                          "tooltip": "0 = greedy (released default), >0 samples slot classes."}),
            },
            "optional": {
                "negative": ("CONDITIONING", {
                    "tooltip": "Optional. JoVox's unconditional branch is the zero conditioning, used when absent.",
                }),
                "latent": ("LATENT", {
                    "tooltip": "Optional default latent (e.g. Empty Trellis2 Latent Structure). "
                               "Only its batch size is used.",
                }),
            },
        }

    RETURN_TYPES = ("VOXEL", "VOXEL")
    RETURN_NAMES = ("voxel", "voxel_structure")
    FUNCTION = "generate"
    CATEGORY = "3d/trellis"
    DESCRIPTION = ("JoVox joint hierarchical flow sampling: DINO conditioning -> 64^3 occupancy voxel. "
                   "Replaces the TRELLIS.2 sparse-structure sampler + VAE decode structure stage.")

    def generate(self, jovox, positive, seed, steps, cfg, temperature, negative=None, latent=None):
        cond = _extract_cond(positive, 'positive')
        neg = _extract_cond(negative, 'negative') if negative else torch.zeros_like(cond)

        n_samples = cond.shape[0]
        if latent is not None and isinstance(latent, dict) and torch.is_tensor(latent.get("samples")):
            n_latent = int(latent["samples"].shape[0])
            if n_latent != n_samples:
                if n_samples == 1:
                    cond = cond.expand(n_latent, *cond.shape[1:]).contiguous()
                    neg = neg.expand(n_latent, *neg.shape[1:]).contiguous()
                    n_samples = n_latent
                else:
                    raise ValueError(
                        f'JoVox: latent batch size {n_latent} does not match the conditioning batch {n_samples}')
        if neg.shape != cond.shape:
            raise ValueError(
                f'JoVox: negative conditioning {tuple(neg.shape)} does not match positive {tuple(cond.shape)}; '
                'both must come from the same Conditioning node (same image / token count)')

        device = comfy.model_management.get_torch_device()
        flow = jovox.flow
        if next(flow.parameters()).device != device:
            _log(f'moving weights to {device}')
            jovox.to(device)
        cond = cond.to(device)
        neg = neg.to(device)

        total = int(steps) + jovox.sched.C
        pbar = comfy.utils.ProgressBar(total)

        def _on_step():
            comfy.model_management.throw_exception_if_processing_interrupted()
            pbar.update(1)

        _log(f'sampling {n_samples} sample(s): steps={steps} cfg={cfg:g} temperature={temperature:g} '
             f'seed={seed} | {jovox.sched.describe()}')
        t0 = time.time()
        keys = sample(
            flow, cond, neg, n_samples, int(steps), seed=int(seed), sched=jovox.sched,
            root_end=jovox.root_end, cfg_strength=float(cfg), temperature=float(temperature),
            device=str(device), log=_log, on_step=_on_step,
        )
        voxel = keys_to_voxel(keys, 64)
        counts = [int(len(k)) for k in keys]
        _log(f'done in {time.time() - t0:.1f}s | occupied cells (64^3 coords): {counts}')

        # The 64^3 occupancy is predicted from the level-5 octant bytes, so the
        # level-5 support (32^3) is its exact "structure" resolution: a 2x max-pool
        # of the 64^3 grid recovers it without loss. That is the grid the
        # TRELLIS.2 shape stage expects (coord_resolution <= 32 -> 512 path).
        voxel_structure = (F.max_pool3d(voxel, 2) > 0.5).float()
        return (Types.VOXEL(voxel, resolution=64), Types.VOXEL(voxel_structure, resolution=32))


NODE_CLASS_MAPPINGS = {
    "JoVoxModelLoader": JoVoxModelLoader,
    "JoVoxVoxelSampler": JoVoxVoxelSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoVoxModelLoader": "JoVox Model Loader",
    "JoVoxVoxelSampler": "JoVox Voxel Sampler",
}
