"""JoVox checkpoint loading through ComfyUI's quantization ops (int8/convrot aware)."""

import logging

import torch

import comfy.ops
import comfy.utils

from .model import DEFAULT_CONFIG, JoVoxFlow
from .schedule import GlobalSchedule

# Sampling configuration of the released JoVox-64 model (configs/voxelgen64.yaml).
SCHEDULE_DEFAULT = dict(
    power=2.0,
    root_end_center=0.8,
    root_end_lo=0.6,
    root_end_hi=1.0,
    root_end_point_mass=0.5,
    steps_lo=36,
    steps_hi=108,
    cleanup_steps=8,
    remask_steps=1,
)
ROOT_END_DEFAULT = (0.8,)
SAMPLING_DEFAULT = dict(steps=72, cfg_strength=1.5, temperature=0.0)


class JoVoxModel:
    """Loaded JoVox generator + its sampler configuration."""

    def __init__(self, flow: JoVoxFlow, path: str, dtype: torch.dtype, quant_layers: int):
        self.flow = flow
        self.path = path
        self.dtype = dtype
        self.quant_layers = quant_layers
        self.sched = GlobalSchedule(**SCHEDULE_DEFAULT)
        self.root_end = ROOT_END_DEFAULT

    def describe(self) -> str:
        return (f'JoVox {self.flow.num_levels}-level ({self.flow.resolution}^3), '
                f'{self.flow.model_channels}x{self.flow.num_blocks}x{self.flow.num_heads}, '
                f'{self.quant_layers} quantized linear layers, weights {self.dtype}')

    def to(self, device):
        self.flow.to(device)
        return self


def _ops_pair():
    disabled = comfy.ops.get_disabled_quant_formats(None)
    ops = comfy.ops.mixed_precision_ops({}, torch.bfloat16, disabled=disabled)
    head_ops = comfy.ops.mixed_precision_ops({}, torch.float32, disabled=disabled)
    return ops, head_ops, disabled


def load_jovox(path: str, device=None) -> JoVoxModel:
    logging.info(f'[JoVox] loading {path}')
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    sd = comfy.utils.state_dict_prefix_replace(
        sd, {'sparse_structure_flow_model.': ''}, filter_keys=True)
    sd.pop('_fm_step', None)

    ops, head_ops, disabled = _ops_pair()
    if disabled:
        logging.info(f'[JoVox] quant formats without native kernels here (emulated): {sorted(disabled)}')

    flow = JoVoxFlow(**DEFAULT_CONFIG, dtype=torch.bfloat16, device=device,
                     operations=ops, head_operations=head_ops)
    missing, unexpected = flow.load_state_dict(sd, strict=False)
    del sd

    if missing or unexpected:
        raise RuntimeError(
            'JoVox checkpoint does not match the 64^3 architecture:'
            f' missing={sorted(missing)[:8]}{"..." if len(missing) > 8 else ""}'
            f' unexpected={sorted(unexpected)[:8]}{"..." if len(unexpected) > 8 else ""}'
        )

    flow.eval()
    flow.requires_grad_(False)
    if device is not None:
        flow.to(device)
    quant = sum(1 for m in flow.modules() if getattr(m, 'quant_format', None) is not None)
    model = JoVoxModel(flow, path, torch.bfloat16, quant)
    logging.info(f'[JoVox] {model.describe()}')
    return model
