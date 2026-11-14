"""JoVox sparse voxel-flow network, re-implemented on top of ComfyUI's native
TRELLIS.2 sparse transformer blocks.

JoVox (Tang et al., arXiv:2611.10690) is a *joint hierarchical discrete flow*
over octant occupancy bytes. It is NOT a TRELLIS.2 variant: the input layer is
gone (the state byte is embedded directly), the RoPE runs on unified
normalised coordinates, the timestep embedding is per-level, and the head
predicts 8 x 256 slot logits. ComfyUI's standard `Trellis2` model therefore
cannot host these weights with a ModelPatcher patch.

What we *can* reuse -- and do reuse -- is ComfyUI's sparse tensor container,
sparse transformer block, sparse attention and (crucially) its quantization
ops, so an `int8_tensorwise` + `convrot` checkpoint runs through Comfy
Kitchen's int8 kernels with no ComfyUI source changes.

Module names match the original `HierVoxelFlow` checkpoint exactly
(`blocks.N.*`, `t_embedder.r0/r1.*`, `out_layer.*`, `point_embed.*`,
`level_embed.*`, `shared_tri_embed`, `grand_proj.*`, `base_embed`).

Original code: github.com/yuyujunjun/JoVox (Apache-2.0),
voxelgen/network/backbone.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.trellis2.model import ModulatedSparseTransformerCrossBlock
from comfy.ldm.trellis2.vae import SparseLinear, SparseTensor

from .hierarchy import MASK_INDEX, NUM_CLASSES, ROOT_LEVEL, native_level_of

UNIFIED_ROPE_RES = 256

# 64^3 generator (the released JoVox-64 checkpoint)
DEFAULT_CONFIG = dict(
    model_channels=1536,
    num_blocks=30,
    num_heads=12,
    cond_channels=1024,
    point_embed_freqs=10,
    num_levels=2,
)


class TimestepEmbedder(nn.Module):
    """Same maths/param names as trellis2 ``sparse_structure_flow.TimestepEmbedder``."""

    def __init__(self, hidden_size, frequency_embedding_size=256, device=None, dtype=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, device=device, dtype=dtype),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class TwoRatioEmbedder(nn.Module):
    """Sum of one timestep embedding per hierarchy level (2 ratios: coarser / finer progress)."""

    def __init__(self, base: nn.Module, d: int, n: int = 2, device=None, dtype=None, operations=None):
        super().__init__()
        assert n >= 2, n
        self.n = int(n)
        self.r0 = base
        for li in range(1, self.n):
            setattr(self, f'r{li}', TimestepEmbedder(d, device=device, dtype=dtype, operations=operations))
        for li in range(2, self.n):
            last = getattr(self, f'r{li}').mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.dim() == 2 and t.shape[1] == self.n, f'progress condition must be [B,{self.n}], got {tuple(t.shape)}'
        return sum(getattr(self, f'r{li}')(t[:, li]) for li in range(self.n))


class PointEmbed(nn.Module):
    """Fourier features of the continuous cell centre, fed through a 2-layer MLP."""

    def __init__(self, out_dim: int, n_freqs: int, device=None, dtype=None, operations=None):
        super().__init__()
        assert n_freqs >= 1, n_freqs
        self.n_freqs = int(n_freqs)
        self.register_buffer(
            'freqs', 2.0 * math.pi * (2.0 ** torch.arange(self.n_freqs, dtype=torch.float32)), persistent=False)
        in_dim = 3 + 6 * self.n_freqs
        self.mlp = nn.Sequential(
            operations.Linear(in_dim, out_dim, device=device, dtype=dtype), nn.SiLU(),
            operations.Linear(out_dim, out_dim, device=device, dtype=dtype),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        assert u.dim() == 2 and u.shape[1] == 3, f'PointEmbed expects [N,3], got {tuple(u.shape)}'
        u = u.float()
        x = u.unsqueeze(-1) * self.freqs
        feat = torch.cat([u, torch.sin(x).flatten(1), torch.cos(x).flatten(1)], dim=1)
        return self.mlp(feat)


class InjectedRope(nn.Module):
    """RoPE whose phases are injected through the sparse tensor's spatial cache.

    Identical rotation maths to JoVox's ``_InjectedRope``; reads
    ``rope_phase_*`` which :meth:`JoVoxFlow.forward` registers on the token set
    before it enters the blocks (unified normalised coordinates, not grid
    coordinates).
    """

    def __init__(self, head_dim: int, dim: int, rope_freq=(1.0, 10000.0)):
        super().__init__()
        self.head_dim = int(head_dim)
        self.dim = int(dim)
        self.rope_freq = tuple(rope_freq)

    def cache_key(self) -> str:
        return (f'rope_phase_{self.dim}d_freq{self.rope_freq[0]}-'
                f'{self.rope_freq[1]}_hd{self.head_dim}')

    @staticmethod
    def _rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        return torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)

    def forward(self, q, k=None):
        key = self.cache_key()
        phases = q.get_spatial_cache(key)
        assert phases is not None, (
            f'rope phases were not injected: looked for {key}, cache has '
            f'{sorted(q.get_spatial_cache().keys())}')
        assert phases.shape[0] == q.feats.shape[0] and phases.shape[-1] == self.head_dim // 2, (
            f'injected phase shape {tuple(phases.shape)} does not match token count '
            f'{q.feats.shape[0]} / head_dim {self.head_dim}')
        q_embed = q.replace(self._rotary_embedding(q.feats, phases))
        if k is None:
            return q_embed
        return q_embed, k.replace(self._rotary_embedding(k.feats, phases))


def _rope_pad(ph: torch.Tensor, head_dim: int) -> torch.Tensor:
    tgt = head_dim // 2
    if ph.shape[-1] < tgt:
        pad = torch.ones(ph.shape[0], tgt - ph.shape[-1], dtype=ph.dtype, device=ph.device)
        ph = torch.cat([ph, pad], dim=-1)
    return ph


class JoVoxFlow(nn.Module):
    """Replaces ``VoxelNetFM.sparse_structure_flow_model`` (HierVoxelFlow).

    Blocks are ComfyUI's ``ModulatedSparseTransformerCrossBlock`` so quantized
    (int8/convrot) linear weights go through ``comfy.ops`` / Comfy Kitchen,
    exactly like the native TRELLIS.2 loader does.
    """

    def __init__(
        self,
        model_channels: int = 1536,
        cond_channels: int = 1024,
        num_blocks: int = 30,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        point_embed_freqs: int = 10,
        num_levels: int = 2,
        dtype: torch.dtype = torch.bfloat16,
        device=None,
        operations=None,      # bf16 mixed-precision ops for the transformer blocks
        head_operations=None,  # fp32 ops for the un-quantized embedding / output layers
    ):
        super().__init__()
        assert num_levels in (2, 3), num_levels
        assert operations is not None and head_operations is not None

        self.resolution = 2 ** native_level_of(num_levels)
        self.model_channels = int(model_channels)
        self.cond_channels = int(cond_channels)
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.num_levels = int(num_levels)
        self.root_level = ROOT_LEVEL
        self.spans = (2,) * self.num_levels
        self.parent_levels = [ROOT_LEVEL + sum(self.spans[:i]) for i in range(self.num_levels)]
        self.dtype = dtype
        self.mask_index = MASK_INDEX
        self.slot_classes = NUM_CLASSES

        self.t_embedder = TwoRatioEmbedder(
            TimestepEmbedder(model_channels, device=device, dtype=torch.float32, operations=head_operations),
            model_channels, n=self.num_levels, device=device, dtype=torch.float32,
            operations=head_operations,
        )

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                share_mod=False,
                qk_rms_norm=True,
                qk_rms_norm_cross=True,
                image_attn_mode="global",
                device=device, dtype=dtype, operations=operations,
            )
            for _ in range(self.num_blocks)
        ])

        self.base_embed = nn.Parameter(torch.zeros(self.model_channels, device=device))
        self.register_buffer('_bit_shifts', torch.arange(8, device=device), persistent=False)
        self.level_embed = nn.Embedding(self.num_levels, self.model_channels, device=device)
        self.point_embed = PointEmbed(
            self.model_channels, int(point_embed_freqs), device=device, dtype=torch.float32,
            operations=head_operations,
        )

        for block in self.blocks:
            rope = getattr(block.self_attn, 'rope', None)
            assert rope is not None, 'block has no rope'
            block.self_attn.rope = InjectedRope(rope.head_dim, rope.dim, rope.rope_freq)

        self._build_rope_unified()

        d = self.model_channels
        self.shared_tri_embed = nn.Parameter(torch.zeros(8, 3, d, device=device))
        self.grand_proj = nn.Linear(8 * d, d, bias=False, device=device, dtype=torch.float32)
        self.out_layer = SparseLinear(d, 8 * NUM_CLASSES, device=device, dtype=torch.float32,
                                      operations=head_operations)
        self.out_channels = 8 * self.slot_classes
        nn.init.zeros_(self.base_embed)
        nn.init.normal_(self.level_embed.weight, std=0.02)
        nn.init.normal_(self.shared_tri_embed, std=0.02)
        nn.init.xavier_uniform_(self.grand_proj.weight)

    # ------------------------------------------------------------------ rope
    def _build_rope_unified(self) -> None:
        head_dim = self.model_channels // self.num_heads
        freq_dim = head_dim // 2 // 3
        freqs = torch.arange(freq_dim, dtype=torch.float32) / freq_dim
        freqs = 1.0 / (10000.0 ** freqs)
        self.register_buffer('_rope_freqs', freqs.to(self.base_embed.device), persistent=False)
        self._rope_fd = int(freq_dim)
        self._rope_hd = int(head_dim)

    @property
    def rope_cache_key(self) -> str:
        return self.blocks[0].self_attn.rope.cache_key()

    def _unified_coords(self, coords: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        # centre of the parent cell, normalised by its grid side: u in (0, 1)
        side = torch.tensor([float(1 << p) for p in self.parent_levels],
                            device=coords.device)[lv]
        return (coords[:, 1:].float() + 0.5) / side.unsqueeze(-1)

    def _unified_phases(self, u: torch.Tensor) -> torch.Tensor:
        x = u.float() * UNIFIED_ROPE_RES
        ph = torch.polar(torch.ones(*x.shape, self._rope_fd, device=x.device),
                         x.unsqueeze(-1) * self._rope_freqs)
        return _rope_pad(ph.reshape(len(u), -1), self._rope_hd)

    # ----------------------------------------------------------------- state
    def _shared_tri(self, child: torch.Tensor, grand: torch.Tensor, has_grand: torch.Tensor) -> torch.Tensor:
        N = child.shape[0]
        T = self.shared_tri_embed.reshape(24, -1)
        sh = self._bit_shifts
        assert child.shape == (N, 8) and int(child.max()) <= 2 and int(child.min()) >= 0
        e = F.embedding_bag(sh.view(1, 8) * 3 + child, T, mode='sum')
        if len(grand):
            n2 = grand.shape[0]
            assert grand.shape == (n2, 8, 8) and int(has_grand.sum()) == n2
            gg = F.embedding_bag((sh.view(1, 1, 8) * 3 + grand).reshape(n2 * 8, 8), T, mode='sum')
            e = e.index_put((torch.nonzero(has_grand).squeeze(1),),
                            e[has_grand] + self.grand_proj(gg.view(n2, -1)))
        return (e + self.base_embed).to(self.base_embed.dtype)

    def embed_state(self, state: torch.Tensor) -> torch.Tensor:
        assert state.dim() == 2 and state.shape[1] == 8, \
            f'two-level state must be [N,8] slots, got {tuple(state.shape)}'
        unk = state == self.mask_index
        patt = torch.where(unk, torch.zeros_like(state), state)
        bits = (patt.unsqueeze(-1) >> self._bit_shifts) & 1
        tri = torch.where(unk.unsqueeze(-1), torch.full_like(bits, 2), bits)
        child = torch.where(unk, torch.full_like(state, 2), (patt > 0).long())
        return self._shared_tri(child, tri,
                                torch.ones(state.shape[0], dtype=torch.bool, device=state.device))

    # --------------------------------------------------------------- forward
    def forward(self, coords: torch.Tensor, state: torch.Tensor, r: torch.Tensor,
                level_idx: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """One velocity/classification pass.

        coords: [N,4] int (batch, x, y, z) parent cells
        state:  [N,8] int64 octant occupancy (0..255) with MASK_INDEX = unknown
        r:      [B,num_levels] progress in [0,1] (per level)
        level_idx: [N] hierarchy level of each token
        cond:   [B,T,1024] DINOv3 tokens
        returns [N,8,256] slot logits
        """
        N = coords.shape[0]
        lv = level_idx.long()
        assert lv.shape == (N,), f'per-token levels {tuple(lv.shape)} do not match token count {N}'

        feats = self.embed_state(state)                      # fp32
        u = self._unified_coords(coords, lv)
        feats = feats + self.point_embed(u).to(feats.dtype)
        feats = feats + self.level_embed(lv)

        x = SparseTensor(feats, coords.int())
        x.register_spatial_cache(self.rope_cache_key, self._unified_phases(u))

        h = x.to(self.dtype)
        t_emb = self.t_embedder(r * 1000.0).to(self.dtype)
        cond = cond.to(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond, transformer_options=None)
        h = h.to(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h.feats.view(N, 8, self.slot_classes)
