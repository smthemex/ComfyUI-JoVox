"""JoVox sampler driver: batched joint hierarchical sampling -> 64^3 occupancy.

Ported from JoVox (github.com/yuyujunjun/JoVox, Apache-2.0),
voxelgen/network/sampler.py (``sample`` / ``_CondBoundNet``).
"""

from typing import List

import numpy as np
import torch

from .hierarchy import MASK_INDEX, NUM_CLASSES, _key, _unkey, expand_two_level
from .schedule import ROOT_TOKEN_LEVEL, run


def _forward_cfg(net, coords, state, r, level_idx, cond, neg_cond, cfg_strength):
    """Classifier-free guidance over slot logits (JoVox guidance, not the eps guider)."""
    if cfg_strength == 1.0 or neg_cond is None:
        return net(coords, state, r, level_idx, cond)

    B = int(cond.shape[0])
    coords2 = coords.clone()
    coords2[:, 0] += B
    if torch.is_tensor(level_idx) and level_idx.dim() == 1 and level_idx.shape[0] == coords.shape[0]:
        level_idx = torch.cat([level_idx, level_idx], 0)
    both = net(torch.cat([coords, coords2], 0),
               torch.cat([state, state], 0),
               torch.cat([r, r], 0), level_idx,
               torch.cat([cond, neg_cond], 0))
    n = coords.shape[0]
    pos, neg = both[:n], both[n:]
    fin_p, fin_n = torch.isfinite(pos), torch.isfinite(neg)
    assert bool((fin_p == fin_n).all()), "conditional and unconditional branches have different invalid masks"
    out = torch.where(fin_p, neg + cfg_strength * (pos - neg),
                      torch.full_like(pos, float('-inf')))
    assert not torch.isnan(out).any(), "NaN after guidance"
    return out


@torch.inference_mode()
def sample(net, cond: torch.Tensor, neg_cond: torch.Tensor, n_samples: int, steps: int,
           seed: int = 0, sched=None, root_end=(0.8,), cfg_strength: float = 1.5,
           temperature: float = 0.0, device: str = 'cuda', log=None,
           on_step=None) -> List[np.ndarray]:
    """Run the JoVox sampler. Returns one sorted level-6 (64^3) key array per sample."""
    greedy = float(temperature) == 0.0
    assert greedy or float(temperature) > 0, f'temperature must be >= 0, got {temperature}'
    assert sched is not None and root_end is not None, 'a GlobalSchedule and the window ends are required'
    assert cfg_strength >= 1.0, f'cfg_strength must be >= 1, got {cfg_strength}'

    L = net.num_levels
    grid = net.root_level + 2 * L  # finest bit level (6 for 64^3)
    assert L in (2, 3), L
    ends = tuple(float(x) for x in root_end)
    assert len(ends) == L - 1, f'{L} levels need {L - 1} window ends, got {ends}'
    e = ends[0] if L == 2 else ends
    N, B = int(steps), int(n_samples)
    if L == 2:
        assert sched.n_lo <= N <= sched.n_hi, \
            f'inference steps {N} outside the training range [{sched.n_lo},{sched.n_hi}]'
    assert all(sched.e_lo <= x <= sched.e_hi for x in ends), \
        f'inference window ends {ends} outside the training range [{sched.e_lo},{sched.e_hi}]'

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    lev_xyz = [torch.as_tensor(_unkey(np.arange(8 ** (ROOT_TOKEN_LEVEL + 2 * li)), ROOT_TOKEN_LEVEL + 2 * li),
                               device=device) for li in range(L)]
    n_fwd = [0]

    def draw(st, ex, active):
        exl = [None, ex] if L == 2 else ex
        idx = [torch.nonzero(exl[li], as_tuple=True) for li in range(1, L)]
        coords = torch.cat(
            [torch.cat([torch.arange(B, device=device).repeat_interleave(64)[:, None], lev_xyz[0].repeat(B, 1)], 1)]
            + [torch.cat([bi[:, None], lev_xyz[li][st.cells_l[li][bi, di]]], 1)
               for li, (bi, di) in enumerate(idx, 1)], 0).int()
        state = torch.cat([st.val[0].reshape(-1, 8)] + [st.val[li][bi, di] for li, (bi, di) in enumerate(idx, 1)], 0)
        lev = torch.cat([torch.zeros(B * 64, dtype=torch.long, device=device)]
                        + [torch.full((len(bi),), li, dtype=torch.long, device=device)
                           for li, (bi, di) in enumerate(idx, 1)])
        R = st.ratios(ex)
        order = torch.argsort(coords[:, 0], stable=True)
        lg = torch.empty((len(lev), 8, NUM_CLASSES), device=device)
        lg[order] = _forward_cfg(net, coords[order], state[order], R, lev[order],
                                 cond, neg_cond, float(cfg_strength)).float()
        assert lg.shape == (len(lev), 8, NUM_CLASSES), tuple(lg.shape)
        if greedy:
            d = lg.reshape(-1, NUM_CLASSES).argmax(-1).view(-1, 8)
        else:
            d = torch.multinomial(torch.softmax(lg.reshape(-1, NUM_CLASSES) / float(temperature), -1), 1,
                                  generator=gen).view(-1, 8)
        vals = [d[:B * 64].view(B, 64, 8)]
        o = B * 64
        for li, (bi, di) in enumerate(idx, 1):
            v = torch.full(st.val[li].shape, 0, dtype=d.dtype, device=device)
            v[bi, di] = d[o:o + len(bi)]
            o += len(bi)
            vals.append(v)
        n_fwd[0] += 1
        if on_step is not None:
            on_step()
        return tuple(vals)

    st = run(B, sched, gen, [(e, N, 1.0)] * B, draw, device=device)
    if log is not None:
        log(f'{N} grid steps + {sched.C} cleanup steps, {n_fwd[0]} forward passes; window ends {ends}')

    last, p_last = L - 1, ROOT_TOKEN_LEVEL + 2 * (L - 1)
    ex = st.exist_all()[last].cpu().numpy()
    cells = st.cells_l[last].cpu().numpy()
    dval = st.val[last].cpu().numpy()
    assert (dval[ex] != MASK_INDEX).all()
    out = []
    for b in range(B):
        keys = cells[b][ex[b]]
        if len(keys) == 0:
            out.append(np.zeros(0, np.int64))
            continue
        order = np.argsort(keys)
        xyz = expand_two_level(keys[order], dval[b][ex[b]][order], p_last)
        out.append(np.sort(_key(xyz, p_last + 2)) if len(xyz) else np.zeros(0, np.int64))
    return out


def keys_to_voxel(keys: List[np.ndarray], resolution: int = 64) -> torch.Tensor:
    """Sorted level-6 key arrays -> dense [B,res,res,res] float32 occupancy (0/1)."""
    assert resolution == 2 ** 6, 'only the 64^3 native grid is produced'
    B = len(keys)
    vox = torch.zeros((B, resolution, resolution, resolution), dtype=torch.float32)
    for b, k in enumerate(keys):
        if len(k) == 0:
            continue
        xyz = _unkey(k, 6)
        vox[b, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = 1.0
    return vox
