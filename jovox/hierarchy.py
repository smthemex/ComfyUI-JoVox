"""JoVox hierarchical occupancy encoding (2 levels: 4^3 -> 32^3 structure -> 64^3).

Ported from JoVox (github.com/yuyujunjun/JoVox, Apache-2.0), voxelgen/network/hierarchy.py.
Coordinate convention matches TRELLIS.2: coords are [x, y, z] (z-up voxel frame),
keys pack them as (x << 2L) | (y << L) | z for an L-level grid.
"""

import numpy as np

ROOT_LEVEL = 2          # root tokens sit on a 4^3 grid
NUM_CLASSES = 256       # one byte of occupancy per root / parent token (8 octant bits)
MASK_INDEX = NUM_CLASSES  # "unknown" slot value
SLOT_OCTANT = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)], dtype=np.int64)


def native_level_of(num_levels: int) -> int:
    """Bit-level of the finest grid: 2 levels -> 6 (64^3), 3 levels -> 8 (256^3)."""
    assert num_levels in (2, 3), f'num_levels must be 2 (64^3) or 3 (256^3), got {num_levels}'
    return ROOT_LEVEL + 2 * num_levels


def _key(coords: np.ndarray, level: int) -> np.ndarray:
    return (coords[:, 0] << (2 * level)) | (coords[:, 1] << level) | coords[:, 2]


def _unkey(keys: np.ndarray, level: int) -> np.ndarray:
    m = (1 << level) - 1
    return np.stack([(keys >> (2 * level)) & m, (keys >> level) & m, keys & m], axis=1)


def expand_children(parent_keys: np.ndarray, labels: np.ndarray, parent_level: int) -> np.ndarray:
    xyz = _unkey(parent_keys, parent_level)
    out = []
    for bit in range(8):
        sel = (labels >> bit) & 1
        idx = np.nonzero(sel)[0]
        if len(idx) == 0:
            continue
        ch = (xyz[idx] << 1) + np.array([(bit >> 2) & 1, (bit >> 1) & 1, bit & 1], dtype=np.int64)
        out.append(ch)
    return np.unique(np.concatenate(out, 0), axis=0) if out else np.zeros((0, 3), dtype=np.int64)


def expand_two_level(parent_keys: np.ndarray, slots: np.ndarray, p: int) -> np.ndarray:
    """Expand level-p cells (+ their 8 slot bits, i.e. level p+1 occupancy) to level p+2 cells."""
    if len(parent_keys) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    pxyz = _unkey(parent_keys, p)
    cxyz = ((pxyz[:, None, :] << 1) + SLOT_OCTANT[None]).reshape(-1, 3)
    vals = slots.reshape(-1)
    keep = vals != 0
    if not keep.any():
        return np.zeros((0, 3), dtype=np.int64)
    return expand_children(_key(cxyz[keep], p + 1), vals[keep], p + 1)
