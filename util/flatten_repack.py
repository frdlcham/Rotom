"""
Slot repacking plan for flatten operations over padded dims.

When a merged dim inside ``start..end`` is stored with power-of-2 padding
(e.g. 14 stored as 16), the logical elements of the flattened dim are not
contiguous in the ciphertext: there are 'gaps' of zero slots between them.
Flattening then has to move every logical element from its slot in the input
layout to its row-major flattened slot in the output layout.

The plan is a sequence of stages. Each stage reads the ciphertexts produced by
the previous stage and builds new ones from masked + rotated copies, so one
stage costs one plaintext-mask level. When the slot shift of every element is
a sum of per-dim terms (always true for roll-free, non-replicated layouts),
the shift is applied one tensor dim at a time, innermost merged dim first.
That needs ~sum(extents) rotations instead of ~prod(extents) for a single
stage. If staging is not possible (rolls, replication, collisions), a single
stage moves every element directly.
"""

from dataclasses import dataclass

import numpy as np

from util.layout_util import _get_apply_layout_plan
from util.util import prod


@dataclass(frozen=True)
class RepackMove:
    """``dst_ct += (cts[src_ct] * mask) << rot``; ``mask`` is None when not needed."""

    src_ct: int
    dst_ct: int
    rot: int
    mask: np.ndarray | None


@dataclass(frozen=True)
class RepackPlan:
    stages: tuple
    num_out_ct: int

    def ops(self):
        ops = {"add": 0, "mul": 0, "rot": 0}
        for stage in self.stages:
            dst_cts = set()
            for move in stage:
                ops["rot"] += int(move.rot != 0)
                ops["mul"] += int(move.mask is not None)
                ops["add"] += int(move.dst_ct in dst_cts)
                dst_cts.add(move.dst_ct)
        return ops

    @property
    def depth(self):
        return sum(any(m.mask is not None for m in stage) for stage in self.stages)


def flattened_shape(logical_shape, start, end):
    return (
        list(logical_shape[:start])
        + [prod(logical_shape[start : end + 1])]
        + list(logical_shape[end + 1 :])
    )


def _slot_indices(layout, ndim):
    """(num_ct, n, ndim) tensor index held by every slot; -1 where none."""
    layout_len = max(len(layout), layout.n)
    plan = _get_apply_layout_plan(layout, ndim, layout_len=layout_len)
    base = np.asarray(plan["base_indices_by_cts"], dtype=np.int64)
    if base.shape[2] < ndim:
        # apply_layout treats dims missing from the layout as index 0
        pad = np.zeros(base.shape[:2] + (ndim - base.shape[2],), dtype=np.int64)
        base = np.concatenate([base, pad], axis=2)
    return base[:, :, :ndim]


def _valid(idx, shape):
    shape = np.asarray(shape, dtype=np.int64)
    return ((idx >= 0) & (idx < shape)).all(axis=-1)


def _fit_per_dim_shifts(in_idx, delta, logical_shape):
    """Write ``delta`` as ``c + sum_k g[k][in_idx[:, k]]``, or return None."""
    is_zero = in_idx == 0
    origin = np.flatnonzero(is_zero.all(axis=1))
    if origin.size == 0:
        return None
    c = delta[origin[0]]
    g = []
    for k, extent in enumerate(logical_shape):
        others_zero = np.delete(is_zero, k, axis=1).all(axis=1)
        g_k = np.full(extent, np.iinfo(np.int64).min, dtype=np.int64)
        rows = np.flatnonzero(others_zero)[::-1]  # reversed: first occurrence wins
        g_k[in_idx[rows, k]] = delta[rows] - c
        if (g_k == np.iinfo(np.int64).min).any():
            return None
        g.append(g_k)
    pred = c + sum(g_k[in_idx[:, k]] for k, g_k in enumerate(g))
    if not np.array_equal(pred, delta):
        return None
    return c, g


def _stage_positions(in_idx, src, dst, logical_shape, start, end):
    """List of (from_pos, to_pos) global slot positions, one pair per stage."""
    one_shot = [(src, dst)]
    fit = _fit_per_dim_shifts(in_idx, dst - src, logical_shape)
    if fit is None:
        return one_shot
    c, g = fit

    merged = list(range(end, start - 1, -1))
    order = merged + [k for k in reversed(range(len(logical_shape))) if k not in merged]
    stages = []
    cur = src
    for i, k in enumerate(order):
        step = g[k][in_idx[:, k]] + (c if i == 0 else 0)
        if not step.any():
            continue
        nxt = cur + step
        if (nxt < 0).any() or np.unique(nxt).size != nxt.size:
            return one_shot
        stages.append((cur, nxt))
        cur = nxt
    assert np.array_equal(cur, dst)
    return stages


def _stage_moves(from_pos, to_pos, occupied, n):
    """Group element moves by (src_ct, dst_ct, rotation)."""
    src_ct, src_slot = np.divmod(from_pos, n)
    dst_ct, dst_slot = np.divmod(to_pos, n)
    # ct << k moves slot s to slot s - k
    rot = (src_slot - dst_slot) % n
    keys = np.stack([src_ct, dst_ct, rot], axis=1)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    moves = []
    for g, (s_ct, d_ct, r) in enumerate(uniq):
        mask = np.zeros(n, dtype=bool)
        mask[src_slot[inverse == g]] = True
        # skip the mask when it would keep everything non-zero in src_ct anyway
        if np.array_equal(mask, occupied[s_ct]):
            mask = None
        moves.append(RepackMove(int(s_ct), int(d_ct), int(r), mask))
    return tuple(moves)


def _occupied(pos, num_ct, n):
    occ = np.zeros((num_ct, n), dtype=bool)
    occ.reshape(-1)[pos] = True
    return occ


_PLAN_CACHE = {}


def flatten_repack_plan(in_layout, out_layout, logical_shape, start, end):
    """Plan to move the logical elements of ``in_layout`` to their flattened slots.

    Args:
        in_layout: layout of the flatten input
        out_layout: layout of the flatten output
        logical_shape: unpadded shape of the flatten input; slots holding an
            index outside it are padding and assumed to be zero
        start, end: inclusive range of merged dims

    Returns:
        RepackPlan
    """
    logical_shape = tuple(logical_shape)
    key = (
        in_layout.layout_str(),
        out_layout.layout_str(),
        in_layout.n,
        logical_shape,
        start,
        end,
    )
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]

    n = in_layout.n
    assert out_layout.n == n

    # first input slot holding each logical element
    in_slots = _slot_indices(in_layout, len(logical_shape))
    in_valid = _valid(in_slots, logical_shape)
    in_pos = np.flatnonzero(in_valid.reshape(-1))
    in_lin = np.ravel_multi_index(
        tuple(in_slots.reshape(-1, len(logical_shape))[in_pos].T), logical_shape
    )
    src_of = np.full(prod(logical_shape), -1, dtype=np.int64)
    src_of[in_lin[::-1]] = in_pos[::-1]
    assert (src_of >= 0).all(), "flatten input layout does not hold every element"

    # every output slot holding a logical element is a target
    out_shape = flattened_shape(logical_shape, start, end)
    out_slots = _slot_indices(out_layout, len(out_shape))
    out_valid = _valid(out_slots, out_shape)
    dst = np.flatnonzero(out_valid.reshape(-1))
    out_idx = out_slots.reshape(-1, len(out_shape))[dst]
    merged_idx = np.unravel_index(out_idx[:, start], logical_shape[start : end + 1])
    in_idx = np.concatenate(
        [out_idx[:, :start], np.stack(merged_idx, axis=1), out_idx[:, start + 1 :]],
        axis=1,
    )
    src = src_of[np.ravel_multi_index(tuple(in_idx.T), logical_shape)]

    stages = []
    occupied = in_valid
    for from_pos, to_pos in _stage_positions(
        in_idx, src, dst, logical_shape, start, end
    ):
        stages.append(_stage_moves(from_pos, to_pos, occupied, n))
        occupied = _occupied(to_pos, int(to_pos.max()) // n + 1, n)

    plan = RepackPlan(tuple(stages), out_layout.num_ct())
    _PLAN_CACHE[key] = plan
    return plan


def flatten_repack_plan_for_kernel(kernel):
    """Plan for a FLATTEN_REPACK kernel (``cs = [input, logical_input_shape]``)."""
    from frontends.tensor_args import FlattenArgs

    args = FlattenArgs.from_term(kernel.layout.term)
    return flatten_repack_plan(
        kernel.cs[0].layout, kernel.layout, kernel.cs[1], args.start, args.end
    )
