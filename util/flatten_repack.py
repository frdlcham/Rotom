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

The same plan also implements GROUP_PAD, which keeps the tensor shape but moves
the elements of dims ``start..end`` into one group dim padded as a whole (see
``Dim.group_dim``).
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


def _stage_positions(in_idx, src, dst, logical_shape, start, end, n):
    """List of (from_pos, to_pos) global slot positions, one pair per stage."""
    one_shot = [(src, dst)]
    fit = _fit_per_dim_shifts(in_idx, dst - src, logical_shape)
    if fit is None:
        return one_shot
    c, g = fit

    merged = list(range(end, start - 1, -1))
    order = merged + [k for k in reversed(range(len(logical_shape))) if k not in merged]
    positions = [src]
    for i, k in enumerate(order):
        step = g[k][in_idx[:, k]] + (c if i == 0 else 0)
        if not step.any():
            continue
        nxt = positions[-1] + step
        if np.unique(nxt).size != nxt.size:
            return one_shot
        positions.append(nxt)
    assert np.array_equal(positions[-1], dst)

    # Intermediate positions may be negative (e.g. a dim shifted left before
    # the dims that move it right). Offsetting them by whole ciphertexts keeps
    # every rotation amount and only renumbers the temporary ciphertexts.
    if len(positions) > 2:
        lo = min(int(p.min()) for p in positions[1:-1])
        if lo < 0:
            offset = -(lo // n) * n
            positions[1:-1] = [p + offset for p in positions[1:-1]]
    return list(zip(positions[:-1], positions[1:]))


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


def _source_positions(in_layout, logical_shape):
    """(first input slot holding each logical element, occupied input slots)."""
    in_slots = _slot_indices(in_layout, len(logical_shape))
    in_valid = _valid(in_slots, logical_shape)
    in_pos = np.flatnonzero(in_valid.reshape(-1))
    in_lin = np.ravel_multi_index(
        tuple(in_slots.reshape(-1, len(logical_shape))[in_pos].T), logical_shape
    )
    src_of = np.full(prod(logical_shape), -1, dtype=np.int64)
    src_of[in_lin[::-1]] = in_pos[::-1]
    assert (src_of >= 0).all(), "repack input layout does not hold every element"
    return src_of, in_valid


def _plan_from_positions(in_idx, src, dst, occupied, logical_shape, start, end, n, num_out_ct):
    """Plan moving element ``in_idx[i]`` from global slot ``src[i]`` to ``dst[i]``.

    ``occupied`` (num_ct, n) marks the input slots that may be non-zero.
    """
    stages = []
    for from_pos, to_pos in _stage_positions(
        in_idx, src, dst, logical_shape, start, end, n
    ):
        stages.append(_stage_moves(from_pos, to_pos, occupied, n))
        occupied = _occupied(to_pos, int(to_pos.max()) // n + 1, n)
    return RepackPlan(tuple(stages), num_out_ct)


def _build_plan(in_layout, out_layout, logical_shape, start, end, dst, in_idx):
    """Plan moving input element ``in_idx[i]`` to global output slot ``dst[i]``."""
    n = in_layout.n
    assert out_layout.n == n
    src_of, in_valid = _source_positions(in_layout, logical_shape)
    src = src_of[np.ravel_multi_index(tuple(in_idx.T), logical_shape)]
    return _plan_from_positions(
        in_idx, src, dst, in_valid, logical_shape, start, end, n, out_layout.num_ct()
    )


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
        "flatten",
        in_layout.layout_str(),
        out_layout.layout_str(),
        in_layout.n,
        logical_shape,
        start,
        end,
    )
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]

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

    plan = _build_plan(in_layout, out_layout, logical_shape, start, end, dst, in_idx)
    _PLAN_CACHE[key] = plan
    return plan


def group_pad_plan(in_layout, out_layout, logical_shape, start, end):
    """Plan to move the logical elements of ``in_layout`` to their ``out_layout`` slots.

    The tensor shape is unchanged; ``out_layout`` holds dims ``start..end`` in
    one group dim (see ``Dim.group_dim``).

    Args:
        in_layout: layout of the group pad input
        out_layout: layout of the group pad output
        logical_shape: unpadded shape of the tensor; slots holding an index
            outside it are padding and assumed to be zero
        start, end: inclusive range of grouped dims

    Returns:
        RepackPlan
    """
    logical_shape = tuple(logical_shape)
    key = (
        "group_pad",
        in_layout.layout_str(),
        out_layout.layout_str(),
        in_layout.n,
        logical_shape,
        start,
        end,
    )
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]

    # every output slot holding a logical element is a target
    out_slots = _slot_indices(out_layout, len(logical_shape))
    out_valid = _valid(out_slots, logical_shape)
    dst = np.flatnonzero(out_valid.reshape(-1))
    in_idx = out_slots.reshape(-1, len(logical_shape))[dst]
    covered = np.zeros(prod(logical_shape), dtype=bool)
    covered[np.ravel_multi_index(tuple(in_idx.T), logical_shape)] = True
    assert covered.all(), "group pad output layout does not hold every element"

    plan = _build_plan(in_layout, out_layout, logical_shape, start, end, dst, in_idx)
    _PLAN_CACHE[key] = plan
    return plan


def depthwise_anchor_rows(in_layout, in_shape, out_dhw, strides):
    """(num_ct, n, 4) output voxel computed at each slot of ``in_layout`` (-1 = none).

    A strided depthwise Conv3D on a GROUP_PAD input first computes output voxel
    ``(c, o)`` in the slot that holds its anchor, input voxel ``(c, o * stride)``:
    every filter tap is then one uniform rotation along the input layout. When
    the input layout replicates a voxel, only its first slot is an anchor.
    """
    rows = _slot_indices(in_layout, 4)
    strides = np.asarray(strides, dtype=np.int64)
    out_shape = (int(in_shape[0]),) + tuple(int(o) for o in out_dhw)
    spatial = rows[..., 1:]
    ok = _valid(rows, in_shape) & (spatial % strides == 0).all(axis=-1)
    out = np.concatenate([rows[..., :1], spatial // strides], axis=-1)
    ok &= _valid(out, out_shape)

    flat_ok = ok.reshape(-1)
    pos = np.flatnonzero(flat_ok)
    lin = np.ravel_multi_index(tuple(out.reshape(-1, 4)[pos].T), out_shape)
    _, first = np.unique(lin, return_index=True)
    keep = np.zeros_like(flat_ok)
    keep[pos[first]] = True

    anchors = np.full_like(rows, -1)
    anchors[keep.reshape(ok.shape)] = out[keep.reshape(ok.shape)]
    return anchors


def depthwise_repack_plan(in_layout, out_layout, in_shape, out_dhw, strides):
    """Plan moving strided depthwise outputs from their anchors to ``out_layout``.

    See ``depthwise_anchor_rows``. Every spatial shift is a sum of per-axis
    terms, so the staged plan needs ~sum(out extents) rotations.
    """
    in_shape = tuple(int(s) for s in in_shape)
    out_shape = (in_shape[0],) + tuple(int(o) for o in out_dhw)
    strides = tuple(int(s) for s in strides)
    key = (
        "depthwise",
        in_layout.layout_str(),
        out_layout.layout_str(),
        in_layout.n,
        in_shape,
        out_shape,
        strides,
    )
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]

    n = in_layout.n
    assert out_layout.n == n
    anchors = depthwise_anchor_rows(in_layout, in_shape, out_shape[1:], strides)
    occupied = anchors[..., 0] >= 0
    flat = anchors.reshape(-1, 4)
    pos = np.flatnonzero(occupied.reshape(-1))
    src_of = np.full(prod(out_shape), -1, dtype=np.int64)
    src_of[np.ravel_multi_index(tuple(flat[pos].T), out_shape)] = pos
    assert (src_of >= 0).all(), "input layout misses a depthwise anchor voxel"

    out_slots = _slot_indices(out_layout, 4)
    dst = np.flatnonzero(_valid(out_slots, out_shape).reshape(-1))
    out_idx = out_slots.reshape(-1, 4)[dst]
    src = src_of[np.ravel_multi_index(tuple(out_idx.T), out_shape)]

    plan = _plan_from_positions(
        out_idx, src, dst, occupied, out_shape, 1, 3, n, out_layout.num_ct()
    )
    _PLAN_CACHE[key] = plan
    return plan


def depthwise_needs_repack(in_layout, in_shape, out_dhw):
    """Whether a depthwise Conv3D on ``in_layout`` repacks its output.

    A plain GROUP_PAD input cannot leave gaps for the dropped voxels, so an
    output smaller than the input gets a new, smaller group (see
    ``gen_depthwise_conv3d``). A stepped group never repacks: its output layouts
    are computed directly (aligned when the steps equal the strides).
    """
    groups = [d for d in in_layout.get_dims() if d.is_group]
    if not groups or any(d.is_stepped for d in groups):
        return False
    return list(out_dhw) != [int(s) for s in in_shape[1:4]]


def depthwise_repack_plan_for_kernel(kernel):
    """Repack plan of a DEPTHWISE_CONV3D kernel, or None if it needs none."""
    from frontends.tensor_args import Conv3dArgs
    from util.shape_util import get_term_shape

    in_layout = kernel.cs[0].layout
    groups = [d for d in in_layout.get_dims() if d.is_group]
    if not groups:
        return None
    # gen_depthwise_conv3d only accepts groups over exactly dims 1..3, so the
    # group holds the logical input [D, H, W]; the filter is [C, Kd, Kh, Kw].
    term = kernel.layout.term
    f_shape = get_term_shape(term.cs[1])
    in_shape = [int(f_shape[0])] + [int(e) for e in groups[0].group]
    out_dhw, _ = Conv3dArgs.depthwise_geometry(
        in_shape[1:], f_shape[1:], term.cs[2], term.cs[3]
    )
    if not depthwise_needs_repack(in_layout, in_shape, out_dhw):
        return None
    return depthwise_repack_plan(in_layout, kernel.layout, in_shape, out_dhw, term.cs[2])


@dataclass(frozen=True)
class GatherMove:
    """``out[dst_ct] += (cts[src_ct] << rot) * mask * w[tap]``."""

    dst_ct: int
    tap: int
    src_ct: int
    rot: int
    mask: np.ndarray


@dataclass(frozen=True)
class DepthwiseGatherPlan:
    """Masked rotations computing a depthwise Conv3D (see ``depthwise_gather_plan``)."""

    moves: tuple
    num_out_ct: int
    num_taps: int

    def ops(self):
        """Ops after common-subexpression elimination: a rotation of the same
        source ciphertext by the same amount is shared across taps."""
        rots = {(m.src_ct, m.rot) for m in self.moves if m.rot != 0}
        per_ct = {}
        for m in self.moves:
            per_ct[m.dst_ct] = per_ct.get(m.dst_ct, 0) + 1
        return {
            "rot": len(rots),
            "mul": 2 * len(self.moves),
            "add": sum(max(k - 1, 0) for k in per_ct.values()),
        }


def depthwise_gather_plan(in_layout, out_layout, in_shape, k_dhw, strides, padding):
    """Masked rotations computing a depthwise Conv3D directly in its slots.

    Slot ``s`` of computing ciphertext ``o`` gets output voxel ``anchor[o, s]``:
    its own voxel in ``out_layout``, or, when a plain group is repacked
    afterwards (``depthwise_needs_repack``), its anchor in ``in_layout`` (see
    ``depthwise_anchor_rows``). Each tap then reads one input slot per valid
    output slot; slots are grouped by (source ciphertext, rotation). An aligned
    layout needs one group per tap and computing ciphertext.
    """
    from frontends.tensor_args import Conv3dArgs

    in_shape = tuple(int(s) for s in in_shape)
    k_dhw = tuple(int(k) for k in k_dhw)
    strides = tuple(int(s) for s in strides)
    key = (
        "depthwise_gather",
        in_layout.layout_str(),
        out_layout.layout_str(),
        in_layout.n,
        in_shape,
        k_dhw,
        strides,
        padding,
    )
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]

    n = int(in_layout.n)
    out_dhw, front = Conv3dArgs.depthwise_geometry(in_shape[1:], k_dhw, strides, padding)
    shape = np.asarray(in_shape, dtype=np.int64)
    out_shape = np.asarray((in_shape[0],) + tuple(out_dhw), dtype=np.int64)
    stride_arr = np.asarray(strides, dtype=np.int64)
    front = np.asarray(front, dtype=np.int64)

    # logical input coordinate -> flat (ct * n + slot), first occurrence
    in_rows = _slot_indices(in_layout, 4).reshape(-1, 4)
    ok = _valid(in_rows, in_shape)
    lookup = np.full(prod(in_shape), -1, dtype=np.int64)
    pos = np.flatnonzero(ok)[::-1]
    lookup[np.ravel_multi_index(in_rows[pos].T, in_shape)] = pos

    if depthwise_needs_repack(in_layout, in_shape, out_dhw):
        anchors = depthwise_anchor_rows(in_layout, in_shape, out_dhw, strides)
    else:
        anchors = _slot_indices(out_layout, 4)

    taps = [
        (i, j, k) for i in range(k_dhw[0]) for j in range(k_dhw[1]) for k in range(k_dhw[2])
    ]
    moves = []
    for o in range(anchors.shape[0]):
        rows = anchors[o]
        valid_out = _valid(rows, out_shape)
        for tap_idx, tap in enumerate(taps):
            src = rows[:, 1:] * stride_arr + np.asarray(tap) - front
            valid = valid_out & ((src >= 0) & (src < shape[1:])).all(axis=1)
            slots = np.flatnonzero(valid)
            if slots.size == 0:
                continue
            coords = np.concatenate([rows[slots, :1], src[slots]], axis=1)
            src_flat = lookup[np.ravel_multi_index(coords.T, in_shape)]
            assert (src_flat >= 0).all(), "input voxel missing from input layout"
            src_ct, src_slot = src_flat // n, src_flat % n
            rot = (src_slot - slots) % n
            uniq, inverse = np.unique(
                np.stack([src_ct, rot], axis=1), axis=0, return_inverse=True
            )
            inverse = inverse.reshape(-1)
            for g, (c, r) in enumerate(uniq):
                mask = np.zeros(n, dtype=np.int32)
                mask[slots[inverse == g]] = 1
                moves.append(GatherMove(o, tap_idx, int(c), int(r), mask))

    plan = DepthwiseGatherPlan(tuple(moves), int(anchors.shape[0]), len(taps))
    _PLAN_CACHE[key] = plan
    return plan


def depthwise_gather_plan_for_kernel(kernel):
    """Gather plan of a DEPTHWISE_CONV3D kernel (its taps, before any repack)."""
    from util.shape_util import get_term_shape

    term = kernel.layout.term
    in_shape = get_term_shape(term.cs[0])
    k_dhw = get_term_shape(term.cs[1])[1:]
    return depthwise_gather_plan(
        kernel.cs[0].layout, kernel.layout, in_shape, k_dhw, term.cs[2], term.cs[3]
    )


def flatten_repack_plan_for_kernel(kernel):
    """Plan for a FLATTEN_REPACK kernel (``cs = [input, logical_input_shape]``)."""
    from frontends.tensor_args import FlattenArgs

    args = FlattenArgs.from_term(kernel.layout.term)
    return flatten_repack_plan(
        kernel.cs[0].layout, kernel.layout, kernel.cs[1], args.start, args.end
    )


def group_pad_plan_for_kernel(kernel):
    """Plan for a GROUP_PAD kernel (``cs = [input, logical_input_shape]``)."""
    from frontends.tensor_args import GroupPadArgs

    args = GroupPadArgs.from_term(kernel.layout.term)
    return group_pad_plan(
        kernel.cs[0].layout, kernel.layout, kernel.cs[1], args.start, args.end
    )


def repack_plan_for_kernel(kernel):
    """Plan for a FLATTEN_REPACK or GROUP_PAD kernel."""
    from ir.kernel import KernelOp

    if kernel.op == KernelOp.GROUP_PAD:
        return group_pad_plan_for_kernel(kernel)
    return flatten_repack_plan_for_kernel(kernel)
