"""
Group pad layout generation utilities.

GROUP_PAD keeps a tensor's shape and values but changes how dims
``start..end`` are laid out: instead of padding each of them to a power of 2
(e.g. [C, 5, 14, 14] stored as [C, 8, 16, 16], 2048 slots per channel), they
are laid out row-major in a single group dim that is padded as a whole
(5*14*14 = 980 -> 1024 slots per channel), see ``Dim.group_dim``.

With per-dim ``steps`` the group walks a strided sub-grid and the offsets the
steps skip get their own (outermost) dims, e.g. steps (1, 2, 2) give
``[2:2:1][3:2:1];[C][1-3@5x14x14:256:(1,2,2)]``: one ciphertext per (H, W)
parity class. That is the packing in which a stride-2 conv only needs uniform
rotations.

With ``order`` the group lays the grouped dims out in that order instead,
e.g. order (2, 1, 3) gives ``[(2,1,3)@14x5x14:1024:1]`` (H outermost, W
innermost). Steps and residue dims work the same way.

The elements are moved into the group dim by the staged repack of
``util.flatten_repack``.

Key functions:
- gen_group_pad: Main function for generating group pad layouts
"""

from copy import deepcopy as copy

from frontends.tensor_args import GroupPadArgs
from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging, merge_dims
from util.flatten_repack import group_pad_move_count
from util.util import prod, round_to_ceiling_power_of_2

# A candidate whose repack needs more than this many times the moves of the
# cheapest candidate is dropped before its plan (one n-slot mask per move) is
# built: misaligned input orders need about one move per element.
MAX_MOVE_RATIO = 4


def _in_range(dim, start, end):
    """Whether ``dim`` holds only tensor dims ``start..end`` (a group counts too)."""
    if dim.is_group:
        return all(start <= d <= end for d in dim.dim)
    return isinstance(dim.dim, int) and start <= dim.dim <= end


def _replace_grouped(dims, group, start, end):
    """Drop gaps, put ``group`` where the first piece of dims ``start..end`` was."""
    new_dims = []
    placed = False
    for dim in dims:
        if dim.dim_type == DimType.EMPTY:
            continue
        if _in_range(dim, start, end):
            if not placed:
                new_dims.append(group)
                placed = True
            continue
        new_dims.append(copy(dim))
    if not placed:
        new_dims.append(group)
    return new_dims


def _residue_dims(group):
    """One dim per stepped axis of ``group``, holding the offsets its step skips."""
    if not group.is_group:
        return []
    return [Dim(d, s, 1) for d, s in zip(group.dim, group.steps) if s > 1]


def _in_place_layout(term, layout, group, start, end):
    """Keep the ciphertext dims; compress the slot dims, leaving a leading gap.

    The residue dims of a stepped group become extra (innermost) ciphertext dims.
    """
    if group.extent > layout.n or any(_in_range(d, start, end) for d in layout.ct_dims):
        return None
    slot_dims = _replace_grouped(layout.slot_dims, group, start, end)
    gap = layout.n // int(prod(d.extent for d in slot_dims))
    dims = copy(layout.ct_dims) + _residue_dims(group)
    if gap > 1:
        dims.append(Dim(None, gap, 1, DimType.EMPTY))
    return Layout(term, [], dims + slot_dims, layout.n, layout.secret)


def _compact_layout(term, layout, group, start, end):
    """Re-split all dims over ciphertexts and slots, filling the freed slots.

    The residue dims of a stepped group go first, i.e. outermost.
    """
    dims = _residue_dims(group) + _replace_grouped(layout.get_dims(), group, start, end)
    dims = merge_dims(dims, set())
    return Layout(term, [], dims, layout.n, layout.secret)


def gen_group_pad(term, cs_kernels, logical_shape):
    """Generates layouts for group pad operations.

    Args:
        term: TensorTerm representing the group pad operation
        cs_kernels: List of input kernels to generate group pad layouts for
        logical_shape: Unpadded shape of the group pad input

    Returns:
        List of Kernel objects representing group pad layouts
    """
    args = GroupPadArgs.from_term(term)
    start, end = args.start, args.end
    assert end < len(logical_shape), (
        f"GROUP_PAD dims {start}..{end} out of range for shape {logical_shape}"
    )

    group_shape = [int(s) for s in logical_shape[start : end + 1]]
    steps = args.steps
    assert len(steps) == len(group_shape), (
        f"GROUP_PAD needs one step per grouped dim, got {steps} for {group_shape}"
    )
    # the group walks ceil(extent / step) positions per dim
    positions = prod(-(-e // s) for e, s in zip(group_shape, steps))
    extent = args.extent or round_to_ceiling_power_of_2(positions)
    assert extent >= positions and extent & (extent - 1) == 0, (
        f"GROUP_PAD extent {extent} must be a power of 2 >= {positions}"
    )
    group = Dim.group_dim(start, group_shape, extent, steps=steps, order=args.order)

    kernels = {}
    for cs_kernel in cs_kernels:
        layout = cs_kernel.layout
        # rolls and existing groups are not supported by the repack yet
        if layout.rolls or any(d.is_group for d in layout.get_dims()):
            continue
        cs_placeholder = Kernel(KernelOp.CS, [0], layout)
        for make_layout in (_in_place_layout, _compact_layout):
            new_layout = make_layout(term, layout, group, start, end)
            if new_layout is None:
                continue
            new_layout = dimension_merging(new_layout)
            kernels[(layout.layout_str(), new_layout.layout_str())] = Kernel(
                KernelOp.GROUP_PAD,
                [cs_placeholder, tuple(logical_shape)],
                new_layout,
            )
    return _drop_misaligned(list(kernels.values()), logical_shape, start, end)


def _drop_misaligned(kernels, logical_shape, start, end):
    """Keep the kernels within ``MAX_MOVE_RATIO`` of the cheapest repack."""
    if len(kernels) <= 1:
        return kernels
    counts = [
        group_pad_move_count(
            k.cs[0].layout, k.layout, tuple(logical_shape), start, end
        )
        for k in kernels
    ]
    limit = MAX_MOVE_RATIO * max(min(counts), 1)
    return [k for k, c in zip(kernels, counts) if c <= limit]
