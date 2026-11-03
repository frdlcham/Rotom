"""
Group pad layout generation utilities.

GROUP_PAD keeps a tensor's shape and values but changes how dims
``start..end`` are laid out: instead of padding each of them to a power of 2
(e.g. [C, 5, 14, 14] stored as [C, 8, 16, 16], 2048 slots per channel), they
are laid out row-major in a single group dim that is padded as a whole
(5*14*14 = 980 -> 1024 slots per channel), see ``Dim.group_dim``.

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
from util.util import prod, round_to_ceiling_power_of_2


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


def _in_place_layout(term, layout, group, start, end):
    """Keep the ciphertext dims; compress the slot dims, leaving a leading gap."""
    if group.extent > layout.n or any(_in_range(d, start, end) for d in layout.ct_dims):
        return None
    slot_dims = _replace_grouped(layout.slot_dims, group, start, end)
    gap = layout.n // int(prod(d.extent for d in slot_dims))
    dims = copy(layout.ct_dims)
    if gap > 1:
        dims.append(Dim(None, gap, 1, DimType.EMPTY))
    return Layout(term, [], dims + slot_dims, layout.n, layout.secret)


def _compact_layout(term, layout, group, start, end):
    """Re-split all dims over ciphertexts and slots, filling the freed slots."""
    dims = merge_dims(_replace_grouped(layout.get_dims(), group, start, end), set())
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
    extent = args.extent or round_to_ceiling_power_of_2(prod(group_shape))
    assert extent >= prod(group_shape) and extent & (extent - 1) == 0, (
        f"GROUP_PAD extent {extent} must be a power of 2 >= {prod(group_shape)}"
    )
    group = Dim.group_dim(start, group_shape, extent)

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
    return list(kernels.values())
