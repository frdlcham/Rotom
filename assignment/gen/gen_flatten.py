"""
Flatten operation layout generation utilities.

This module provides functions for generating layouts for flatten operations,
which merge consecutive tensor dimensions ``start..end`` into one dimension.
Shape analysis pads the merged dimension as a whole (group, then pad), e.g.
[C, 5, 14, 14] -> [C, 980] -> padded [C, 1024].

If every merged dimension after ``start`` is laid out with its logical extent
(no padding), the row-major flatten index of the logical tensor coincides with
the existing slot positions and the flatten is a zero-copy relabel of the
layout dims (FLATTEN).

If a merged dim carries padding (e.g. 14 stored as 16), the first
``logical_shape[k]`` entries of each padded dim hold the data and the rest are
zero 'gaps'. The logical elements are then repacked into a contiguous
flattened dim (FLATTEN_REPACK), see ``util.flatten_repack``.

Key functions:
- gen_flatten: Main function for generating flatten operation layouts
"""

from copy import deepcopy as copy

from frontends.tensor_args import FlattenArgs
from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging
from util.util import prod, round_to_ceiling_power_of_2


def _layout_extents(dims):
    """Total extent each tensor dim occupies in a layout (max stride * extent over its pieces)."""
    extents = {}
    for dim in dims:
        if dim.dim is None:
            continue
        extents[dim.dim] = max(extents.get(dim.dim, 1), dim.stride * dim.extent)
    return extents


def _gen_flatten_repack(term, cs_kernels, logical_shape, start, end):
    """Flatten over padded dims by repacking the logical elements into slots.

    The output layout keeps the input's dim order, with all pieces of the
    merged dims replaced by one contiguous ``[start:P:1]`` dim (P = merged
    logical size rounded up to a power of 2) at the position of the first piece.
    """
    flat_extent = round_to_ceiling_power_of_2(prod(logical_shape[start : end + 1]))

    kernels = []
    for cs_kernel in cs_kernels:
        new_dims = []
        placed = False
        for dim in cs_kernel.layout.get_dims():
            if dim.dim is not None and start <= dim.dim <= end:
                if not placed:
                    new_dims.append(Dim(start, flat_extent, 1))
                    placed = True
                continue
            dim = copy(dim)
            if dim.dim is not None and dim.dim > end:
                dim.dim -= end - start
            new_dims.append(dim)

        cs_placeholder = Kernel(KernelOp.CS, [0], cs_kernel.layout)
        new_layout = dimension_merging(
            Layout(
                term,
                [],
                new_dims,
                cs_kernel.layout.n,
                cs_kernel.layout.secret,
            )
        )
        kernels.append(
            Kernel(
                KernelOp.FLATTEN_REPACK,
                [cs_placeholder, tuple(logical_shape)],
                new_layout,
            )
        )
    return kernels


def gen_flatten(term, cs_kernels, logical_shape, padded_shape):
    """Generates layouts for flatten operations.

    Each merged dim ``k`` in ``start..end`` is relabelled to ``start`` with its
    stride scaled by the extent of the dims inside it (``k+1..end``); dims after
    ``end`` shift down by ``end - start``. If any dim in ``start+1..end`` is
    padded, the elements are repacked instead (see ``_gen_flatten_repack``).

    Args:
        term: TensorTerm representing the flatten operation
        cs_kernels: List of input kernels to generate flatten layouts for
        logical_shape: Unpadded shape of the flatten input
        padded_shape: Padded shape of the flatten input

    Returns:
        List of Kernel objects representing flatten operation layouts
    """
    args = FlattenArgs.from_term(term)
    start, end = args.start, args.end
    assert end < len(logical_shape), (
        f"FLATTEN dims {start}..{end} out of range for shape {logical_shape}"
    )

    padded_inner = [
        k
        for k in range(start + 1, end + 1)
        if logical_shape[k] != padded_shape[k]
    ]
    if padded_inner:
        return _gen_flatten_repack(term, cs_kernels, logical_shape, start, end)

    kernels = []
    for cs_kernel in cs_kernels:
        # Rolls reference Dim objects, which we relabel below.
        if cs_kernel.layout.rolls:
            continue

        dims = cs_kernel.layout.get_dims()
        extents = _layout_extents(dims)
        if any(
            extents.get(k, 1) != logical_shape[k] for k in range(start + 1, end + 1)
        ):
            continue

        # inner[k] = number of flattened elements spanned by one step of dim k
        inner = {end: 1}
        for k in range(end - 1, start - 1, -1):
            inner[k] = inner[k + 1] * logical_shape[k + 1]

        new_dims = []
        for dim in dims:
            dim = copy(dim)
            if dim.dim is not None:
                if start <= dim.dim <= end:
                    dim.stride *= inner[dim.dim]
                    dim.dim = start
                elif dim.dim > end:
                    dim.dim -= end - start
            new_dims.append(dim)

        cs_placeholder = Kernel(KernelOp.CS, [0], cs_kernel.layout)
        new_layout = dimension_merging(
            Layout(
                term,
                [],
                new_dims,
                cs_kernel.layout.n,
                cs_kernel.layout.secret,
            )
        )
        kernels.append(Kernel(KernelOp.FLATTEN, [cs_placeholder], new_layout))
    return kernels
