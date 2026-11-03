"""
3D Convolution layout generation utilities (layout assignment only).

This mirrors `gen_conv2d` but for 3D convolutions with shapes:
- input:  [C_in, D_in, H_in, W_in]   (secret)
- filter: [C_out, C_in, Kd, Kh, Kw] (public)
- output: [C_out, D_out, H_out, W_out]
"""

from copy import deepcopy as copy

from assignment.gen.gen_compaction import find_compaction
from assignment.gen.gen_group_pad import _compact_layout, _in_place_layout
from frontends.tensor_args import Conv3dArgs
from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging
from util.flatten_repack import depthwise_needs_repack
from util.util import prod, round_to_ceiling_power_of_2


def apply_replication(term, kernel, dim):
    dims = copy(kernel.layout.get_dims())
    replicated_dims = [dim] + dims
    replicated_layout = Layout(
        term,
        kernel.layout.rolls,
        replicated_dims,
        kernel.layout.n,
        kernel.layout.secret,
    )
    return Kernel(KernelOp.REPLICATE, [kernel], replicated_layout)


def calculate_padding_3d(input_shape, filter_shape, stride, padding):
    # input_shape: [C_in, D, H, W], filter_shape: [C_out, C_in, Kd, Kh, Kw]
    d_in, h_in, w_in = input_shape[1], input_shape[2], input_shape[3]
    k_d, k_h, k_w = filter_shape[2], filter_shape[3], filter_shape[4]
    s = stride
    if padding == "valid":
        return [0, 0, 0, 0, 0, 0]
    if padding != "same":
        raise NotImplementedError("unknown padding: " + padding)

    if s == 1:
        o_d = (d_in + s - 1) // s
        o_h = (h_in + s - 1) // s
        o_w = (w_in + s - 1) // s
        total_pd = max(0, (o_d - 1) * s + k_d - d_in)
        total_ph = max(0, (o_h - 1) * s + k_h - h_in)
        total_pw = max(0, (o_w - 1) * s + k_w - w_in)
        pf = total_pd // 2
        pb = total_pd - pf
        pt = total_ph // 2
        pbot = total_ph - pt
        pl = total_pw // 2
        pr = total_pw - pl
        return [pf, pb, pt, pbot, pl, pr]

    # stride > 1: symmetric k//2 per side (match evaluator convention)
    p_d = k_d // 2
    p_h = k_h // 2
    p_w = k_w // 2
    return [p_d, p_d, p_h, p_h, p_w, p_w]


def add_replicated_dimensions_3d(a_shape, b_shape):
    # Replicate output channels and filter spatial dims onto the secret input layout.
    replicated_dims = []
    b_dims = []
    c_out = b_shape[0]
    c_in = b_shape[1]
    k_d, k_h, k_w = b_shape[2], b_shape[3], b_shape[4]

    # Replicate output channels
    if c_out > 1:
        replicated_dims.append(
            Dim(None, c_out, c_in * a_shape[1] * a_shape[2] * a_shape[3])
        )
        b_dims.append(Dim(0, c_out, 1))

    # Replicate spatial kernel dims (based on INPUT spatial strides)
    if k_d > 1:
        replicated_dims.append(Dim(None, k_d, a_shape[2] * a_shape[3]))
        b_dims.append(Dim(2, k_d, 1))
    if k_h > 1:
        replicated_dims.append(Dim(None, k_h, a_shape[3]))
        b_dims.append(Dim(3, k_h, 1))
    if k_w > 1:
        replicated_dims.append(Dim(None, k_w, 1))
        b_dims.append(Dim(4, k_w, 1))

    return replicated_dims, b_dims


def _materialized_tensor_shape(output_layout):
    shape_map = {}
    for d in output_layout.get_dims():
        if d.dim_type == DimType.EMPTY or d.dim is None:
            continue
        if d.is_group:
            # a group split over ciphertexts and slots repeats the same group
            shape_map.update(zip(d.dim, d.group))
            continue
        shape_map[d.dim] = shape_map.get(d.dim, 1) * d.extent
    if not shape_map:
        return []
    keys_2 = []
    for key in shape_map.keys():
        if type(key) is tuple:
            keys_2.append(max(key))
        else:
            keys_2.append(key)
    return [shape_map.get(i, 1) for i in range(max(keys_2) + 1)]


def _conv3d_output_dims_zip(a_dims, b_dims, d_o_p2, h_o_p2, w_o_p2, stride):
    output_dims = []
    spatial_out_done = {1: False, 2: False, 3: False}
    for a_dim, b_dim in zip(copy(a_dims), copy(b_dims)):
        if a_dim.dim_type == DimType.EMPTY:
            output_dims.append(Dim(None, a_dim.extent, 1, DimType.EMPTY))
        elif a_dim.dim in [0] or b_dim.dim in [1, 2, 3, 4]:
            # Sum dims / filter dims do not appear in output.
            output_dims.append(Dim(None, a_dim.extent, 1, DimType.EMPTY))
        elif a_dim.dim is None and b_dim.dim == 0:
            # output channel
            output_dims.append(b_dim)
        elif a_dim.is_group:
            # grouped spatial dims: stride-1 "same" output keeps the input group
            output_dims.append(copy(a_dim))
        elif (
            a_dim.dim in (1, 2, 3)
            and a_dim.dim_type == DimType.FILL
            and b_dim.dim is None
        ):
            d = a_dim.dim
            if spatial_out_done[d]:
                return None
            spatial_out_done[d] = True
            extent = d_o_p2 if d == 1 else (h_o_p2 if d == 2 else w_o_p2)
            output_dims.append(Dim(d, extent, 1))
            if stride == 2:
                output_dims.append(Dim(None, stride, 1, DimType.EMPTY))
            elif stride != 1:
                raise NotImplementedError("stride not supported: " + str(stride))
        else:
            output_dims.append(Dim(a_dim.dim, a_dim.extent, 1, DimType.FILL))
    return output_dims


def _supports_grouped_input(layout, grouped, b_shape, args):
    """Whether CONV3D can consume a GROUP_PAD layout of its input.

    Only stride-1 "same" convs keep the input group as output layout. Filter taps
    are rotations along the row-major group, so for filters larger than 1x1x1
    the group must sit in the slots.
    """
    if args.stride != 1 or args.padding != "same":
        return False
    if any(not set(d.dim) <= {1, 2, 3} for d in grouped):
        return False
    if any(k > 1 for k in b_shape[2:5]):
        return not any(d.is_group for d in layout.ct_dims)
    return True


def gen_conv3d(term, cs_kernels, shapes):
    a_shape = shapes[0]
    b_shape = shapes[1]
    args = Conv3dArgs.from_term(term)

    padding = calculate_padding_3d(a_shape, b_shape, args.stride, args.padding)
    term.cs.append(padding)
    b_term = args.filter
    b_term.cs.append(padding)

    output_kernels = set()
    for a_kernel in cs_kernels:
        if a_kernel.layout.rolls:
            continue
        grouped = [d for d in a_kernel.layout.get_dims() if d.is_group]
        if grouped and not _supports_grouped_input(
            a_kernel.layout, grouped, b_shape, args
        ):
            continue

        replicated_dims, b_dims = add_replicated_dimensions_3d(a_shape, b_shape)
        for dim in replicated_dims[::-1]:
            a_kernel = apply_replication(args.input, a_kernel, dim)

        # Build b layout to align with a layout geometry.
        for dim in a_kernel.layout.get_dims():
            if dim.dim == 0:
                b_dims.append(Dim(1, dim.extent, dim.stride))
            elif dim.dim is not None:
                b_dims.append(Dim(None, dim.extent, 1))
            elif dim.dim_type == DimType.EMPTY:
                b_dims.append(Dim(None, dim.extent, 1, DimType.EMPTY))

        if len(b_dims) != len(a_kernel.layout.get_dims()):
            continue

        b_layout = Layout(b_term, [], b_dims, a_kernel.layout.n, False)
        b_kernel = Kernel(KernelOp.PUNCTURED_TENSOR, [], b_layout)

        stride = args.stride
        padding_mode = args.padding
        d_i, h_i, w_i = a_shape[1], a_shape[2], a_shape[3]
        k_d, k_h, k_w = b_shape[2], b_shape[3], b_shape[4]
        if padding_mode == "valid":
            d_o = (d_i - k_d) // stride + 1
            h_o = (h_i - k_h) // stride + 1
            w_o = (w_i - k_w) // stride + 1
            # Shape analysis still uses dense [D_out,H_out,W_out] (p2);
            # see Conv3D(valid) in ``assignment.shape_check``.
            d_o_layout, h_o_layout, w_o_layout = d_i, h_i, w_i
        else:
            if stride == 1:
                d_o, h_o, w_o = d_i, h_i, w_i
                d_o_layout, h_o_layout, w_o_layout = d_o, h_o, w_o
            else:
                p_d, p_h, p_w = k_d // 2, k_h // 2, k_w // 2
                d_o = (d_i + 2 * p_d - k_d) // stride + 1
                h_o = (h_i + 2 * p_h - k_h) // stride + 1
                w_o = (w_i + 2 * p_w - k_w) // stride + 1
                d_o_layout, h_o_layout, w_o_layout = d_o, h_o, w_o

        d_o_p2 = round_to_ceiling_power_of_2(d_o_layout)
        h_o_p2 = round_to_ceiling_power_of_2(h_o_layout)
        w_o_p2 = round_to_ceiling_power_of_2(w_o_layout)

        output_dims = _conv3d_output_dims_zip(
            a_kernel.layout.get_dims(), b_dims, d_o_p2, h_o_p2, w_o_p2, stride
        )
        if output_dims is None:
            continue

        output_layout = Layout(
            term, [], output_dims, a_kernel.layout.n, a_kernel.layout.secret
        )
        if grouped:
            # the group pads the spatial dims as a whole, they keep their extents
            expected = [b_shape[0], d_o, h_o, w_o]
        else:
            expected = [b_shape[0], d_o_p2, h_o_p2, w_o_p2]
        if _materialized_tensor_shape(output_layout) != expected:
            continue

        kernel = Kernel(KernelOp.CONV3D, [a_kernel, b_kernel], output_layout)
        output_kernels.add(kernel)
        output_kernels.add(find_compaction(kernel)) # add compacted kernel as well

    return output_kernels


def depthwise_conv3d_filter_layout(b_term, out_layout, k_dhw):
    """Public filter layout for depthwise Conv3D: one ciphertext group per tap.

    Ciphertext index = tap_index * out_layout.num_ct() + out_ct_index, where
    ``tap_index`` enumerates (kd, kh, kw) row-major. Within a ciphertext each slot
    holds the weight of that slot's output channel (spatial dims are replicated).
    """
    tap_dims = [Dim(i + 1, k, 1) for i, k in enumerate(k_dhw) if k > 1]
    dims = []
    for d in out_layout.get_dims():
        if d.dim_type == DimType.EMPTY:
            dims.append(Dim(None, d.extent, 1, DimType.EMPTY))
        elif d.dim == 0:
            dims.append(Dim(0, d.extent, d.stride))
        else:
            dims.append(Dim(None, d.extent, 1))
    layout = Layout(b_term, [], tap_dims + dims, out_layout.n, False)
    # Keep tap ct dims outermost so the ct index formula above holds.
    layout.ct_dims = tap_dims + [d for d in layout.ct_dims if d not in tap_dims]
    return layout


def _axis_extent(layout, axis):
    extent = 1
    for d in layout.get_dims():
        if d.dim == axis:
            extent *= d.extent
    return extent


def _grouped_depthwise_out_layouts(term, layout, out_dhw):
    """Output layouts of a depthwise Conv3D on a GROUP_PAD input.

    The input group must hold exactly dims 1..3. When the output keeps the input
    size (stride 1, "same") it keeps the input layout and every tap is a uniform
    rotation along the group. Otherwise a group cannot leave gaps for the dropped
    voxels: the output gets a new group of the logical output size, either in
    place of the old one or re-split over ciphertexts and slots, and the lowering
    repacks the outputs into it (see ``depthwise_repack_plan``).
    """
    groups = [d for d in layout.get_dims() if d.is_group]
    if any(d.dim != (1, 2, 3) for d in groups):
        return []
    if list(groups[0].group) == list(out_dhw):
        return [Layout(term, [], copy(layout.get_dims()), layout.n, layout.secret)]
    group = Dim.group_dim(1, out_dhw, round_to_ceiling_power_of_2(prod(out_dhw)))
    layouts = {}
    for make_layout in (_in_place_layout, _compact_layout):
        new_layout = make_layout(term, layout, group, 1, 3)
        if new_layout is not None:
            new_layout = dimension_merging(new_layout)
            layouts[new_layout.layout_str()] = new_layout
    return list(layouts.values())


def gen_depthwise_conv3d(term, cs_kernels, shapes):
    """Depthwise Conv3D: output keeps the input packing.

    Each spatial dim ``[d:E]`` becomes ``[d:E_out][G:E/E_out]`` (split axes shrink
    from their innermost part). When ``E/E_out`` equals the stride, output voxel
    ``o`` sits where input voxel ``o * stride`` sat, so every filter tap is a
    single uniform rotation per ciphertext (plus boundary masks). Lowering is
    correct for any resulting layout; alignment only affects rotation count.

    A GROUP_PAD input (one group over dims 1..3) is handled by
    ``_grouped_depthwise_out_layouts``.
    """
    args = Conv3dArgs.from_term(term)
    a_shape, b_shape = shapes[0], shapes[1]
    out_dhw, _ = Conv3dArgs.depthwise_geometry(
        a_shape[1:], b_shape[1:], args.stride, args.padding
    )
    out_p2 = {d + 1: round_to_ceiling_power_of_2(o) for d, o in enumerate(out_dhw)}

    output_kernels = set()
    for a_kernel in cs_kernels:
        layout = a_kernel.layout
        if layout.rolls:
            continue
        if any(d.is_group for d in layout.get_dims()):
            # taps are computed in the input layout when the output is repacked
            repack = depthwise_needs_repack(layout, a_shape, out_dhw)
            for out_layout in _grouped_depthwise_out_layouts(term, layout, out_dhw):
                b_layout = depthwise_conv3d_filter_layout(
                    args.filter, layout if repack else out_layout, b_shape[1:]
                )
                b_kernel = Kernel(KernelOp.TENSOR, [], b_layout)
                output_kernels.add(
                    Kernel(KernelOp.DEPTHWISE_CONV3D, [a_kernel, b_kernel], out_layout)
                )
            continue
        # Shrink each spatial axis to its output extent, innermost part first,
        # leaving a gap of the removed factor after the shrunk part. An axis may
        # be split over several Dims (e.g. across ciphertexts and slots).
        shrink = {
            axis: _axis_extent(layout, axis) // out_p2[axis] for axis in (1, 2, 3)
        }
        new_extent = {}
        for d in reversed(layout.get_dims()):
            if d.dim in (1, 2, 3):
                g = min(shrink[d.dim], d.extent)
                shrink[d.dim] //= g
                new_extent[id(d)] = (d.extent // g, g)
        dims = []
        inner = {}
        for d in reversed(layout.get_dims()):
            if d.dim in (1, 2, 3):
                e, g = new_extent[id(d)]
                if g > 1:
                    dims.append(Dim(None, g, 1, DimType.EMPTY))
                if e > 1:
                    dims.append(Dim(d.dim, e, inner.get(d.dim, 1)))
                    inner[d.dim] = inner.get(d.dim, 1) * e
            else:
                dims.append(copy(d))
        dims.reverse()
        out_layout = Layout(term, [], dims, layout.n, layout.secret)
        b_layout = depthwise_conv3d_filter_layout(args.filter, out_layout, b_shape[1:])
        b_kernel = Kernel(KernelOp.TENSOR, [], b_layout)
        output_kernels.add(
            Kernel(KernelOp.DEPTHWISE_CONV3D, [a_kernel, b_kernel], out_layout)
        )
    return output_kernels
