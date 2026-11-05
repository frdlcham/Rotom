import numpy as np

from frontends.tensor_args import Conv3dArgs
from ir.analysis.shape import Shape
from ir.dim import DimType
from ir.he import HEOp, HETerm
from ir.layout import Layout
from ir.layout_utils import is_residue_dim, stepped_axes
from lower.layout_cts import LayoutCiphertexts, create_layout_without_dims
from lower.lower_flatten import apply_repack_plan
from lower.lower_util import find_sum_dim, rotate_and_sum
from util.flatten_repack import (
    depthwise_gather_plan,
    depthwise_needs_repack,
    depthwise_repack_plan,
)
from util.layout_util import (
    _get_apply_layout_plan,
    convert_layout_to_mask,
    get_cts_by_dim,
    get_dim_indices,
    get_dim_map,
    get_segment,
)
from util.shape_util import get_term_shape


def _slot_mask_conv3d_same_stride1(
    *,
    base_indices_ct,
    rot_amt: int,
    fd: int,
    fh: int,
    fw: int,
    pad_front: int,
    pad_top: int,
    pad_left: int,
    din: int,
    hin: int,
    win: int,
    n: int,
) -> list[int]:
    """Per-slot mask for ``same`` conv3d, stride 1 (mirrors ``_slot_mask_conv2d_same_stride1``).

    Toy rotation: ``out[s] = vec[(s + rot_amt) % n]``. Zero slots where the rotated
    source would read an out-of-bounds neighbor or map outside the output box.
    """
    if (
        isinstance(base_indices_ct, np.ndarray)
        and base_indices_ct.ndim == 2
        and base_indices_ct.shape[0] == n
        and base_indices_ct.shape[1] >= 4
        and base_indices_ct.dtype == np.int32
    ):
        idx = np.arange(n, dtype=np.int64)
        src = (idx + int(rot_amt)) % n
        rows = base_indices_ct[src].astype(np.int64, copy=False)
        c0, du, hu, wu = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        bad = (c0 < 0) | (du < 0) | (hu < 0) | (wu < 0)
        bad |= (du >= din) | (hu >= hin) | (wu >= win)
        o_d = du + int(pad_front) - int(fd)
        o_h = hu + int(pad_top) - int(fh)
        o_w = wu + int(pad_left) - int(fw)
        bad |= (
            (o_d < 0)
            | (o_d >= din)
            | (o_h < 0)
            | (o_h >= hin)
            | (o_w < 0)
            | (o_w >= win)
        )
        return np.logical_not(bad).astype(np.int32).tolist()

    mask: list[int] = [1] * n
    for s in range(n):
        src = (s + int(rot_amt)) % n
        if src < 0 or src >= len(base_indices_ct):
            mask[s] = 0
            continue
        idx = base_indices_ct[src]
        if isinstance(idx, np.ndarray):
            if idx.shape[0] < 4:
                mask[s] = 0
                continue
            if int(idx[0]) < 0 or int(idx[1]) < 0 or int(idx[2]) < 0 or int(idx[3]) < 0:
                mask[s] = 0
                continue
            du, hu, wu = int(idx[1]), int(idx[2]), int(idx[3])
        else:
            if not isinstance(idx, (list, tuple)) or len(idx) < 4:
                mask[s] = 0
                continue
            if any(idx[i] is None for i in range(4)):
                mask[s] = 0
                continue
            du, hu, wu = int(idx[1]), int(idx[2]), int(idx[3])
        if du < 0 or du >= din or hu < 0 or hu >= hin or wu < 0 or wu >= win:
            mask[s] = 0
            continue
        o_d = du + pad_front - fd
        o_h = hu + pad_top - fh
        o_w = wu + pad_left - fw
        if o_d < 0 or o_d >= din or o_h < 0 or o_h >= hin or o_w < 0 or o_w >= win:
            mask[s] = 0
        else:
            mask[s] = 1
    return mask


def _replica_ids(layout) -> np.ndarray:
    """(num_ct, n) id of the copy each slot belongs to.

    Only replication dims (no tensor dim) count: two slots holding the same
    input voxel in the same copy have the same id.
    """

    def digits(dims, idx):
        rep = np.zeros_like(idx)
        mult = inner = 1
        for d in reversed(dims):
            digit = (idx // inner) % d.extent
            inner *= d.extent
            if d.dim is None and d.dim_type == DimType.FILL:
                rep = rep + digit * mult
                mult *= d.extent
        return rep, mult

    n = int(layout.n)
    ct_rep, _ = digits(layout.ct_dims, np.arange(layout.num_ct(), dtype=np.int64))
    slot_total = int(np.prod([d.extent for d in layout.slot_dims], dtype=np.int64))
    slot_idx = np.arange(n, dtype=np.int64) % max(slot_total, 1)
    slot_rep, slot_mult = digits(layout.slot_dims, slot_idx)
    return ct_rep[:, None] * slot_mult + slot_rep[None, :]


class _GroupedConv3dGather:
    """Filter taps of a CONV3D on a GROUP_PAD input, as masked rotations.

    Each slot of input ciphertext ``a_idx`` computes the partial product for its
    own (anchor) voxel ``v``; tap ``f`` needs voxel ``v + f - pad`` of the same
    channel and copy. Rotating along a (possibly stepped) group, that voxel may
    sit in another ciphertext (a different residue class) or at a non-uniform
    offset, so every slot looks it up and slots are grouped by (source
    ciphertext, rotation). A plain group in the slots needs one rotation per tap;
    a stepped group whose residue dims index ciphertexts needs one per
    (tap, residue class) pair.
    """

    def __init__(self, layout_a, in_shape, stride):
        self.n = int(layout_a.n)
        self.in_shape = np.asarray(in_shape, dtype=np.int64)
        self.stride = int(stride)
        rows = _logical_slot_coords(layout_a)
        self.rows = rows
        self.rep = _replica_ids(layout_a)
        self.volume = int(np.prod(self.in_shape))

        flat = rows.reshape(-1, 4)
        ok = np.all((flat >= 0) & (flat < self.in_shape), axis=1)
        pos = np.flatnonzero(ok)
        keys = self.rep.reshape(-1)[pos] * self.volume + np.ravel_multi_index(
            flat[pos].T, self.in_shape
        )
        order = np.argsort(keys, kind="stable")
        keys, pos = keys[order], pos[order]
        _, first = np.unique(keys, return_index=True)  # first occurrence wins
        self.keys, self.pos = keys[first], pos[first]

    def _find(self, query):
        i = np.searchsorted(self.keys, query)
        i = np.minimum(i, len(self.keys) - 1)
        found = self.keys[i] == query
        assert found.all(), "input voxel missing from conv3d input layout"
        return self.pos[i]

    def groups(self, a_idx, offsets):
        """(union mask, [(src_ct, rot, mask)]) for one tap on ciphertext ``a_idx``."""
        rows = self.rows[a_idx]
        spatial = rows[:, 1:4]
        ok = np.all((rows >= 0) & (rows < self.in_shape), axis=1)
        if self.stride != 1:
            # only anchors o * stride compute an output voxel
            ok &= np.all(spatial % self.stride == 0, axis=1)
        want = spatial + np.asarray(offsets, dtype=np.int64)
        ok &= np.all((want >= 0) & (want < self.in_shape[1:]), axis=1)
        slots = np.flatnonzero(ok)
        if slots.size == 0:
            return np.zeros(self.n, dtype=np.int32), []
        coords = np.concatenate([rows[slots, :1], want[slots]], axis=1)
        query = self.rep[a_idx, slots] * self.volume + np.ravel_multi_index(
            coords.T, self.in_shape
        )
        src = self._find(query)
        src_ct, src_slot = src // self.n, src % self.n
        # ``ct << r`` moves slot s + r to slot s
        rot = (src_slot - slots) % self.n
        keys = np.stack([src_ct, rot], axis=1)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        out = []
        for g, (c, r) in enumerate(uniq):
            m = np.zeros(self.n, dtype=np.int32)
            m[slots[inverse.reshape(-1) == g]] = 1
            out.append((int(c), int(r), m))
        return ok.astype(np.int32), out


def _keep_first_residue_class(layout_a, b_layout_cts):
    """Keep only the ciphertexts of residue class 0 of a stepped group.

    A strided conv on a stepped group computes its outputs at the anchors
    ``o * stride``, which all sit in residue class 0; ciphertexts indexed by a
    nonzero residue hold no output. Residue dims are matched by position (the
    filter layout is aligned dim by dim with ``layout_a``), because their
    filter counterparts ``[R:2:1]`` compare equal by value.

    Returns:
        (LayoutCiphertexts without the residue ct dims, positions of the dropped
        dims in ``layout_a.get_dims()``)
    """
    axes = stepped_axes(layout_a.get_dims())
    residue = [
        i for i, d in enumerate(layout_a.ct_dims) if is_residue_dim(d, axes)
    ]
    if not residue:
        return b_layout_cts, set()
    b_layout = b_layout_cts.layout
    assert len(b_layout.ct_dims) == len(layout_a.ct_dims)
    ct_indices = get_dim_indices(b_layout.ct_dims)
    keep = [
        ct
        for ct in range(b_layout.num_ct())
        if all(ct_indices[i][ct] == 0 for i in residue)
    ]
    new_ct_dims = [d for i, d in enumerate(b_layout.ct_dims) if i not in residue]
    new_layout = Layout(
        b_layout.term,
        [],
        new_ct_dims + list(b_layout.slot_dims),
        b_layout.n,
        b_layout.secret,
    )
    cts = {j: b_layout_cts.cts[ct] for j, ct in enumerate(keep)}
    return LayoutCiphertexts(layout=new_layout, cts=cts), set(residue)


def lower_conv3d(env, kernel):
    """Lower CONV3D kernel to circuit IR (secret input, public filter)."""
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    assert a_cts.keys() == b_cts.keys()

    shape = Shape(kernel.layout.term)
    shape.run()

    a_cs = [
        HETerm(HEOp.CS, [a_cts[ct_idx]], a_cts[ct_idx].secret)
        for ct_idx in sorted(a_cts.keys())
    ]
    b_cs = [
        HETerm(HEOp.CS, [b_cts[ct_idx]], b_cts[ct_idx].secret)
        for ct_idx in sorted(b_cts.keys())
    ]

    b_shape = get_term_shape(kernel.cs[1].layout.term)
    # Padding: [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]
    pad_front = kernel.layout.term.cs[4][0]
    _pad_back = kernel.layout.term.cs[4][1]
    pad_top = kernel.layout.term.cs[4][2]
    _pad_bottom = kernel.layout.term.cs[4][3]
    pad_left = kernel.layout.term.cs[4][4]
    _pad_right = kernel.layout.term.cs[4][5]

    k_d, k_h, k_w = b_shape[2], b_shape[3], b_shape[4]
    _stride = kernel.layout.term.cs[2]
    _a_shape = get_term_shape(kernel.cs[0].layout.term)

    layout_a = kernel.cs[0].layout
    grouped = any(dim.is_group for dim in layout_a.get_dims())
    gather = None
    rot_amts = []
    if grouped:
        # GROUP_PAD input: taps are found per slot (see _GroupedConv3dGather)
        gather = _GroupedConv3dGather(
            layout_a, [int(e) for e in _a_shape[:4]], int(_stride)
        )
    else:
        spatial = {}
        offset = 1
        for dim in layout_a.get_dims()[::-1]:
            if dim.dim in (1, 2, 3) and dim.dim_type != DimType.EMPTY:
                spatial[dim.dim] = (dim.extent, offset)
            offset *= dim.extent
        assert 1 in spatial and 2 in spatial and 3 in spatial
        d_extent, d_stride = spatial[1]
        h_extent, h_stride = spatial[2]
        w_extent, w_stride = spatial[3]

        rot_offset = -pad_front * d_stride - pad_top * h_stride - pad_left * w_stride
        for i in range(k_d):
            for j in range(k_h):
                for k in range(k_w):
                    rot_amts.append(
                        i * d_stride + j * h_stride + k * w_stride + rot_offset
                    )

    filter_masks: list[list[int]] = []
    base_by_ct: list | None = None
    din = hin = win = 0
    if grouped:
        pass
    elif int(_stride) != 1:
        for i in range(k_d):
            for j in range(k_h):
                for k in range(k_w):
                    rot_d = i * d_stride
                    rot_h = j * h_stride
                    rot_w = k * w_stride
                    mask = [1] * kernel.cs[0].layout.n

                    segment_d = d_extent * d_stride
                    for loop in range(kernel.cs[0].layout.n // segment_d):
                        for idx in range(segment_d):
                            if (
                                not 0
                                <= idx + rot_d - (pad_front * d_stride)
                                < segment_d
                            ):
                                mask[loop * segment_d + idx] = 0

                    segment_h = h_extent * h_stride
                    for loop in range(kernel.cs[0].layout.n // segment_h):
                        for idx in range(segment_h):
                            if not 0 <= idx + rot_h - (pad_top * h_stride) < segment_h:
                                mask[loop * segment_h + idx] = 0

                    segment_w = w_extent * w_stride
                    for loop in range(kernel.cs[0].layout.n // segment_w):
                        for idx in range(segment_w):
                            if not 0 <= idx + rot_w - (pad_left * w_stride) < segment_w:
                                mask[loop * segment_w + idx] = 0

                    filter_masks.append(mask)
    else:
        layout_a = kernel.cs[0].layout
        layout_len = max(len(layout_a), layout_a.n)
        plan = _get_apply_layout_plan(layout_a, 4, layout_len=layout_len)
        base_by_ct = plan["base_indices_by_cts"]
        din, hin, win = int(_a_shape[1]), int(_a_shape[2]), int(_a_shape[3])

    b_num_cts = kernel.cs[1].layout.num_ct()
    b_ct_dims = kernel.cs[1].layout.ct_dims
    dim_indices_masks = get_dim_indices(b_ct_dims)
    dim_map_masks = get_dim_map(b_ct_dims)
    dim_to_pos_masks = {
        dim.dim: dim_map_masks[dim] for dim in b_ct_dims if dim.dim is not None
    }
    flat_w = k_w
    flat_hw = k_h * k_w
    masks = []
    grouped_taps = []
    for ct_idx in range(b_num_cts):
        f_d = (
            dim_indices_masks[dim_to_pos_masks[2]][ct_idx]
            if 2 in dim_to_pos_masks
            else 0
        )
        f_h = (
            dim_indices_masks[dim_to_pos_masks[3]][ct_idx]
            if 3 in dim_to_pos_masks
            else 0
        )
        f_w = (
            dim_indices_masks[dim_to_pos_masks[4]][ct_idx]
            if 4 in dim_to_pos_masks
            else 0
        )
        filter_idx = f_d * flat_hw + f_h * flat_w + f_w
        a_idx = min(ct_idx, len(a_cs) - 1)
        if grouped:
            offsets = (
                int(f_d) - int(pad_front),
                int(f_h) - int(pad_top),
                int(f_w) - int(pad_left),
            )
            m, tap_groups = gather.groups(a_idx, offsets)
            grouped_taps.append(tap_groups)
            masks.append(m.tolist())
            continue
        rot_amt = rot_amts[filter_idx]
        if int(_stride) == 1:
            assert base_by_ct is not None
            m = _slot_mask_conv3d_same_stride1(
                base_indices_ct=base_by_ct[a_idx],
                rot_amt=rot_amt,
                fd=int(f_d),
                fh=int(f_h),
                fw=int(f_w),
                pad_front=int(pad_front),
                pad_top=int(pad_top),
                pad_left=int(pad_left),
                din=din,
                hin=hin,
                win=win,
                n=kernel.cs[0].layout.n,
            )
        else:
            m = filter_masks[filter_idx]
        masks.append(m)

    if len(kernel.cs[1].layout.term.cs) <= 4:
        kernel.cs[1].layout.term.cs.append(masks)
    if len(kernel.cs[1].layout.term.cs) <= 5:
        kernel.cs[1].layout.term.cs.append(rot_amts)

    _mask_term_by_key: dict[bytes, HETerm] = {}
    mask_terms: list[HETerm] = []
    for m in masks:
        k = bytes(m)
        t = _mask_term_by_key.get(k)
        if t is None:
            t = HETerm.mask([m])
            _mask_term_by_key[k] = t
        mask_terms.append(t)

    dim_indices = get_dim_indices(b_ct_dims)
    dim_map = get_dim_map(b_ct_dims)
    dim_to_pos = {dim.dim: dim_map[dim] for dim in b_ct_dims if dim.dim is not None}

    cts = {}
    for b_idx, b_ct in enumerate(b_cs):
        if grouped:
            parts = []
            for c, r, m in grouped_taps[b_idx]:
                key = m.tobytes()
                if key not in _mask_term_by_key:
                    _mask_term_by_key[key] = HETerm.mask([m.tolist()])
                moved = a_cs[c] if r == 0 else a_cs[c] << r
                parts.append(moved * _mask_term_by_key[key])
            cts[b_idx] = (
                _tree_sum(parts) * b_ct
                if parts
                else HETerm(HEOp.ZERO_MASK, [], False)
            )
            continue
        f_d = dim_indices[dim_to_pos[2]][b_idx] if 2 in dim_to_pos else 0
        f_h = dim_indices[dim_to_pos[3]][b_idx] if 3 in dim_to_pos else 0
        f_w = dim_indices[dim_to_pos[4]][b_idx] if 4 in dim_to_pos else 0
        filter_idx = f_d * flat_hw + f_h * flat_w + f_w
        rot_amt = rot_amts[filter_idx]
        a_idx = min(b_idx, len(a_cs) - 1)
        a_rot = a_cs[a_idx] << rot_amt
        a_masked = a_rot * mask_terms[b_idx]
        cts[b_idx] = a_masked * b_ct

    b_layout_cts = LayoutCiphertexts(layout=kernel.cs[1].layout, cts=cts)
    dropped = set()
    if grouped and int(_stride) != 1:
        b_layout_cts, dropped = _keep_first_residue_class(
            layout_a, b_layout_cts
        )
    filter_ct_dims = [
        dim
        for dim in b_layout_cts.layout.ct_dims
        if (dim.dim is not None and dim.dim != 0) and dim.dim_type == DimType.FILL
    ]

    filter_a_dims = []
    for i, (a_dim, b_dim) in enumerate(
        zip(kernel.cs[0].layout.get_dims(), kernel.cs[1].layout.get_dims())
    ):
        if i not in dropped and b_dim not in filter_ct_dims:
            filter_a_dims.append(a_dim)

    for ct_sum_dim in filter_ct_dims:
        if ct_sum_dim not in b_layout_cts.layout.ct_dims:
            continue
        ct_groups = get_cts_by_dim(b_layout_cts, ct_sum_dim)
        sum_cts = {}
        for i, group in enumerate(ct_groups):
            base = group[0]
            for j in range(1, len(group)):
                base = base + group[j]
            sum_cts[i] = base
        new_layout = create_layout_without_dims(b_layout_cts.layout, [ct_sum_dim])
        b_layout_cts = LayoutCiphertexts(layout=new_layout, cts=sum_cts)

    a_layout = Layout(
        kernel.cs[0].layout.term,
        [],
        filter_a_dims,
        kernel.cs[0].layout.n,
        kernel.cs[0].layout.secret,
    )
    a_layout_cts = LayoutCiphertexts(layout=a_layout, cts=b_layout_cts.cts)

    while True:
        _, slot_sum_dims = find_sum_dim(a_layout_cts.layout, 0)
        if not slot_sum_dims:
            break
        slot_dims_cur = a_layout_cts.layout.slot_dims
        slot_sum_dim = min(
            slot_sum_dims,
            key=lambda d: get_segment(d, slot_dims_cur)[2],
        )
        segment = get_segment(slot_sum_dim, slot_dims_cur)
        extent = segment[1]
        mul_offset = segment[2]
        summed_cts = {}
        for index, term in a_layout_cts.cts.items():
            summed_cts[index] = rotate_and_sum(term, extent, mul_offset)
        new_layout = create_layout_without_dims(a_layout_cts.layout, [slot_sum_dim])
        a_layout_cts = LayoutCiphertexts(layout=new_layout, cts=summed_cts)

    layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=a_layout_cts.cts)

    # Valid: zero slots outside the logical output box (d>=D_out, h>=H_out, w>=W_out).
    logical_out_shape = shape.get_shape(kernel.layout.term)
    if kernel.layout.term.cs[3] == "valid" and logical_out_shape is not None:
        _c_o, d_o, h_o, w_o = logical_out_shape
        slot_dims = kernel.layout.slot_dims
        slot_dim_indices = get_dim_indices(slot_dims)
        dim_to_indices = {}
        for dim, indices in zip(slot_dims, slot_dim_indices):
            if dim.dim_type == DimType.EMPTY:
                continue
            if dim.dim in (1, 2, 3):
                dim_to_indices[dim.dim] = indices

        if 1 in dim_to_indices and 2 in dim_to_indices and 3 in dim_to_indices:
            mask = [1] * kernel.layout.n
            d_idx = dim_to_indices[1]
            h_idx = dim_to_indices[2]
            w_idx = dim_to_indices[3]
            for i in range(kernel.layout.n):
                if d_idx[i] >= d_o or h_idx[i] >= h_o or w_idx[i] >= w_o:
                    mask[i] = 0
            mask_term = HETerm.mask([mask])
            masked_cts = {}
            for index, term in layout_cts.cts.items():
                masked_cts[index] = term * mask_term
            layout_cts = LayoutCiphertexts(layout=layout_cts.layout, cts=masked_cts)

    needs_mask = False
    for dim in kernel.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm.mask([convert_layout_to_mask(kernel.layout)])
        masked_cts = {}
        for index, term in layout_cts.cts.items():
            masked_cts[index] = term * mask
        layout_cts = LayoutCiphertexts(layout=layout_cts.layout, cts=masked_cts)

    return layout_cts


def _logical_slot_coords(layout, ndims: int = 4) -> np.ndarray:
    """(num_ct, n, ndims) logical tensor coordinate held by each slot (-1 = none)."""
    plan = _get_apply_layout_plan(
        layout, ndims, layout_len=max(len(layout), layout.n)
    )
    rows = plan["base_indices_by_cts"]
    if rows.shape[2] < ndims:
        pad = np.zeros(rows.shape[:2] + (ndims - rows.shape[2],), dtype=rows.dtype)
        rows = np.concatenate([rows, pad], axis=2)
    return rows.astype(np.int64)


def _tree_sum(terms):
    """Sum as a balanced tree: a long gather stays shallow in the circuit."""
    terms = list(terms)
    while len(terms) > 1:
        terms = [
            terms[i] + terms[i + 1] if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    return terms[0]


def lower_depthwise_conv3d(env, kernel):
    """Lower DEPTHWISE_CONV3D: out = sum_taps rot(in, r) * mask * w_tap.

    The masked rotations come from ``util.flatten_repack.depthwise_gather_plan``
    (shared with the cost model): for each output ciphertext and filter tap,
    every valid output slot reads one input slot, and slots are grouped by
    (source ct, rotation), so a dense or aligned layout costs one rotation per
    tap per ciphertext. ``w_tap`` is the packed public filter (see
    ``depthwise_conv3d_filter_layout``); masks zero padding and out-of-range
    slots.

    A strided plain GROUP_PAD input cannot keep its outputs in place (see
    ``gen_depthwise_conv3d``): the taps are then computed in the input layout, at
    each output voxel's anchor slot, and the results repacked into the output
    group (see ``util.flatten_repack.depthwise_repack_plan``). A stepped group is
    computed directly in the output layout.
    """
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    in_layout = kernel.cs[0].layout
    out_layout = kernel.layout
    term = out_layout.term
    n = int(out_layout.n)

    in_shape = [int(s) for s in get_term_shape(term.cs[0])]
    k_dhw = [int(k) for k in get_term_shape(term.cs[1])[1:]]
    out_dhw, _ = Conv3dArgs.depthwise_geometry(
        in_shape[1:], k_dhw, term.cs[2], term.cs[3]
    )
    plan = depthwise_gather_plan(
        in_layout, out_layout, in_shape, k_dhw, term.cs[2], term.cs[3]
    )

    mask_terms: dict[bytes, HETerm] = {}
    parts: dict[int, list] = {o: [] for o in range(plan.num_out_ct)}
    for m in plan.moves:
        key = m.mask.tobytes()
        if key not in mask_terms:
            mask_terms[key] = HETerm.mask([m.mask.tolist()])
        moved = a_cts[m.src_ct] if m.rot == 0 else a_cts[m.src_ct] << m.rot
        w = b_cts[m.tap * plan.num_out_ct + m.dst_ct]
        parts[m.dst_ct].append(moved * mask_terms[key] * w)
    cts = {
        o: _tree_sum(ts) if ts else HETerm(HEOp.ZERO_MASK, [], False)
        for o, ts in parts.items()
    }
    if depthwise_needs_repack(in_layout, in_shape, out_dhw):
        repack = depthwise_repack_plan(
            in_layout, out_layout, in_shape, out_dhw, term.cs[2]
        )
        cts = apply_repack_plan(cts, repack, n)
    return LayoutCiphertexts(layout=out_layout, cts=cts)
