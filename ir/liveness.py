"""
Dead ciphertexts: ciphertexts of a layout that only hold padding.

A tensor whose (logical) extents are not powers of 2 is padded to a power of 2
in its layouts, e.g. 184 channels are laid out as ``[0:8:32];[0:32:1]``: 256
channel positions over 8 ciphertexts, of which the last two (channels 192..255)
only hold padding. Such a ciphertext is *dead* when the padding is known to be
zero: then it is never packed or computed, its ops cost nothing and lowering
emits an explicit ``ZERO_MASK`` for it (which the circuit opts fold away).

Padding is only known to be zero along the *zero-padded dims* of a term (see
``zero_padded_dims``): every dim of an input tensor (``apply_layout`` and the
tensor evaluator fill it with zeros), and the dims along which an op maps zero
padding to zero padding, e.g. the output channels of a conv (zero filter rows).
A polynomial call, for one, maps 0 to its constant term, so its padding is not
zero and its ciphertexts are never dead.

Only the kernel ops in ``LIVENESS_OPS`` use dead ciphertexts; the cost model
(``ir.kernel_cost``), the lowering (``lower.lower``) and the printed layouts
(``Kernel.__repr__``) all go through ``dead_cts`` so they agree.
"""

from math import prod

from frontends.tensor import TensorOp
from frontends.tensor_args import Conv3dArgs
from ir.dim import DimType
from util.layout_util import get_dim_indices

# kernel ops whose dead ciphertexts are skipped (cost, lowering and printing)
LIVENESS_OP_NAMES = (
    "TENSOR",
    "PUNCTURED_TENSOR",
    "ADD",
    "SUB",
    "MUL",
    "CONV3D",
    "DEPTHWISE_CONV3D",
)

# Per-term results are cached on the term object itself: terms compare (and
# hash) by their string, which leaves out the shapes, so equal-looking terms of
# different programs must not share a cache entry.
_LOGICAL_SHAPE_ATTR = "_liveness_logical_shape"
_ZERO_DIMS_ATTR = "_liveness_zero_dims"
_DEAD_CTS_CACHE = {}


def _liveness_ops():
    from ir.kernel import KernelOp

    return {KernelOp[name] for name in LIVENESS_OP_NAMES}


def logical_shape(term):
    """Logical (unpadded) shape of ``term``, or None when it has none."""
    if not hasattr(term, _LOGICAL_SHAPE_ATTR):
        from util.shape_util import get_term_shape

        try:
            shape = get_term_shape(term)
        except Exception:
            shape = None
        setattr(term, _LOGICAL_SHAPE_ATTR, None if shape is None else tuple(shape))
    return getattr(term, _LOGICAL_SHAPE_ATTR)


def zero_padded_dims(term):
    """Tensor dims of ``term`` along which its padding is zero.

    Positions past the logical extent of such a dim hold 0, both in the
    circuit and in the tensor evaluator the toy backend checks against.
    """
    if hasattr(term, _ZERO_DIMS_ATTR):
        return getattr(term, _ZERO_DIMS_ATTR)
    dims = frozenset()
    shape = logical_shape(term)
    if shape is not None:
        match term.op:
            case TensorOp.TENSOR:
                dims = frozenset(range(len(shape)))
            case TensorOp.ADD | TensorOp.SUB | TensorOp.MUL:
                a, b = term.cs[0], term.cs[1]
                # no broadcasting: a broadcast operand repeats its values
                # into the other operand's padding
                if (
                    a.op != TensorOp.CONST
                    and b.op != TensorOp.CONST
                    and logical_shape(a) == shape
                    and logical_shape(b) == shape
                ):
                    if term.op == TensorOp.MUL:
                        dims = zero_padded_dims(a) | zero_padded_dims(b)
                    else:
                        dims = zero_padded_dims(a) & zero_padded_dims(b)
            case TensorOp.CONV3D:
                args = Conv3dArgs.from_term(term)
                if args.depthwise:
                    # output channel c only reads input and filter channel c
                    if 0 in zero_padded_dims(args.input) or 0 in zero_padded_dims(
                        args.filter
                    ):
                        dims = frozenset({0})
                elif 0 in zero_padded_dims(args.filter):
                    # output channel c only reads filter row c
                    dims = frozenset({0})
            case TensorOp.GROUP_PAD:
                if logical_shape(term.cs[0]) == shape:
                    dims = zero_padded_dims(term.cs[0])
    setattr(term, _ZERO_DIMS_ATTR, dims)
    return dims


def _layout_dead_cts(layout, shape, zero_dims):
    """Indices of the ciphertexts of ``layout`` holding only zero padding."""
    ct_dims = layout.ct_dims
    if not ct_dims:
        return frozenset()
    digits = get_dim_indices(ct_dims)
    num_ct = layout.num_ct()
    dead = set()
    for ct in range(num_ct):
        # lower bound of the index along each tensor dim within this ct: the
        # slot dims (and a group's slot part) start at index 0
        low = {}
        for dim, digit in zip(ct_dims, digits):
            if dim.dim_type != DimType.FILL or dim.dim is None:
                continue
            offset = digit[ct] * dim.stride
            if dim.is_group:
                # positions past the walked positions are padding
                if offset >= prod(dim.group_counts) and set(dim.dim) <= zero_dims:
                    dead.add(ct)
                    break
                continue
            low[dim.dim] = low.get(dim.dim, 0) + offset
        else:
            if any(
                d in zero_dims and d < len(shape) and lb >= shape[d]
                for d, lb in low.items()
            ):
                dead.add(ct)
    return frozenset(dead)


def dead_cts(kernel):
    """Indices of the dead ciphertexts of ``kernel``'s layout (see module doc).

    Empty for kernel ops outside ``LIVENESS_OPS``, layouts with rolls (their
    ciphertexts mix indices) and terms without zero padding.
    """
    layout = kernel.layout
    if kernel.op not in _liveness_ops() or layout.rolls or not layout.ct_dims:
        return frozenset()
    term = layout.term
    if term is None:
        return frozenset()
    zero_dims = zero_padded_dims(term)
    if not zero_dims:
        return frozenset()
    shape = logical_shape(term)
    key = (layout.layout_str(), layout.n, shape, zero_dims)
    if key not in _DEAD_CTS_CACHE:
        _DEAD_CTS_CACHE[key] = _layout_dead_cts(layout, shape, zero_dims)
    return _DEAD_CTS_CACHE[key]


def num_live_cts(kernel):
    """Number of ciphertexts of ``kernel``'s layout that are not dead."""
    return kernel.layout.num_ct() - len(dead_cts(kernel))


def num_live_unique_cts(kernel):
    """``kernel.layout.num_ct_unique()`` without the dead ciphertexts.

    Unique ciphertexts are those at index 0 of every replicated ct dim.
    """
    layout = kernel.layout
    dead = dead_cts(kernel)
    if not dead:
        return layout.num_ct_unique()
    ct_dims = layout.ct_dims
    digits = get_dim_indices(ct_dims)
    replicated = [i for i, d in enumerate(ct_dims) if d.dim is None]
    return sum(
        1
        for ct in range(layout.num_ct())
        if ct not in dead and all(digits[i][ct] == 0 for i in replicated)
    )


def live_layout_str(kernel, layout):
    """``layout.layout_str()``, leaving out dead ciphertexts at the end.

    ``layout`` is ``kernel.layout`` or an equivalent layout with the same
    ciphertext order (e.g. ``dimension_merging(kernel.layout)``). When the dead
    ciphertexts of ``kernel`` are exactly the last ones, and the live ones are
    a prefix of the outermost ciphertext dim, that dim is printed with its
    live extent, e.g. ``[0:8:32];[0:32:1]`` with ciphertexts 6 and 7 dead
    prints as ``[0:6:32];[0:32:1]``. Otherwise the layout prints unchanged.
    """
    dead = dead_cts(kernel)
    ct_dims = layout.ct_dims
    if not dead or layout.rolls or not ct_dims:
        return layout.layout_str()
    num_ct = layout.num_ct()
    if num_ct != kernel.layout.num_ct():
        return layout.layout_str()
    num_live = num_ct - len(dead)
    if dead != frozenset(range(num_live, num_ct)):
        return layout.layout_str()
    inner = prod(d.extent for d in ct_dims[1:])
    if num_live == 0 or num_live % inner:
        return layout.layout_str()
    outer = ct_dims[0].copy()
    outer.extent = num_live // inner
    ct_str = str(outer) + "".join(str(d) for d in ct_dims[1:])
    slot_str = "".join(str(d) for d in layout.slot_dims)
    return f"{ct_str};{slot_str}"
