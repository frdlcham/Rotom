"""One MoViNet bottleneck, without the gate and the skip:

    # | operator         | input         | weight            | output
    1 | expand conv 1x1  | 5x28x28x32    | 184x32, b184      | 5x28x28x184
    2 | depth-conv, s2   | 5x28x28x184   | 5x3x3x184, b184   | 5x14x14x184
    3 | project conv 1x1 | 5x14x14x184   | 56x184, b56       | 5x14x14x56

Tensors are ``[C, T, H, W]``, as in ``movinet_a0.py``. The depthwise conv is
causal in time (it keeps all 5 frames) and TensorFlow ``SAME`` on H and W.

Each bias is a public ``[C, T, H, W]`` tensor, not ``[C, 1, 1, 1]`` as in
``movinet_a0.py``: a ``[C, 1, 1, 1]`` bias is replicated over the padded
extent of T, H and W, which fills the layout's gaps when an extent is not a
power of two (see ``test_per_channel_bias_fills_the_gaps``).

Run with ``-s`` to see the layouts:

    pytest tests/e2e/movinet/test_movinet_block.py -s
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

from assignment.assignment import LayoutAssignment
from benchmarks.e2e.movinet.movinet_a0 import _out_extent, _spatial_padding
from frontends.tensor import TensorTerm
from ir.kernel_cost import KernelCost
from tests.test_conv_direct import _run
from tests.test_util import get_default_args


@dataclass(frozen=True)
class BlockShape:
    channels: int  # C_in
    expand: int  # width after the 1x1 expand
    filters: int  # C_out of the project
    frames: int
    size: int  # H = W
    kernel: Tuple[int, int, int]  # depthwise (T, H, W)
    stride: Tuple[int, int, int]
    n: int  # slots per ciphertext


# the block from the table
BLOCK = BlockShape(32, 184, 56, 5, 28, (5, 3, 3), (1, 2, 2), n=32768)
# the same structure, small enough to compile quickly; like the block's, its
# extents are not powers of two (3 frames, 6 -> 3)
SMALL = BlockShape(4, 8, 4, 3, 6, (3, 3, 3), (1, 2, 2), n=1024)


def block_ir(shape: BlockShape, seed: int = 0):
    """The block over a secret ``x`` of shape ``[C, T, H, W]``.

    Returns:
        (term, inputs): the output ``[filters, T, H/2, W/2]`` and the value of
        every tensor the term names
    """
    rng = np.random.default_rng(seed)
    inputs = {
        "x": rng.normal(size=(shape.channels, shape.frames, shape.size, shape.size))
    }

    def public(name, value_shape, fan_in):
        inputs[name] = rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=value_shape)
        return TensorTerm.Tensor(name, list(value_shape), False)

    def bias(name, channels, size):
        """One value per channel, at the full ``[C, T, H, W]`` shape."""
        b = rng.normal(0.0, 0.1, size=(channels, 1, 1, 1))
        full = (channels, shape.frames, size, size)
        inputs[name + "/b"] = np.broadcast_to(b, full).copy()
        return TensorTerm.Tensor(name + "/b", list(full), False)

    kt, kh, kw = shape.kernel
    out = _out_extent(shape.size, shape.stride[1])
    x = TensorTerm.Tensor("x", list(inputs["x"].shape), True)

    # 1: expand 1x1x1, weight [C_out, C_in, 1, 1, 1]
    w = public("expand/w", (shape.expand, shape.channels, 1, 1, 1), shape.channels)
    h = TensorTerm.conv3d(x, w, 1, "same") + bias("expand", shape.expand, shape.size)

    # 2: depthwise kt x kh x kw, weight [1, K_t, K_h, K_w, C]
    w = public("depthwise/w", (1, kt, kh, kw, shape.expand), kt * kh * kw)
    _, h_pad, w_pad = _spatial_padding(
        shape.size, shape.size, shape.kernel, shape.stride
    )
    padding = ((kt - 1, 0), h_pad, w_pad)  # causal in time
    h = TensorTerm.depthwise_conv3d(h, w, shape.stride, padding)
    h = h + bias("depthwise", shape.expand, out)

    # 3: project 1x1x1
    w = public("project/w", (shape.filters, shape.expand, 1, 1, 1), shape.expand)
    h = TensorTerm.conv3d(h, w, 1, "same") + bias("project", shape.filters, out)
    return h, inputs


def _show_layouts(kernel, network="lan"):
    """Every kernel's layout; with ``cost``, also the cost model's HE
    operations per kernel and in total."""
    print()
    for k in kernel.post_order():
        line = str(k)
        ops = KernelCost(k, network).ops()
        if any(ops.values()):
            line += "  " + " ".join(f"{op}={count}" for op, count in ops.items())
        print(line)
    print("\noutput layout:", kernel.layout)
    _show_cost(kernel, network)


def _show_cost(kernel, network="lan"):
    """The cost model's totals, as the layout search scores them."""
    model = KernelCost(kernel, network)
    prices = model.cost_model()
    ops = model.total_operations()
    print(f"\ncost ({network}):")
    for op, count in ops.items():
        print(f"  {op:<4} {count:>10}  x {prices[op]:>6} = {count * prices[op]:>12.2f}")
    print(f"  comm {model.comm_cost():>36.2f}")
    print(f"  total {model.total_cost():>35.2f}")


def test_block_shapes():
    term, inputs = block_ir(BLOCK)
    out = _out_extent(BLOCK.size, 2)
    assert term.eval(inputs).shape == (BLOCK.filters, BLOCK.frames, out, out)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(SMALL, id="small"),
        pytest.param(BLOCK, id="block", marks=pytest.mark.slow),
    ],
)
def test_block_layout(shape):
    """Only the layout search: no lowering, no backend, no check.

    The args are ``_run``'s, so the layouts are the ones the compile tests
    run."""
    term, _ = block_ir(shape)
    args = get_default_args()
    args.n = shape.n
    _show_layouts(LayoutAssignment(term, args).run())


def test_small_block_compiles():
    term, inputs = block_ir(SMALL)
    kernel = _run(term, inputs, SMALL.n)
    _show_layouts(kernel)


@pytest.mark.slow
def test_block_compiles():
    term, inputs = block_ir(BLOCK)
    kernel = _run(term, inputs, BLOCK.n)
    _show_layouts(kernel)


@pytest.mark.xfail(
    strict=True,
    reason="a [C, 1, 1, 1] public operand is replicated over the padding of T",
)
def test_per_channel_bias_fills_the_gaps():
    """The expand conv with a ``[C, 1, 1, 1]`` bias on 3 frames (padded to 4).

    The values are right, but the bias also lands in the slots of frame 3,
    which should hold zero."""
    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.normal(size=(4, 3, 8, 8)),
        "w": rng.normal(size=(8, 4, 1, 1, 1)),
        "b": rng.normal(size=(8, 1, 1, 1)),
    }
    x = TensorTerm.Tensor("x", [4, 3, 8, 8], True)
    w = TensorTerm.Tensor("w", [8, 4, 1, 1, 1], False)
    b = TensorTerm.Tensor("b", [8, 1, 1, 1], False)
    _run(TensorTerm.conv3d(x, w, 1, "same") + b, inputs, 1024)
