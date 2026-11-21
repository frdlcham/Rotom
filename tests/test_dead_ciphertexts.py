"""Dead ciphertexts: ciphertexts of a padded layout holding only zero padding.

Unpadded channel counts (e.g. 6 instead of 8) are laid out padded to a power
of 2; ciphertexts that only hold padded channels are dead (``ir.liveness``):
they are not counted by the cost model, lowered to an explicit ZERO_MASK and,
when they are the last ciphertexts, left out of the printed layout.
"""

import numpy as np
import pytest

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from ir.layout import Layout
from ir.liveness import dead_cts, num_live_unique_cts, zero_padded_dims
from lower.lower import Lower
from tests.test_util import get_default_args
from util.util import ceil_div


def _args(n):
    args = get_default_args()
    args.n = n
    args.rolls = True
    args.backend = "toy"
    return args


def _tensor_kernel(shape, dims, n):
    term = TensorTerm.Tensor("t", shape, True)
    return Kernel(KernelOp.TENSOR, [], Layout(term, [], dims, n, True))


class TestLiveness:
    def test_trailing_dead_cts(self):
        # channels 6, 7 of [0:8:1] are padding
        kernel = _tensor_kernel([6, 2, 4], [Dim(0, 8, 1), Dim(1, 2, 1), Dim(2, 4, 1)], 4)
        assert dead_cts(kernel) == frozenset(range(12, 16))
        assert "[0:6:1][1:2:1];[2:4:1]" in repr(kernel)
        # the layout itself (cache keys, parsing) is unchanged
        assert kernel.layout.layout_str() == "[0:8:1][1:2:1];[2:4:1]"

    def test_non_trailing_dead_cts_print_unchanged(self):
        kernel = _tensor_kernel([6, 2, 4], [Dim(1, 2, 1), Dim(0, 8, 1), Dim(2, 4, 1)], 4)
        assert dead_cts(kernel) == frozenset({6, 7, 14, 15})
        assert "[1:2:1][0:8:1];[2:4:1]" in repr(kernel)

    def test_channels_split_over_slots_and_cts(self):
        # channel = 2 * ct + slot: ct 3 holds channels 6, 7 only
        kernel = _tensor_kernel([6, 4], [Dim(0, 4, 2), Dim(0, 2, 1), Dim(1, 4, 1)], 8)
        assert dead_cts(kernel) == frozenset({3})
        assert "[0:3:2];[0:2:1][1:4:1]" in repr(kernel)

    def test_partially_padded_ct_is_live(self):
        # ct 1 holds channels 4..7 of which 4 is real
        kernel = _tensor_kernel([5, 4], [Dim(0, 2, 4), Dim(0, 4, 1), Dim(1, 4, 1)], 16)
        assert dead_cts(kernel) == frozenset()

    def test_powers_of_2_have_no_dead_cts(self):
        kernel = _tensor_kernel([8, 2, 4], [Dim(0, 8, 1), Dim(1, 2, 1), Dim(2, 4, 1)], 4)
        assert dead_cts(kernel) == frozenset()
        assert num_live_unique_cts(kernel) == kernel.layout.num_ct_unique()

    def test_replicated_cts_are_not_unique(self):
        dims = [Dim(None, 2, 1), Dim(0, 8, 1), Dim(1, 4, 1)]
        kernel = _tensor_kernel([6, 4], dims, 4)
        assert dead_cts(kernel) == frozenset({6, 7, 14, 15})
        assert num_live_unique_cts(kernel) == 6

    def test_poly_call_padding_is_not_zero(self):
        x = TensorTerm.Tensor("x", [6, 4], True)
        assert zero_padded_dims(x.poly_call("silu", -3.0, 3.0)) == frozenset()

    def test_broadcast_add_padding_is_not_zero(self):
        x = TensorTerm.Tensor("x", [6, 4, 4], True)
        b = TensorTerm.Tensor("b", [6, 1, 1], False)
        assert zero_padded_dims(x + b) == frozenset()
        y = TensorTerm.Tensor("y", [6, 4, 4], False)
        assert zero_padded_dims(x + y) == frozenset({0, 1, 2})

    def test_conv3d_output_channels(self):
        x = TensorTerm.Tensor("x", [4, 2, 4, 4], True)
        f = TensorTerm.Tensor("f", [6, 4, 1, 1, 1], False)
        assert zero_padded_dims(TensorTerm.conv3d(x, f, 1, "same")) == frozenset({0})

    def test_equal_looking_terms_do_not_share_shapes(self):
        # terms compare by string, which leaves out the shapes
        dims = [Dim(0, 8, 1), Dim(1, 2, 1), Dim(2, 4, 1)]
        assert dead_cts(_tensor_kernel([6, 2, 4], dims, 4))
        assert not dead_cts(_tensor_kernel([8, 2, 4], dims, 4))


def _conv_bias(c_out):
    x = TensorTerm.Tensor("x", [4, 2, 4, 4], True)
    f = TensorTerm.Tensor("f", [c_out, 4, 1, 1, 1], False)
    b = TensorTerm.Tensor("b", [c_out, 2, 4, 4], False)
    return TensorTerm.conv3d(x, f, 1, "same") + b


class TestCostAndPrinting:
    def test_unpadded_conv_skips_dead_cts(self):
        padded = LayoutAssignment(_conv_bias(8), _args(32)).run()
        unpadded = LayoutAssignment(_conv_bias(6), _args(32)).run()
        assert padded.layout.layout_str() == unpadded.layout.layout_str()
        assert padded.layout.layout_str().startswith("[0:8:1];")
        assert repr(unpadded).endswith("[0:6:1];[1:2:1][2:4:1][3:4:1]")
        ops_padded = KernelCost(padded, "lan").total_operations()
        ops_unpadded = KernelCost(unpadded, "lan").total_operations()
        # 6 of the 8 output ciphertexts are computed
        assert ops_unpadded == {k: v * 6 // 8 for k, v in ops_padded.items()}

    def test_dead_input_cts_are_not_sent(self):
        x = TensorTerm.Tensor("x", [6, 2, 4, 4], True)
        f = TensorTerm.Tensor("f", [4, 6, 1, 1, 1], False)
        kernel = LayoutAssignment(TensorTerm.conv3d(x, f, 1, "same"), _args(32)).run()
        inputs = [k for k in kernel.post_order() if k.op == KernelOp.TENSOR]
        assert len(inputs) == 1
        live = num_live_unique_cts(inputs[0])
        assert live < inputs[0].layout.num_ct_unique()
        assert KernelCost(kernel, "lan").num_input_cts() == live


def _small_mobile_block(c1, c3, n):
    """GROUP_PAD -> conv3d -> depthwise (strided) -> conv3d, with biases."""
    t, h, w, c_in = 3, 6, 6, 4
    s2 = (1, 2, 2)
    to, ho, wo = (ceil_div(e, s) for e, s in zip((t, h, w), s2))
    x = TensorTerm.Tensor("input", [c_in, t, h, w], True)
    x = x.group_pad(order=(2, 3, 1), steps=s2)
    out = TensorTerm.conv3d(
        x, TensorTerm.Tensor("f1", [c1, c_in, 1, 1, 1], False), 1, "same"
    ) + TensorTerm.Tensor("b1", [c1, t, h, w], False)
    out = TensorTerm.depthwise_conv3d(
        out, TensorTerm.Tensor("f2", [c1, t, 3, 3], False), s2, "same"
    ) + TensorTerm.Tensor("b2", [c1, to, ho, wo], False)
    out = TensorTerm.conv3d(
        out, TensorTerm.Tensor("f3", [c3, c1, 1, 1, 1], False), 1, "same"
    ) + TensorTerm.Tensor("b3", [c3, to, ho, wo], False)
    rng = np.random.default_rng(0)
    inputs = {
        "input": rng.standard_normal((c_in, t, h, w)),
        "f1": rng.standard_normal((c1, c_in, 1, 1, 1)),
        "b1": rng.standard_normal((c1, t, h, w)),
        "f2": rng.standard_normal((c1, t, 3, 3)),
        "b2": rng.standard_normal((c1, to, ho, wo)),
        "f3": rng.standard_normal((c3, c1, 1, 1, 1)),
        "b3": rng.standard_normal((c3, to, ho, wo)),
    }
    return out, inputs


@pytest.mark.parametrize("n", [32, 64])
def test_unpadded_block_toy(n):
    """Every kernel of an unpadded block matches the tensor evaluator."""
    term, inputs = _small_mobile_block(6, 3, n)
    args = _args(n)
    kernel = LayoutAssignment(term, args).run()
    assert any(dead_cts(k) for k in kernel.post_order())
    Toy(Lower(kernel).run(), inputs, args).run()
