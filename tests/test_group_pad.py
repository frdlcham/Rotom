"""
Test suite for the GROUP_PAD operation.

GROUP_PAD keeps a tensor's shape but lays out consecutive dims together in one
group dim padded as a whole, e.g. [C, 5, 14, 14] grouped over dims 1..3 takes
980 -> 1024 slots per channel instead of 8*16*16 = 2048.
"""

import numpy as np
import pytest

from assignment.assignment import LayoutAssignment
from frontends.tensor import TensorTerm
from ir.analysis.shape import Shape
from ir.dim import Dim
from ir.kernel import KernelOp
from ir.layout import Layout
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.flatten_repack import group_pad_plan_for_kernel
from util.layout_util import apply_layout


def _shapes(term):
    shape_an = Shape(term)
    shape_an.run()
    return shape_an.shapes[term], shape_an.padded_shapes[term]


def _group_dims(layout):
    return [d for d in layout.get_dims() if d.is_group]


class TestGroupDim:
    def test_string_round_trip(self):
        layout = Layout.from_string("[0:4:1][1-3@5x14x14:1024:1]", 4096)
        assert layout.layout_str() == "[0:4:1][1-3@5x14x14:1024:1]"
        (group,) = _group_dims(layout)
        assert group == Dim.group_dim(1, [5, 14, 14], 1024)

    def test_apply_layout_is_row_major_with_padding_at_the_end(self):
        x = np.arange(2 * 3 * 2 * 3, dtype=np.float64).reshape(2, 3, 2, 3)
        layout = Layout.from_string("[0:2:1][1-3@3x2x3:32:1]", 64)
        (ct,) = apply_layout(x, layout)
        expected = np.zeros((2, 32))
        expected[:, :18] = x.reshape(2, 18)
        np.testing.assert_array_equal(np.array(ct).reshape(2, 32), expected)

    def test_group_split_over_ciphertexts(self):
        x = np.arange(2 * 3 * 2 * 3, dtype=np.float64).reshape(2, 3, 2, 3)
        layout = Layout.from_string("[0:2:1][1-3@3x2x3:32:1]", 16)
        assert layout.num_ct() == 4
        packed = np.array(apply_layout(x, layout)).reshape(2, 32)
        np.testing.assert_array_equal(packed[:, :18], x.reshape(2, 18))
        assert not packed[:, 18:].any()


class TestGroupPadShape:
    def test_shape_unchanged_and_group_unpadded(self):
        a = TensorTerm.Tensor("a", [4, 5, 14, 14], True)
        shape, padded = _shapes(a.group_pad(1, 3))
        assert shape == [4, 5, 14, 14]
        assert padded == [4, 5, 14, 14]

    def test_eval_is_identity(self):
        a = TensorTerm.Tensor("a", [2, 3, 5], True)
        x = np.arange(30, dtype=np.float64).reshape(2, 3, 5)
        np.testing.assert_array_equal(a.group_pad(1, 2).eval({"a": x}), x)


class TestGroupPadToy:
    def _run(self, tensor_ir, inputs, args, backend):
        expected = tensor_ir.eval(inputs)
        results, kernel = run_compiler_and_backend(tensor_ir, inputs, args, backend)
        assert_results_equal(apply_layout(expected, kernel.layout), results, backend)
        return kernel

    def test_group_pad(self, backend):
        args = get_default_args()
        args.n = 256
        rng = np.random.default_rng(0)
        inputs = {"x": rng.normal(size=(2, 3, 6, 6))}
        x = TensorTerm.Tensor("x", [2, 3, 6, 6], True)
        kernel = self._run(x.group_pad(1, 3), inputs, args, backend)
        assert kernel.op == KernelOp.GROUP_PAD
        assert _group_dims(kernel.layout)

    def test_group_pad_keeps_trailing_dims(self, backend):
        args = get_default_args()
        args.n = 256
        rng = np.random.default_rng(0)
        inputs = {"x": rng.normal(size=(3, 5, 6, 7))}
        x = TensorTerm.Tensor("x", [3, 5, 6, 7], True)
        self._run(x.group_pad(0, 1), inputs, args, backend)

    def test_group_pad_spans_ciphertexts(self, backend):
        """3*6*7 = 126 -> 128 per channel, 4 channels over 2 ciphertexts of 256."""
        args = get_default_args()
        args.n = 256
        rng = np.random.default_rng(0)
        inputs = {"x": rng.normal(size=(4, 3, 6, 7))}
        x = TensorTerm.Tensor("x", [4, 3, 6, 7], True)
        self._run(x.group_pad(1, 3), inputs, args, backend)

    def test_group_pad_after_computation(self, backend):
        args = get_default_args()
        args.n = 256
        rng = np.random.default_rng(0)
        inputs = {"x": rng.normal(size=(2, 3, 6, 6)), "y": rng.normal(size=(2, 3, 6, 6))}
        x = TensorTerm.Tensor("x", [2, 3, 6, 6], True)
        y = TensorTerm.Tensor("y", [2, 3, 6, 6], False)
        self._run((x * y).group_pad(1, 3), inputs, args, backend)

    def test_explicit_extent(self, backend):
        args = get_default_args()
        args.n = 256
        rng = np.random.default_rng(0)
        inputs = {"x": rng.normal(size=(2, 3, 6, 6))}
        x = TensorTerm.Tensor("x", [2, 3, 6, 6], True)
        kernel = self._run(x.group_pad(1, 3, extent=128), inputs, args, backend)
        assert {d.extent for d in _group_dims(kernel.layout)} == {128}


class TestSteppedGroupDim:
    """Groups whose axes jump by a per-axis step, e.g. (H, W) parity packing."""

    def test_string_forms(self):
        layout = Layout.from_string("[2:2:1][3:2:1];[1-3@5x14x14:256:(1, 2, 2)]", 256)
        assert layout.layout_str() == "[2:2:1][3:2:1];[1-3@5x14x14:256:(1,2,2)]"
        (group,) = _group_dims(layout)
        assert group.steps == (1, 2, 2) and group.group_counts == (5, 7, 7)
        assert group.is_stepped
        assert group != Dim.group_dim(1, [5, 14, 14], 256)
        # unit steps print as a plain group
        assert str(Dim.group_dim(1, [5, 14, 14], 1024, steps=(1, 1, 1))) == (
            "[1-3@5x14x14:1024:1]"
        )

    def test_split_over_ciphertexts_keeps_steps(self):
        layout = Layout.from_string("[1-3@5x14x14:512:(1,2,2)]", 256)
        assert layout.layout_str() == (
            "[1-3@5x14x14:2:256:(1,2,2)];[1-3@5x14x14:256:(1,2,2)]"
        )
        again = Layout.from_string(layout.layout_str(), 256)
        assert again.layout_str() == layout.layout_str()

    def test_parity_packing(self):
        """Each ciphertext holds one (H, W) parity class, row-major."""
        x = np.arange(64, dtype=np.float64).reshape(1, 1, 8, 8)
        layout = Layout.from_string("[2:2:1][3:2:1];[1-3@1x8x8:16:(1,2,2)]", 16)
        cts = np.array(apply_layout(x, layout)).reshape(2, 2, 4, 4)
        for a in range(2):
            for b in range(2):
                np.testing.assert_array_equal(cts[a, b], x[0, 0, a::2, b::2])

    def test_odd_extent_is_padded(self):
        """7 = 4 + 3: the odd parity class has one padding row/column."""
        x = np.arange(1, 50, dtype=np.float64).reshape(1, 1, 7, 7)
        layout = Layout.from_string("[2:2:1][3:2:1];[1-3@1x7x7:16:(1,2,2)]", 16)
        cts = np.array(apply_layout(x, layout)).reshape(2, 2, 4, 4)
        for a in range(2):
            for b in range(2):
                sub = x[0, 0, a::2, b::2]
                expected = np.zeros((4, 4))
                expected[: sub.shape[0], : sub.shape[1]] = sub
                np.testing.assert_array_equal(cts[a, b], expected)


class TestSteppedGroupPadToy:
    _run = TestGroupPadToy._run

    def _check_stepped(self, shape, n, backend, **kwargs):
        args = get_default_args()
        args.n = n
        rng = np.random.default_rng(0)
        inputs = {"x": rng.normal(size=shape)}
        x = TensorTerm.Tensor("x", list(shape), True)
        kernel = self._run(x.group_pad(1, 3, steps=(1, 2, 2), **kwargs), inputs, args, backend)
        assert kernel.op == KernelOp.GROUP_PAD
        (steps,) = {d.steps for d in _group_dims(kernel.layout)}
        assert steps == (1, 2, 2)
        # one residue dim per stepped axis holds the parity
        residues = [
            d for d in kernel.layout.get_dims() if not d.is_group and d.dim in (2, 3)
        ]
        assert sorted((d.dim, d.extent) for d in residues) == [(2, 2), (3, 2)]
        return kernel

    def test_parity_group_pad(self, backend):
        kernel = self._check_stepped((2, 3, 6, 6), 256, backend)
        # 3 * 3 * 3 = 27 positions -> 32
        assert {d.extent for d in _group_dims(kernel.layout)} == {32}

    def test_parity_group_pad_odd_extents(self, backend):
        self._check_stepped((2, 3, 7, 7), 256, backend)

    def test_parity_group_pad_explicit_extent(self, backend):
        kernel = self._check_stepped((2, 3, 6, 6), 256, backend, extent=64)
        assert {d.extent for d in _group_dims(kernel.layout)} == {64}

    def test_steps_must_match_grouped_dims(self):
        x = TensorTerm.Tensor("x", [2, 3, 6, 6], True)
        with pytest.raises(AssertionError):
            x.group_pad(1, 3, steps=(2, 2))


class TestPermutedGroupDim:
    """Groups that lay their dims out in another order, e.g. (H, T, W)."""

    def test_string_forms(self):
        layout = Layout.from_string("[0:4:1][(2, 1, 3)@14x5x14:1024:1]", 4096)
        assert layout.layout_str() == "[0:4:1][(2,1,3)@14x5x14:1024:1]"
        (group,) = _group_dims(layout)
        assert group.dim == (2, 1, 3) and group.group == (14, 5, 14)
        assert group.is_permuted
        assert group == Dim.group_dim(1, [5, 14, 14], 1024, order=(2, 1, 3))
        assert group != Dim.group_dim(1, [5, 14, 14], 1024)
        # the default order prints as before
        assert not Dim.group_dim(1, [5, 14, 14], 1024, order=(1, 2, 3)).is_permuted
        assert str(Dim.group_dim(1, [5, 14, 14], 1024, order=(1, 2, 3))) == (
            "[1-3@5x14x14:1024:1]"
        )

    def test_stepped_string_forms(self):
        # steps are given per tensor dim and printed in layout order
        group = Dim.group_dim(1, [5, 14, 14], 256, steps=(1, 2, 4), order=(3, 1, 2))
        assert str(group) == "[(3,1,2)@14x5x14:256:(4,1,2)]"
        assert group.group_counts == (4, 5, 7)
        layout = Layout.from_string("[3:4:1][2:2:1];" + str(group), 256)
        assert layout.layout_str() == "[3:4:1][2:2:1];[(3,1,2)@14x5x14:256:(4,1,2)]"
        split = Layout.from_string("[(3,1,2)@14x5x14:512:(4,1,2)]", 256)
        assert split.layout_str() == (
            "[(3,1,2)@14x5x14:2:256:(4,1,2)];[(3,1,2)@14x5x14:256:(4,1,2)]"
        )
        again = Layout.from_string(split.layout_str(), 256)
        assert again.layout_str() == split.layout_str()

    def test_apply_layout_follows_order(self):
        x = np.arange(2 * 3 * 2 * 3, dtype=np.float64).reshape(2, 3, 2, 3)
        layout = Layout.from_string("[0:2:1][(2,1,3)@2x3x3:32:1]", 64)
        (ct,) = apply_layout(x, layout)
        expected = np.zeros((2, 32))
        expected[:, :18] = x.transpose(0, 2, 1, 3).reshape(2, 18)
        np.testing.assert_array_equal(np.array(ct).reshape(2, 32), expected)

    def test_order_must_be_permutation(self):
        x = TensorTerm.Tensor("x", [2, 3, 6, 6], True)
        with pytest.raises(AssertionError):
            x.group_pad(1, 3, order=(1, 2, 4))
        with pytest.raises(AssertionError):
            x.group_pad(1, 3, order=(1, 1, 3))


class TestPermutedGroupPadToy:
    _run = TestGroupPadToy._run

    def _check(self, shape, n, backend, order, **kwargs):
        args = get_default_args()
        args.n = n
        rng = np.random.default_rng(0)
        inputs = {"x": rng.normal(size=shape)}
        x = TensorTerm.Tensor("x", list(shape), True)
        kernel = self._run(
            x.group_pad(1, 3, order=order, **kwargs), inputs, args, backend
        )
        assert kernel.op == KernelOp.GROUP_PAD
        assert {d.dim for d in _group_dims(kernel.layout)} == {tuple(order)}
        return kernel

    def test_permuted_group_pad(self, backend):
        kernel = self._check((2, 3, 5, 6), 256, backend, (2, 1, 3))
        assert {d.group for d in _group_dims(kernel.layout)} == {(5, 3, 6)}

    def test_permuted_group_pad_spans_ciphertexts(self, backend):
        self._check((4, 3, 6, 7), 256, backend, (3, 2, 1))

    def test_permuted_stepped_group_pad(self, backend):
        kernel = self._check((2, 3, 6, 6), 256, backend, (2, 1, 3), steps=(1, 2, 2))
        assert {d.steps for d in _group_dims(kernel.layout)} == {(2, 1, 2)}

    def test_permuted_group_into_conv3d(self, backend):
        args = get_default_args()
        args.n = 256
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=(2, 3, 6, 6)),
            "f": rng.normal(size=(4, 2, 3, 3, 3)),
        }
        x = TensorTerm.Tensor("x", [2, 3, 6, 6], True).group_pad(1, 3, order=(2, 1, 3))
        f = TensorTerm.Tensor("f", [4, 2, 3, 3, 3], False)
        kernel = self._run(TensorTerm.conv3d(x, f, 1, "same"), inputs, args, backend)
        assert kernel.op == KernelOp.CONV3D
        assert {d.dim for d in _group_dims(kernel.layout)} == {(2, 1, 3)}


class TestGroupPadEdwardShape:
    def test_group_pad_saves_slots(self):
        """[4, 5, 14, 14]: 4 channels * 1024 fit one ciphertext instead of two."""
        args = get_default_args()
        args.n = 4096
        x = TensorTerm.Tensor("x", [4, 5, 14, 14], True)
        assert LayoutAssignment(x, args).run().layout.num_ct() == 2

        kernel = LayoutAssignment(x.group_pad(1, 3), args).run()
        assert kernel.op == KernelOp.GROUP_PAD
        assert kernel.layout.num_ct() == 1
        (group,) = _group_dims(kernel.layout)
        assert group.extent == 1024 and group.group == (5, 14, 14)

        plan = group_pad_plan_for_kernel(kernel)
        assert len(plan.stages) > 1
        assert plan.ops()["rot"] < 5 * 14


class TestGroupPadConv3d:
    """CONV3D (stride 1, "same") consuming a GROUP_PAD input keeps its group."""

    def _run(self, tensor_ir, inputs, args, backend):
        expected = tensor_ir.eval(inputs)
        results, kernel = run_compiler_and_backend(tensor_ir, inputs, args, backend)
        assert_results_equal(apply_layout(expected, kernel.layout), results, backend)
        assert _group_dims(kernel.layout)
        return kernel

    def _conv(self, c_in, c_out, k, shape, bias=False):
        x = TensorTerm.Tensor("x", [c_in] + shape, True).group_pad(1, 3)
        f = TensorTerm.Tensor("f", [c_out, c_in, k, k, k], False)
        out = TensorTerm.conv3d(x, f, 1, "same")
        if bias:
            out = out + TensorTerm.Tensor("b", [c_out] + shape, False)
        return out

    def _inputs(self, c_in, c_out, k, shape, bias=False):
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=[c_in] + shape),
            "f": rng.normal(size=(c_out, c_in, k, k, k)),
        }
        if bias:
            inputs["b"] = rng.normal(size=[c_out] + shape)
        return inputs

    def test_pointwise_conv(self, backend):
        args = get_default_args()
        args.n = 256
        kernel = self._run(
            self._conv(2, 4, 1, [3, 6, 6]), self._inputs(2, 4, 1, [3, 6, 6]), args, backend
        )
        assert kernel.op == KernelOp.CONV3D

    def test_conv_3x3x3(self, backend):
        args = get_default_args()
        args.n = 256
        self._run(
            self._conv(2, 4, 3, [3, 6, 6]), self._inputs(2, 4, 3, [3, 6, 6]), args, backend
        )

    def test_conv_3x3x3_channels_share_ciphertext(self, backend):
        """Taps rotate across row, channel and group-padding boundaries."""
        args = get_default_args()
        args.n = 1024
        self._run(
            self._conv(2, 4, 3, [3, 5, 6]), self._inputs(2, 4, 3, [3, 5, 6]), args, backend
        )

    def test_pointwise_conv_with_bias(self, backend):
        args = get_default_args()
        args.n = 256
        self._run(
            self._conv(2, 4, 1, [3, 6, 6], bias=True),
            self._inputs(2, 4, 1, [3, 6, 6], bias=True),
            args,
            backend,
        )

    def test_bias_add_compacts_before_next_conv(self, backend):
        """conv 4->8 leaves 8 sparse ciphertexts; the bias add compacts them to 1."""
        args = get_default_args()
        args.n = 1024
        shape = [3, 6, 6]
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=[4] + shape),
            "f": rng.normal(size=(8, 4, 1, 1, 1)),
            "b": rng.normal(size=[8] + shape),
            "g": rng.normal(size=(8, 8, 1, 1, 1)),
        }
        x = TensorTerm.Tensor("x", [4] + shape, True).group_pad(1, 3)
        f = TensorTerm.Tensor("f", [8, 4, 1, 1, 1], False)
        b = TensorTerm.Tensor("b", [8] + shape, False)
        g = TensorTerm.Tensor("g", [8, 8, 1, 1, 1], False)
        out = TensorTerm.conv3d(TensorTerm.conv3d(x, f, 1, "same") + b, g, 1, "same")
        kernel = self._run(out, inputs, args, backend)
        (compact,) = [k for k in kernel.post_order() if k.op == KernelOp.COMPACT]
        assert _group_dims(compact.layout)
        assert compact.layout.num_ct() == 1

    def test_depthwise_then_grouped_project_conv(self, backend):
        """Edward_example structure: depthwise -> group_pad -> project conv + bias."""
        args = get_default_args()
        args.n = 1024
        c, c_out, t, h, w = 4, 2, 3, 6, 6
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=(c, t, h, w)),
            "fd": rng.normal(size=(c, 3, 3, 3)),
            "bd": rng.normal(size=(c, t, h // 2, w // 2)),
            "fp": rng.normal(size=(c_out, c, 1, 1, 1)),
            "bp": rng.normal(size=(c_out, t, h // 2, w // 2)),
        }
        x = TensorTerm.Tensor("x", [c, t, h, w], True)
        fd = TensorTerm.Tensor("fd", [c, 3, 3, 3], False)
        bd = TensorTerm.Tensor("bd", [c, t, h // 2, w // 2], False)
        fp = TensorTerm.Tensor("fp", [c_out, c, 1, 1, 1], False)
        bp = TensorTerm.Tensor("bp", [c_out, t, h // 2, w // 2], False)
        depth = (TensorTerm.depthwise_conv3d(x, fd, (1, 2, 2), "same") + bd).group_pad(1, 3)
        out = TensorTerm.conv3d(depth, fp, 1, "same") + bp
        self._run(out, inputs, args, backend)


class TestGroupPadDepthwiseConv3d:
    """DEPTHWISE_CONV3D consuming a GROUP_PAD input produces a grouped output."""

    def _run(self, tensor_ir, inputs, args, backend):
        expected = tensor_ir.eval(inputs)
        results, kernel = run_compiler_and_backend(tensor_ir, inputs, args, backend)
        assert_results_equal(apply_layout(expected, kernel.layout), results, backend)
        return kernel

    def _depthwise(self, c, shape, k, stride, padding="same"):
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=[c] + shape),
            "f": rng.normal(size=(c, k, k, k)),
        }
        x = TensorTerm.Tensor("x", [c] + shape, True).group_pad(1, 3)
        f = TensorTerm.Tensor("f", [c, k, k, k], False)
        return TensorTerm.depthwise_conv3d(x, f, stride, padding), inputs

    def _check(self, c, shape, k, stride, n, backend, padding="same"):
        args = get_default_args()
        args.n = n
        out, inputs = self._depthwise(c, shape, k, stride, padding)
        kernel = self._run(out, inputs, args, backend)
        (dw,) = [k for k in kernel.post_order() if k.op == KernelOp.DEPTHWISE_CONV3D]
        assert _group_dims(dw.cs[0].layout)
        assert _group_dims(dw.layout)
        return dw

    def test_stride1_keeps_input_layout(self, backend):
        dw = self._check(2, [3, 5, 6], 3, 1, 256, backend)
        assert dw.layout.layout_str() == dw.cs[0].layout.layout_str()

    def test_stride1_channels_share_ciphertext(self, backend):
        """Taps rotate across row, channel and group-padding boundaries."""
        self._check(4, [3, 5, 6], 3, 1, 512, backend)

    def test_stride_1_2_2_repacks_to_smaller_group(self, backend):
        dw = self._check(2, [3, 6, 6], 3, (1, 2, 2), 256, backend)
        (group,) = {d.group for d in _group_dims(dw.layout)}
        assert group == (3, 3, 3)

    def test_stride_2_odd_extents(self, backend):
        self._check(4, [3, 5, 7], 3, 2, 256, backend)

    def test_stride_1_2_2_group_spans_ciphertexts(self, backend):
        """3*6*7 = 126 -> 128 per channel, larger than a 64-slot ciphertext."""
        self._check(2, [3, 6, 7], 3, (1, 2, 2), 64, backend)

    def test_valid_stride1_repacks(self, backend):
        dw = self._check(2, [4, 5, 6], 3, 1, 256, backend, padding="valid")
        (group,) = {d.group for d in _group_dims(dw.layout)}
        assert group == (2, 3, 4)

    def test_expand_depthwise_project(self, backend):
        """Edward_example structure with the input grouped once at the start."""
        args = get_default_args()
        args.n = 1024
        c_in, c, c_out, t, h, w = 2, 4, 2, 3, 6, 6
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=(c_in, t, h, w)),
            "fe": rng.normal(size=(c, c_in, 1, 1, 1)),
            "be": rng.normal(size=(c, t, h, w)),
            "fd": rng.normal(size=(c, 3, 3, 3)),
            "bd": rng.normal(size=(c, t, h // 2, w // 2)),
            "fp": rng.normal(size=(c_out, c, 1, 1, 1)),
            "bp": rng.normal(size=(c_out, t, h // 2, w // 2)),
        }
        x = TensorTerm.Tensor("x", [c_in, t, h, w], True).group_pad(1, 3)
        fe = TensorTerm.Tensor("fe", [c, c_in, 1, 1, 1], False)
        be = TensorTerm.Tensor("be", [c, t, h, w], False)
        fd = TensorTerm.Tensor("fd", [c, 3, 3, 3], False)
        bd = TensorTerm.Tensor("bd", [c, t, h // 2, w // 2], False)
        fp = TensorTerm.Tensor("fp", [c_out, c, 1, 1, 1], False)
        bp = TensorTerm.Tensor("bp", [c_out, t, h // 2, w // 2], False)
        expand = TensorTerm.conv3d(x, fe, 1, "same") + be
        depth = TensorTerm.depthwise_conv3d(expand, fd, (1, 2, 2), "same") + bd
        out = TensorTerm.conv3d(depth, fp, 1, "same") + bp
        kernel = self._run(out, inputs, args, backend)
        (dw,) = [k for k in kernel.post_order() if k.op == KernelOp.DEPTHWISE_CONV3D]
        assert _group_dims(dw.layout)
        assert _group_dims(kernel.layout)


# ---------------------------------------------------------------------------
# Stepped (parity) groups through conv3d, depthwise_conv3d and add
# ---------------------------------------------------------------------------


def _residue_dims(layout):
    """Non-group dims of tensor dims that a stepped group walks with step > 1."""
    from ir.layout_utils import is_residue_dim, stepped_axes

    axes = stepped_axes(layout.get_dims())
    return [d for d in layout.get_dims() if is_residue_dim(d, axes)]


def _kernel_ops(tensor_ir, n, op):
    """(kernel, #rotations, #multiplications) added by the first ``op`` kernel."""
    from ir.he import HEOp
    from lower.lower import Lower

    def terms(v):
        cts = v.cts if hasattr(v, "cts") else v
        seen = set()
        for t in cts.values():
            seen.update(t.post_order())
        return seen

    args = get_default_args()
    args.n = n
    root = LayoutAssignment(tensor_ir, args).run()
    env = Lower(root).run()
    (kernel,) = [k for k in root.post_order() if k.op == op]
    new = terms(env[kernel]) - set().union(*[terms(env[c]) for c in kernel.cs if c in env])
    return (
        kernel,
        sum(t.op == HEOp.ROT for t in new),
        sum(t.op == HEOp.MUL for t in new),
    )


class TestSteppedConv3d:
    """CONV3D on a stepped group: stride 1 keeps it, stride s == step drops parity."""

    @pytest.fixture
    def backend(self):
        return "toy"

    def _conv(self, c_in, c_out, k, shape, stride, steps):
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=[c_in] + shape),
            "f": rng.normal(size=(c_out, c_in, k, k, k)),
        }
        x = TensorTerm.Tensor("x", [c_in] + shape, True).group_pad(1, 3, steps=steps)
        f = TensorTerm.Tensor("f", [c_out, c_in, k, k, k], False)
        return TensorTerm.conv3d(x, f, stride, "same"), inputs

    def _check(self, c_in, c_out, k, shape, stride, steps, n, backend):
        args = get_default_args()
        args.n = n
        out, inputs = self._conv(c_in, c_out, k, shape, stride, steps)
        kernel = TestGroupPadToy._run(self, out, inputs, args, backend)
        assert kernel.op == KernelOp.CONV3D
        assert kernel.cs[0].op == KernelOp.REPLICATE or k == 1
        return kernel

    def test_pointwise_keeps_parity(self, backend):
        kernel = self._check(2, 4, 1, [3, 6, 6], 1, (1, 2, 2), 256, backend)
        (group,) = set(_group_dims(kernel.layout))
        assert group.steps == (1, 2, 2)
        assert sorted(d.dim for d in _residue_dims(kernel.layout)) == [2, 3]

    def test_3x3x3_taps_cross_parity_ciphertexts(self, backend):
        """n=32: both parity dims index ciphertexts, so H/W taps change ciphertext."""
        kernel = self._check(2, 3, 3, [3, 6, 6], 1, (1, 2, 2), 32, backend)
        residues = _residue_dims(kernel.layout)
        assert len(residues) == 2 and all(d in kernel.layout.ct_dims for d in residues)

    def test_3x3x3_parity_in_slots(self, backend):
        self._check(2, 4, 3, [3, 6, 6], 1, (1, 2, 2), 256, backend)

    def test_stride2_keeps_first_residue_class(self, backend):
        """Anchors o * 2 are residue class 0: output is a plain 2x3x3 group."""
        kernel = self._check(2, 3, 3, [4, 6, 6], 2, (2, 2, 2), 32, backend)
        # the group may be split over ciphertexts and slots
        groups = _group_dims(kernel.layout)
        assert groups
        assert all(g.group == (2, 3, 3) and not g.is_stepped for g in groups)
        assert not _residue_dims(kernel.layout)

    def test_stride2_odd_extents_group_spans_ciphertexts(self, backend):
        self._check(2, 3, 3, [3, 5, 7], 2, (2, 2, 2), 16, backend)

    def test_stride2_needs_matching_steps(self):
        out, _ = self._conv(2, 3, 3, [4, 6, 6], 2, (1, 2, 2))
        args = get_default_args()
        args.n = 256
        with pytest.raises(AssertionError):
            LayoutAssignment(out, args).run()


class TestSteppedDepthwiseConv3d:
    """DEPTHWISE_CONV3D on a stepped group is computed directly, never repacked."""

    @pytest.fixture
    def backend(self):
        return "toy"

    def _depthwise(self, c, shape, k_dhw, stride, steps):
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=[c] + shape),
            "f": rng.normal(size=[c] + list(k_dhw)),
        }
        x = TensorTerm.Tensor("x", [c] + shape, True).group_pad(1, 3, steps=steps)
        f = TensorTerm.Tensor("f", [c] + list(k_dhw), False)
        return TensorTerm.depthwise_conv3d(x, f, stride, "same"), inputs

    def _check(self, c, shape, k_dhw, stride, steps, n, backend):
        from util.flatten_repack import depthwise_repack_plan_for_kernel

        args = get_default_args()
        args.n = n
        out, inputs = self._depthwise(c, shape, k_dhw, stride, steps)
        kernel = TestGroupPadToy._run(self, out, inputs, args, backend)
        assert kernel.op == KernelOp.DEPTHWISE_CONV3D
        assert depthwise_repack_plan_for_kernel(kernel) is None
        return kernel

    def test_parity_stride_1_2_2_drops_parity(self, backend):
        kernel = self._check(2, [3, 6, 6], (3, 3, 3), (1, 2, 2), (1, 2, 2), 256, backend)
        (group,) = set(_group_dims(kernel.layout))
        assert group.group == (3, 3, 3) and not group.is_stepped
        assert not _residue_dims(kernel.layout)

    def test_parity_in_ciphertexts(self, backend):
        self._check(2, [3, 6, 6], (3, 3, 3), (1, 2, 2), (1, 2, 2), 32, backend)

    def test_parity_odd_extents(self, backend):
        kernel = self._check(2, [3, 7, 5], (3, 3, 3), (1, 2, 2), (1, 2, 2), 16, backend)
        assert {d.group for d in _group_dims(kernel.layout)} == {(3, 4, 3)}

    def test_stride1_keeps_stepped_layout(self, backend):
        kernel = self._check(2, [3, 6, 6], (3, 3, 3), (1, 1, 1), (1, 2, 2), 32, backend)
        assert kernel.layout.layout_str() == kernel.cs[0].layout.layout_str()

    def test_unaligned_strides(self, backend):
        """Stride 2 on T too, but the group only steps H and W."""
        self._check(2, [4, 6, 6], (3, 3, 3), (2, 2, 2), (1, 2, 2), 32, backend)

    def test_parity_needs_one_rotation_per_tap(self):
        """Edward's 5x3x3 stride-(1,2,2) depthwise: 45 taps, one rotation each
        except the unshifted tap, and no repack (a plain group needs one)."""
        rots = {}
        for steps in (None, (1, 2, 2)):
            out, _ = self._depthwise(4, [5, 14, 14], (5, 3, 3), (1, 2, 2), steps)
            _, rots[steps], muls = _kernel_ops(out, 4096, KernelOp.DEPTHWISE_CONV3D)
        assert rots[(1, 2, 2)] == 5 * 3 * 3 - 1
        assert rots[(1, 2, 2)] < rots[None]


class TestPermutedDepthwiseConv3d:
    """DEPTHWISE_CONV3D on a group over dims 1..3 in any order keeps that order."""

    ORDERS = [(2, 1, 3), (2, 3, 1), (3, 2, 1)]

    def _check(self, order, c, shape, k_dhw, stride, n, backend, steps=None):
        args = get_default_args()
        args.n = n
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=[c] + shape),
            "f": rng.normal(size=[c] + list(k_dhw)),
        }
        x = TensorTerm.Tensor("x", [c] + shape, True).group_pad(
            1, 3, steps=steps, order=order
        )
        f = TensorTerm.Tensor("f", [c] + list(k_dhw), False)
        out = TensorTerm.depthwise_conv3d(x, f, stride, "same")
        kernel = TestGroupPadToy._run(self, out, inputs, args, backend)
        (dw,) = [k for k in kernel.post_order() if k.op == KernelOp.DEPTHWISE_CONV3D]
        assert {d.dim for d in _group_dims(dw.layout)} == {tuple(order)}
        return dw

    @pytest.mark.parametrize("order", ORDERS)
    def test_stride1_keeps_input_layout(self, order, backend):
        dw = self._check(order, 2, [3, 5, 6], (3, 3, 3), 1, 256, backend)
        assert dw.layout.layout_str() == dw.cs[0].layout.layout_str()

    @pytest.mark.parametrize("order", ORDERS)
    def test_stride_1_2_2_repacks_to_smaller_group(self, order, backend):
        dw = self._check(order, 2, [3, 6, 8], (3, 3, 3), (1, 2, 2), 256, backend)
        (group,) = set(_group_dims(dw.layout))
        assert group.per_dim(group.group) == (3, 3, 4)

    @pytest.mark.parametrize("order", ORDERS)
    def test_stepped_stride_1_2_2_aligned(self, order, backend):
        from util.flatten_repack import depthwise_repack_plan_for_kernel

        dw = self._check(
            order, 2, [3, 6, 8], (3, 3, 3), (1, 2, 2), 256, backend, steps=(1, 2, 2)
        )
        assert depthwise_repack_plan_for_kernel(dw) is None
        (group,) = set(_group_dims(dw.layout))
        assert group.per_dim(group.group) == (3, 3, 4) and not group.is_stepped

    def test_stride_1_2_2_group_spans_ciphertexts(self, backend):
        self._check((3, 1, 2), 2, [3, 6, 7], (3, 3, 3), (1, 2, 2), 64, backend)

    def test_expand_depthwise_project(self, backend):
        """Edward_example structure with a permuted, stepped input group."""
        args = get_default_args()
        args.n = 1024
        c_in, c, c_out, t, h, w = 2, 4, 2, 3, 6, 6
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=(c_in, t, h, w)),
            "f1": rng.normal(size=(c, c_in, 1, 1, 1)),
            "f2": rng.normal(size=(c, 3, 3, 3)),
            "f3": rng.normal(size=(c_out, c, 1, 1, 1)),
        }
        x = TensorTerm.Tensor("x", [c_in, t, h, w], True).group_pad(
            steps=(1, 2, 2), order=(2, 3, 1)
        )
        f1 = TensorTerm.Tensor("f1", [c, c_in, 1, 1, 1], False)
        f2 = TensorTerm.Tensor("f2", [c, 3, 3, 3], False)
        f3 = TensorTerm.Tensor("f3", [c_out, c, 1, 1, 1], False)
        out = TensorTerm.conv3d(x, f1, 1, "same")
        out = TensorTerm.depthwise_conv3d(out, f2, (1, 2, 2), "same")
        out = TensorTerm.conv3d(out, f3, 1, "same")
        kernel = TestGroupPadToy._run(self, out, inputs, args, backend)
        assert {d.dim for d in _group_dims(kernel.layout)} == {(2, 3, 1)}


class TestSteppedAdd:

    @pytest.fixture
    def backend(self):
        return "toy"
    def test_secret_plus_secret(self, backend):
        args = get_default_args()
        args.n = 256
        shape = [2, 3, 6, 6]
        rng = np.random.default_rng(0)
        inputs = {"x": rng.normal(size=shape), "y": rng.normal(size=shape)}
        x = TensorTerm.Tensor("x", shape, True).group_pad(1, 3, steps=(1, 2, 2))
        y = TensorTerm.Tensor("y", shape, True).group_pad(1, 3, steps=(1, 2, 2))
        kernel = TestGroupPadToy._run(self, x + y, inputs, args, backend)
        assert kernel.op == KernelOp.ADD
        assert {d.steps for d in _group_dims(kernel.layout)} == {(1, 2, 2)}

    def test_bias_on_stepped_conv(self, backend):
        args = get_default_args()
        args.n = 256
        shape = [3, 6, 6]
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=[2] + shape),
            "f": rng.normal(size=(4, 2, 1, 1, 1)),
            "b": rng.normal(size=[4] + shape),
        }
        x = TensorTerm.Tensor("x", [2] + shape, True).group_pad(1, 3, steps=(1, 2, 2))
        f = TensorTerm.Tensor("f", [4, 2, 1, 1, 1], False)
        b = TensorTerm.Tensor("b", [4] + shape, False)
        out = TensorTerm.conv3d(x, f, 1, "same") + b
        kernel = TestGroupPadToy._run(self, out, inputs, args, backend)
        assert {d.steps for d in _group_dims(kernel.layout)} == {(1, 2, 2)}


class TestSteppedEdwardChain:
    """Edward_example with a parity group_pad: expand -> depthwise -> project."""

    @pytest.fixture
    def backend(self):
        return "toy"

    @pytest.mark.parametrize("n", [256, 1024])
    def test_expand_depthwise_project(self, backend, n):
        args = get_default_args()
        args.n = n
        c_in, c, c_out, t, h, w = 2, 4, 3, 3, 6, 6
        rng = np.random.default_rng(0)
        shapes = {
            "x": [c_in, t, h, w],
            "fe": [c, c_in, 1, 1, 1],
            "be": [c, t, h, w],
            "fd": [c, t, 3, 3],
            "bd": [c, t, h // 2, w // 2],
            "fp": [c_out, c, 1, 1, 1],
            "bp": [c_out, t, h // 2, w // 2],
        }
        inputs = {k: rng.normal(size=s) for k, s in shapes.items()}
        T = {k: TensorTerm.Tensor(k, s, k == "x") for k, s in shapes.items()}
        x = T["x"].group_pad(1, 3, steps=(1, 2, 2))
        expand = TensorTerm.conv3d(x, T["fe"], 1, "same") + T["be"]
        depth = TensorTerm.depthwise_conv3d(expand, T["fd"], (1, 2, 2), "same") + T["bd"]
        out = TensorTerm.conv3d(depth, T["fp"], 1, "same") + T["bp"]
        kernel = TestGroupPadToy._run(self, out, inputs, args, backend)
        (dw,) = [k for k in kernel.post_order() if k.op == KernelOp.DEPTHWISE_CONV3D]
        assert {d.steps for d in _group_dims(dw.cs[0].layout)} == {(1, 2, 2)}
        assert not any(d.is_stepped for d in _group_dims(dw.layout))
        assert not _residue_dims(dw.layout)
