"""
Test suite for the GROUP_PAD operation.

GROUP_PAD keeps a tensor's shape but lays out consecutive dims together in one
group dim padded as a whole, e.g. [C, 5, 14, 14] grouped over dims 1..3 takes
980 -> 1024 slots per channel instead of 8*16*16 = 2048.
"""

import numpy as np

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
