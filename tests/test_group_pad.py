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
