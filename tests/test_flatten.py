"""
Test suite for the FLATTEN operation.

FLATTEN merges consecutive dims and pads the merged dim as a whole
(group, then pad), e.g. [C, 5, 14, 14] -> [C, 980] -> padded [C, 1024].
"""

import numpy as np

from assignment.assignment import LayoutAssignment
from frontends.tensor import TensorTerm
from ir.kernel import KernelOp
from ir.analysis.shape import Shape
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.flatten_repack import flatten_repack_plan_for_kernel
from util.layout_util import apply_layout


def _shapes(term):
    shape_an = Shape(term)
    shape_an.run()
    return shape_an.shapes[term], shape_an.padded_shapes[term]


class TestFlattenShape:
    def test_group_then_pad(self):
        """5*14*14 = 980 pads to 1024, not 8*16*16 = 2048."""
        a = TensorTerm.Tensor("a", [4, 5, 14, 14], True)
        shape, padded = _shapes(a.flatten(1, 3))
        assert shape == [4, 980]
        assert padded == [4, 1024]

    def test_partial_flatten_keeps_trailing_dims(self):
        a = TensorTerm.Tensor("a", [3, 5, 6, 7], True)
        shape, padded = _shapes(a.flatten(0, 1))
        assert shape == [15, 6, 7]
        assert padded == [16, 8, 8]

    def test_eval_trims_padding_before_flatten(self):
        """2-D placeholders are padded by eval; flatten must use logical row-major order."""
        a = TensorTerm.Tensor("a", [3, 5], True)
        x = np.arange(15, dtype=np.float64).reshape(3, 5)
        out = a.flatten(0, 1).eval({"a": x})
        np.testing.assert_array_equal(out, x.reshape(-1))


class TestFlattenToy:
    def _run(self, tensor_ir, inputs, args, backend):
        expected = tensor_ir.eval(inputs)
        results, kernel = run_compiler_and_backend(tensor_ir, inputs, args, backend)
        assert_results_equal(apply_layout(expected, kernel.layout), results, backend)

    def test_flatten_then_pointwise_matmul(self, backend):
        """Pointwise (1x1x1) conv as matmul over a flattened [T, H, W] axis."""
        args = get_default_args()
        args.n = 256
        c_in, c_out, t, h, w = 4, 4, 3, 4, 4  # inner dims (H, W) are powers of 2

        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=(c_in, t, h, w)),
            "f": rng.normal(size=(c_out, c_in)),
        }
        x = TensorTerm.Tensor("x", [c_in, t, h, w], True)
        f = TensorTerm.Tensor("f", [c_out, c_in], False)
        self._run(f @ x.flatten(1, 3), inputs, args, backend)

    def test_flatten_padded_inner_dims(self, backend):
        """Padded inner dims (6 stored as 8) are repacked; spans multiple cts."""
        args = get_default_args()
        args.n = 256
        rng = np.random.default_rng(0)
        inputs = {"x": rng.normal(size=(2, 3, 6, 6))}
        x = TensorTerm.Tensor("x", [2, 3, 6, 6], True)
        self._run(x.flatten(1, 3), inputs, args, backend)

    def test_flatten_padded_keeps_trailing_dims(self, backend):
        args = get_default_args()
        args.n = 256
        rng = np.random.default_rng(0)
        inputs = {"x": rng.normal(size=(3, 5, 6, 7))}
        x = TensorTerm.Tensor("x", [3, 5, 6, 7], True)
        self._run(x.flatten(0, 1), inputs, args, backend)

    def test_flatten_padded_then_pointwise_matmul(self, backend):
        args = get_default_args()
        args.n = 256
        c_in, c_out = 2, 4
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.normal(size=(c_in, 3, 6, 6)),
            "f": rng.normal(size=(c_out, c_in)),
        }
        x = TensorTerm.Tensor("x", [c_in, 3, 6, 6], True)
        f = TensorTerm.Tensor("f", [c_out, c_in], False)
        self._run(f @ x.flatten(1, 3), inputs, args, backend)

    def test_flatten_padded_uses_staged_repack(self):
        """[4, 5, 14, 14] -> [4, 980]: staged per dim, far fewer rotations than one shot."""
        args = get_default_args()
        args.n = 4096
        x = TensorTerm.Tensor("x", [4, 5, 14, 14], True)
        kernel = LayoutAssignment(x.flatten(1, 3), args).run()
        assert kernel.op == KernelOp.FLATTEN_REPACK
        plan = flatten_repack_plan_for_kernel(kernel)
        assert len(plan.stages) > 1
        assert plan.ops()["rot"] < 5 * 14
