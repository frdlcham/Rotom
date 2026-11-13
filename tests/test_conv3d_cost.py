"""The CONV3D cost model counts exactly the HE ops of the lowered circuit.

``KernelCost`` counts a CONV3D kernel with ``lower.lower_conv3d.conv3d_ops``,
which mirrors ``lower_conv3d`` without building the circuit. Each case lowers a
program ending in a CONV3D and compares the model with the secret
ADD / MUL / ROT terms the CONV3D adds to the (optimized) circuit.
"""

import pytest

from assignment.assignment import LayoutAssignment
from frontends.tensor import TensorTerm
from ir.he import HEOp
from ir.kernel import KernelOp
from ir.kernel_cost import KernelCost
from lower.lower import Lower
from tests.test_util import get_default_args

_OPS = {HEOp.ADD: "add", HEOp.MUL: "mul", HEOp.ROT: "rot"}


def _reachable(cts):
    seen = set()
    for ct in cts:
        ct.helper_post_order(seen)
    return seen


def _lowered_conv3d_ops(kernel):
    """(lowered ops, modelled ops) of ``kernel``, which must be a CONV3D."""
    assert kernel.op == KernelOp.CONV3D
    env = Lower(kernel).run()
    new = _reachable(env[kernel].cts.values()) - _reachable(
        env[kernel.cs[0]].cts.values()
    )
    ops = {"add": 0, "mul": 0, "rot": 0}
    for t in new:
        if t.secret and t.op in _OPS:
            ops[_OPS[t.op]] += 1
    return ops, KernelCost(kernel, "lan").ops()


def _x(shape, group=None):
    x = TensorTerm.Tensor("x", shape, True)
    return x if group is None else x.group_pad(1, 3, **group)


def _conv(x, fshape, stride, padding="same"):
    return TensorTerm.conv3d(x, TensorTerm.Tensor("f", fshape, False), stride, padding)


CASES = {
    "1x1x1": (lambda: _conv(_x([4, 2, 4, 4]), [8, 4, 1, 1, 1], 1), 1024),
    "3x3x3": (lambda: _conv(_x([2, 4, 4, 4]), [4, 2, 3, 3, 3], 1), 1024),
    "3x3x3_stride2": (lambda: _conv(_x([2, 4, 8, 8]), [4, 2, 3, 3, 3], 2), 1024),
    "3x3x3_valid": (
        lambda: _conv(_x([1, 4, 4, 4]), [1, 1, 3, 3, 3], 1, "valid"),
        1024,
    ),
    "c_in_over_cts_3x3x3": (
        lambda: _conv(_x([16, 2, 4, 4]), [4, 16, 3, 3, 3], 1),
        256,
    ),
    "c_in_over_cts_1x1x1": (
        lambda: _conv(_x([16, 2, 4, 4]), [4, 16, 1, 1, 1], 1),
        128,
    ),
    "computed_input": (
        lambda: _conv(
            _x([4, 2, 4, 4]) * TensorTerm.Tensor("y", [4, 2, 4, 4], True),
            [8, 4, 3, 3, 3],
            1,
        ),
        512,
    ),
    "group_1x1x1": (lambda: _conv(_x([4, 3, 6, 6], {}), [8, 4, 1, 1, 1], 1), 256),
    "group_3x3x3": (lambda: _conv(_x([2, 3, 6, 6], {}), [4, 2, 3, 3, 3], 1), 256),
    "group_permuted_3x3x3": (
        lambda: _conv(_x([2, 3, 6, 6], {"order": (2, 1, 3)}), [4, 2, 3, 3, 3], 1),
        256,
    ),
    "group_stepped_1x1x1": (
        lambda: _conv(_x([4, 3, 6, 6], {"steps": (1, 2, 2)}), [8, 4, 1, 1, 1], 1),
        256,
    ),
    "group_stepped_3x3x3_stride2": (
        lambda: _conv(_x([2, 4, 8, 8], {"steps": (2, 2, 2)}), [4, 2, 3, 3, 3], 2),
        256,
    ),
    "group_stepped_c_in_over_cts": (
        lambda: _conv(_x([8, 3, 6, 6], {"steps": (1, 2, 2)}), [4, 8, 3, 3, 3], 1),
        64,
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_conv3d_cost_matches_lowered_circuit(name):
    build, n = CASES[name]
    args = get_default_args()
    args.n = n
    args.rolls = True
    kernel = LayoutAssignment(build(), args).run()
    lowered, modelled = _lowered_conv3d_ops(kernel)
    assert modelled == lowered
