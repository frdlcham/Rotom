from ir.he import HETerm
from lower.layout_cts import LayoutCiphertexts
from util.flatten_repack import repack_plan_for_kernel


def lower_flatten_repack(env, kernel):
    """Lower a FLATTEN_REPACK or GROUP_PAD kernel by masking, rotating and summing slots."""
    plan = repack_plan_for_kernel(kernel)
    cts = dict(env[kernel.cs[0]].cts)
    for stage in plan.stages:
        new_cts = {}
        for move in stage:
            ct = cts[move.src_ct]
            if move.mask is not None:
                ct = ct * HETerm.mask([move.mask.astype(int).tolist()])
            if move.rot:
                ct = ct << move.rot
            new_cts[move.dst_ct] = (
                new_cts[move.dst_ct] + ct if move.dst_ct in new_cts else ct
            )
        cts = new_cts

    # output ciphertexts that only hold padding
    any_ct = next(iter(cts.values()))
    zero = any_ct * HETerm.mask([[0] * kernel.layout.n])
    out = {i: cts.get(i, zero) for i in range(plan.num_out_ct)}
    return LayoutCiphertexts(layout=kernel.layout, cts=out)
