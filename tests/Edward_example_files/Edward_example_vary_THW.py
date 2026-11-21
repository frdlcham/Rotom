import argparse

from assignment.assignment import LayoutAssignment
from frontends.tensor import TensorTerm
from ir.kernel_cost import KernelCost

from util.util import ceil_div, round_to_ceiling_power_of_2

# Conv3d in Rotom expects:
#  input:  [C_in, D_in, H_in, W_in]   (secret)
#  filter: [C_out, C_in, Kd, Kh, Kw] (public)

# SHAPES
T_array = [3, 5] # T = 1 causes an issue with the depthwise conv (since it doesn't really make sense to do a depthwise conv with only 1 frame, and the code doesn't handle it)
H_W_array = [3, 6, 8, 12, 14]

# need ceil_div to deal with H = W not divisible by STRIDE_conv_2[1] = STRIDE_conv_2[2]

# CHANNELS
C_in_1 = 32 # channels input to first conv
C_out_1 = 184

C_in_2 = C_out_1
C_out_2 = C_in_2 # 2nd conv keeps channels the same

C_in_3 = C_out_2
C_out_3 = 56

# Rules:

# If you have non-power-of-two T,H,W dimensions (or only H and W, or some other subset of them), 
# you can first group those and then pad, which can make padding more efficient.

# If you have a conv with stride = (i, j, k), it can be useful to group_pad(1, 3, steps=(i, j, k)), 
# because this then does the parity layout stuff. 
# are these steps related to group_pad?
# could also not group, just change the steps for the specific dimensions

# I don't think these steps all need to be powers of two, 
# since they become the middle number in the outer brackets: [1:i:1][2:j:1][3:k:1];[1-3@TxHxW:power_of_2:(i,j,k)].
# Currently though, the code does not support non-power-of-two strides. 
# And conv3d only supports stride = 1 or stride = 2.


STRIDE_conv_1 = 1
STRIDE_conv_2 = (1, 2, 2)
STRIDE_conv_3 = 1

n = 32768 # number of slots in ciphertext

# --- FILTERS ---
filter_shape_1st_conv = [C_out_1, C_in_1, 1, 1, 1]  # C_out, C_in, Kd, Kh, Kw
# filter 2nd conv defined in loop below since it depends on T
filter_shape_3rd_conv = [C_out_3, C_in_3, 1, 1, 1] # C_out, C_in, Kd, Kh, Kw

for i in range(len(T_array)):
    T = T_array[i % len(T_array)]
    filter_shape_2nd_conv = [C_out_2, T, 3, 3] # C, Kd, Kh, Kw

    for j in range(len(H_W_array)):
        H = W = H_W_array[j % len(H_W_array)]

        print("--- T = {}, H = {}, W = {} ---".format(T, H, W))

        # --- CREATE TENSOR OPERATIONS ---
        # group_pad here pads 5*28*28 = 3920 to 4096
        input = TensorTerm.Tensor("input", [C_in_1, T, H, W], True)  # input ciphertext matrix

        # if this is the case, there is no need to group_pad and groupd_pad will probably make things worse
        if (round_to_ceiling_power_of_2(T*H*W) == round_to_ceiling_power_of_2(T)*round_to_ceiling_power_of_2(H)*round_to_ceiling_power_of_2(W)):
            print("Expecting no benefit from group_pad for T = {}, H = {}, W = {}".format(T, H, W))
            # but the steps can still help optimimise
            # though then it's nearly identical to have a non-grouped layout with the correct steps for each dimension

        input_opt_order_steps = input.group_pad(order=(2, 3, 1), steps=STRIDE_conv_2)  # steps are in order 1, 2, 3
        input_opt_order = input.group_pad(order=(2, 3, 1))

        filter_expand_conv = TensorTerm.Tensor("filter_expand_conv", filter_shape_1st_conv, False)  # filter plaintext matrix
        bias_expand_conv = TensorTerm.Tensor("bias_expand_conv", [C_out_1, T, H, W], False)  # bias plaintext vector

        filter_depth_conv = TensorTerm.Tensor("filter_depth_conv", filter_shape_2nd_conv, False)  # filter plaintext matrix
        bias_depth_conv = TensorTerm.Tensor("bias_depth_conv", [C_out_2, ceil_div(T, STRIDE_conv_2[0]), ceil_div(H, STRIDE_conv_2[1]), ceil_div(W, STRIDE_conv_2[2])], False)  # bias plaintext vector

        filter_project_conv = TensorTerm.Tensor("filter_project_conv", filter_shape_3rd_conv, False)  # filter plaintext matrix
        bias_project_conv = TensorTerm.Tensor("bias_project_conv", [C_out_3, ceil_div(T, STRIDE_conv_2[0]), ceil_div(H, STRIDE_conv_2[1]), ceil_div(W, STRIDE_conv_2[2])], False)  # bias plaintext vector

        # --- EXPAND CONV s---
        output_expand_conv = TensorTerm.conv3d(input, filter_expand_conv, stride=STRIDE_conv_1, padding="same") + bias_expand_conv
        output_expand_conv_opt_order_steps = TensorTerm.conv3d(input_opt_order_steps, filter_expand_conv, stride=STRIDE_conv_1, padding="same") + bias_expand_conv
        output_expand_conv_opt_order = TensorTerm.conv3d(input_opt_order, filter_expand_conv, stride=STRIDE_conv_1, padding="same") + bias_expand_conv

        # --- DEPTH CONV ---
        output_depth_conv = TensorTerm.depthwise_conv3d(output_expand_conv, filter_depth_conv, STRIDE_conv_2, "same") + bias_depth_conv
        output_depth_conv_opt_order_steps = TensorTerm.depthwise_conv3d(output_expand_conv_opt_order_steps, filter_depth_conv, STRIDE_conv_2, "same") + bias_depth_conv
        output_depth_conv_opt_order = TensorTerm.depthwise_conv3d(output_expand_conv_opt_order, filter_depth_conv, STRIDE_conv_2, "same") + bias_depth_conv

        # --- PROJECT CONV ---
        output_project_conv = TensorTerm.conv3d(output_depth_conv, filter_project_conv, stride=STRIDE_conv_3, padding="same") + bias_project_conv
        output_project_conv_opt_order_steps = TensorTerm.conv3d(output_depth_conv_opt_order_steps, filter_project_conv, stride=STRIDE_conv_3, padding="same") + bias_project_conv
        output_project_conv_opt_order = TensorTerm.conv3d(output_depth_conv_opt_order, filter_project_conv, stride=STRIDE_conv_3, padding="same") + bias_project_conv

        # --- LAYOUT ASSIGNMENT ---

        # Run layout assignment for last conv
        # Since the other convs are in the input of last conv
        args = argparse.Namespace(n=n, rolls=True, backend="toy")

        # --- KERNEL COST ---
        kernel = LayoutAssignment(output_project_conv, args).run()
        print("--- KERNEL COST ---")
        # All layouts, inputs first and output last
        for k in kernel.post_order():
            print(k)

        kc = KernelCost(kernel, "lan") # same default network LayoutAssignment uses
        print("ops:   ", kc.total_operations()) # {'add': ..., 'mul': ..., 'rot': ...}
        print("op cost:  ", kc.total_cost() - kc.comm_cost())
        print("comm cost:", kc.comm_cost())
        print("total cost:", kc.total_cost())
        print("mult. depth:", kc.depth())

        # --- OPTIMIZED KERNEL COST ---
        kernel_opt_order_steps = LayoutAssignment(output_project_conv_opt_order_steps, args).run()
        print("--- OPTIMIZED KERNEL COST WITH ORDER (2, 3, 1) AND STEPS ---")
        # All layouts, inputs first and output last
        for k in kernel_opt_order_steps.post_order():
            print(k)

        kc_opt_order_steps = KernelCost(kernel_opt_order_steps, "lan") # same default network LayoutAssignment uses
        print("ops:   ", kc_opt_order_steps.total_operations()) # {'add': ..., 'mul': ..., 'rot': ...}
        print("op cost:  ", kc_opt_order_steps.total_cost() - kc_opt_order_steps.comm_cost())
        print("comm cost:", kc_opt_order_steps.comm_cost())
        print("total cost:", kc_opt_order_steps.total_cost())
        print("mult. depth:", kc_opt_order_steps.depth())

        kernel_opt_order = LayoutAssignment(output_project_conv_opt_order, args).run()
        print("--- OPTIMIZED KERNEL COST WITH ORDER (2, 3, 1) ---")
        # All layouts, inputs first and output last
        for k in kernel_opt_order.post_order():
            print(k)

        kc_opt_order = KernelCost(kernel_opt_order, "lan") # same default network LayoutAssignment uses
        print("ops:   ", kc_opt_order.total_operations()) # {'add': ..., 'mul': ..., 'rot': ...}
        print("op cost:  ", kc_opt_order.total_cost() - kc_opt_order.comm_cost())
        print("comm cost:", kc_opt_order.comm_cost())
        print("total cost:", kc_opt_order.total_cost())
        print("mult. depth:", kc_opt_order.depth())