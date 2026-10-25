import argparse
import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from lower.lower import Lower

# Conv3d in Rotom expects:
#  input:  [C_in, D_in, H_in, W_in]   (secret)
#  filter: [C_out, C_in, Kd, Kh, Kw] (public)

# SHAPES - doesn't seem to matter if the dimensions are powers of 2 or not?
# layout always works, actual conv always fails with a mismatch between toy and the expected output
T = 5
H = 28
W = 28

C_in_1 = 32 # channels input to first conv
C_out_1 = 184

C_in_2 = C_out_1
C_out_2 = C_in_2 # 2nd conv keeps channels the same

C_in_3 = C_out_2
C_out_3 = 56

# T, H, and W, padded to powers of 2
# T = 8
# H = 32
# W = 32

STRIDE_conv_1 = 1
STRIDE_conv_2 = 2
STRIDE_conv_3 = 1

# pretty sure that 1st and 3rd conv don't go across frames
# unsure if I'm specifying these shapes correctly
filter_shape_1st_conv = [C_out_1, C_in_1, 1, 1, 1]  # C_out, C_in, Kd, Kh, Kw
filter_shape_2nd_conv = [1, 1, T, 3, 3]  # don't think this matches depthwise conv?
filter_shape_3rd_conv = [C_out_3, C_in_3, 1, 1, 1] # C_out, C_in, Kd, Kh, Kw

# # CONCRETE INPUTS AND KERNELS - only needed if operation is actually executed
# concrete_input = np.random.randn(C_in_1, T, H, W)  # input frames, [C_in, D_in, H_in, W_in]
# concrete_filter_expand_conv = np.random.randn(filter_shape_1st_conv[0], filter_shape_1st_conv[1], filter_shape_1st_conv[2], filter_shape_1st_conv[3], filter_shape_1st_conv[4])
# concrete_filter_depth_conv = np.random.randn(filter_shape_2nd_conv[0], filter_shape_2nd_conv[1], filter_shape_2nd_conv[2], filter_shape_2nd_conv[3], filter_shape_2nd_conv[4])
# concrete_filter_project_conv = np.random.randn(filter_shape_3rd_conv[0], filter_shape_3rd_conv[1], filter_shape_3rd_conv[2], filter_shape_3rd_conv[3], filter_shape_3rd_conv[4])
# concrete_bias_expand_conv = np.random.randn(C_out_1)  # bias for depthwise conv, one per output channel
# would then need to broadcast this bias to [C_out_1, T, H/STRIDE_conv_1, W/STRIDE_conv_1] for the first conv
# concrete_bias_depth_conv = np.random.randn(C_out_2)  # bias for depthwise conv, one per output channel
# then broadcast this bias to [C_out_2, T, H/(STRIDE_conv_2*STRIDE_conv_1), W/(STRIDE_conv_2*STRIDE_conv_1)] for the second conv
# concrete_bias_project_conv = np.random.randn(C_out_3)  # bias for depthwise conv, one per output channel
# then broadcast this bias to [C_out_3, T, H/(STRIDE_conv_3*STRIDE_conv_2*STRIDE_conv_1), W/(STRIDE_conv_3*STRIDE_conv_2*STRIDE_conv_1)] for the third conv

# Create tensor operations
input = TensorTerm.Tensor("input", [C_in_1, T, H, W], True)  #input ciphertext matrix
filter_expand_conv = TensorTerm.Tensor("filter_expand_conv", [filter_shape_1st_conv[0], filter_shape_1st_conv[1], filter_shape_1st_conv[2], filter_shape_1st_conv[3], filter_shape_1st_conv[4]], False)  # filter plaintext matrix
bias_expand_conv = TensorTerm.Tensor("bias_expand_conv", [C_out_1, T, H/STRIDE_conv_1, W/STRIDE_conv_1], False)  # bias plaintext vector
# having some issue with bias_expand_conv, prob due to the fact that conv2 is not proper depthwise

filter_depth_conv = TensorTerm.Tensor("filter_depth_conv", [filter_shape_2nd_conv[0], filter_shape_2nd_conv[1], filter_shape_2nd_conv[2], filter_shape_2nd_conv[3], filter_shape_2nd_conv[4]], False)  # filter plaintext matrix
bias_depth_conv = TensorTerm.Tensor("bias_depth_conv", [C_out_2, T, H/(STRIDE_conv_2*STRIDE_conv_1), W/(STRIDE_conv_2*STRIDE_conv_1)], False)  # bias plaintext vector

filter_project_conv = TensorTerm.Tensor("filter_project_conv", [filter_shape_3rd_conv[0], filter_shape_3rd_conv[1], filter_shape_3rd_conv[2], filter_shape_3rd_conv[3], filter_shape_3rd_conv[4]], False)  # filter plaintext matrix
bias_project_conv = TensorTerm.Tensor("bias_project_conv", [C_out_3, T, H/(STRIDE_conv_3*STRIDE_conv_2*STRIDE_conv_1), W/(STRIDE_conv_3*STRIDE_conv_2*STRIDE_conv_1)], False)  # bias plaintext vector

output_expand_conv = TensorTerm.conv3d(input, filter_expand_conv, stride=STRIDE_conv_1, padding="same") + bias_expand_conv
output_depth_conv = TensorTerm.conv3d(output_expand_conv, filter_depth_conv, stride=STRIDE_conv_2, padding="same") + bias_depth_conv
output_project_conv = TensorTerm.conv3d(output_depth_conv, filter_project_conv, stride=STRIDE_conv_3, padding="same") + bias_project_conv

# Run layout assignment for last conv
# Since the other convs are in the input of last conv? unsure if this works
args = argparse.Namespace(n=32768, rolls=True, backend="toy")
kernel = LayoutAssignment(output_project_conv, args).run()

# All layouts, inputs first and output last
for k in kernel.post_order():
    print(k)

# annotated output of this file:
# KernelOp.TENSOR: input [0:64:4];[0:4:1][1:8:1][2:32:1][3:32:1]
# KernelOp.REPLICATE: [R:184:125440][0:64:4];[0:4:1][1:8:1][2:32:1][3:32:1] -> input packing to #1

# KernelOp.PUNCTURED_TENSOR: filter_expand_conv [0:184:1][1:64:4];[1:4:1][R:8192:1]

# KernelOp.CONV3D: [0:184:1];[G:4][1:8:1][2:32:1][3:32:1]


# KernelOp.REPLICATE: [R:3:1][0:184:1];[G:4][1:8:1][2:32:1][3:32:1]
# KernelOp.REPLICATE: [R:9:1][0:184:1];[G:4][1:8:1][2:32:1][3:32:1]
# KernelOp.REPLICATE: [R:45:1][0:184:1];[G:4][1:8:1][2:32:1][3:32:1] -> input packing to #2
# That's equal to: [R:45:1][C:184:1];[G:4][T:8:1][H:32:1][W:32:1]
# Edward found:      [C:6][H:2][W:2];[H:14:1][W:14:1][T:5:1][C:32:1]

# KernelOp.PUNCTURED_TENSOR: filter_depth_conv [2:5:1][3:3:1][4:3:1][1:184:1];[G:4][R:8192:1]

# KernelOp.CONV3D: [G:4][1:4:1][G:2][2:16:1][G:2][3:16:1][G:2]


# KernelOp.REPLICATE: [R:56:108192];[G:4][1:4:1][G:2][2:16:1][G:2][3:16:1][G:2] -> input packing to #3

# KernelOp.PUNCTURED_TENSOR: filter_project_conv [0:56:1];[G:4][R:4:1][G:2][R:16:1][G:2][R:16:1][G:2]

# KernelOp.CONV3D: [0:56:1];[G:4][1:4:1][G:2][2:16:1][G:2][3:16:1][G:2]