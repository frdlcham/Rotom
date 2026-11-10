import argparse
import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from lower.lower import Lower

# Conv3d in Rotom expects:
#  input:  [C_in, D_in, H_in, W_in]   (secret)
#  filter: [C_out, C_in, Kd, Kh, Kw] (public)

# SHAPES
T = 5
H = 28
W = 28

# Channels padded to powers of 2
C_in_1 = 32 # channels input to first conv
C_out_1_unpadded = 184
C_out_1 = 256

C_in_2 = C_out_1
C_in_2_unpadded = C_out_1_unpadded
C_out_2_unpadded = C_in_2_unpadded
C_out_2 = C_in_2 # 2nd conv keeps channels the same

C_in_3 = C_out_2
C_in_3_unpadded = C_out_2_unpadded
C_out_3_unpadded = 56
C_out_3 = 64

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

# --- FILTERS ---
filter_shape_1st_conv = [C_out_1, C_in_1, 1, 1, 1]  # C_out, C_in, Kd, Kh, Kw
filter_shape_2nd_conv = [C_out_2, T, 3, 3] # C, Kd, Kh, Kw
filter_shape_3rd_conv = [C_out_3, C_in_3, 1, 1, 1] # C_out, C_in, Kd, Kh, Kw

# --- CONCRETE INPUTS AND KERNELS --- only needed if operation is actually executed
concrete_input = np.random.randn(C_in_1, T, H, W)  # input frames, [C_in, D_in, H_in, W_in]

concrete_filter_expand_conv = np.concatenate([
    np.random.randn(C_out_1_unpadded, filter_shape_1st_conv[1], filter_shape_1st_conv[2], filter_shape_1st_conv[3], filter_shape_1st_conv[4]),
    np.zeros((C_out_1-C_out_1_unpadded, filter_shape_1st_conv[1], filter_shape_1st_conv[2], filter_shape_1st_conv[3], filter_shape_1st_conv[4]))
    ])
concrete_filter_depth_conv = np.concatenate([
    np.random.randn(C_out_2_unpadded, filter_shape_2nd_conv[1], filter_shape_2nd_conv[2], filter_shape_2nd_conv[3]),
    np.zeros((C_out_2-C_out_2_unpadded, filter_shape_2nd_conv[1], filter_shape_2nd_conv[2], filter_shape_2nd_conv[3]))
    ])
concrete_filter_project_conv = np.concatenate([
    np.random.randn(C_out_3_unpadded, filter_shape_3rd_conv[1], filter_shape_3rd_conv[2], filter_shape_3rd_conv[3], filter_shape_3rd_conv[4]),
    np.zeros((C_out_3-C_out_3_unpadded, filter_shape_3rd_conv[1], filter_shape_3rd_conv[2], filter_shape_3rd_conv[3], filter_shape_3rd_conv[4]))
    ])

zero_pad_bias_expand_conv = np.concatenate([np.random.randn(C_out_1_unpadded), np.zeros(C_out_1-C_out_1_unpadded)])
zero_pad_bias_depth_conv = np.concatenate([np.random.randn(C_out_2_unpadded), np.zeros(C_out_2-C_out_2_unpadded)])
zero_pad_bias_project_conv = np.concatenate([np.random.randn(C_out_3_unpadded), np.zeros(C_out_3-C_out_3_unpadded)])

concrete_bias_expand_conv  = np.broadcast_to(zero_pad_bias_expand_conv[:, None, None, None], (C_out_1, T, H, W)).copy()
concrete_bias_depth_conv   = np.broadcast_to(zero_pad_bias_depth_conv[:, None, None, None], (C_out_2, T, H//2, W//2)).copy()
concrete_bias_project_conv = np.broadcast_to(zero_pad_bias_project_conv[:, None, None, None], (C_out_3, T, H//2, W//2)).copy()

# --- CREATE TENSOR OPERATIONS ---
# group_pad here pads 5*28*28 = 3920 to 4096
input = TensorTerm.Tensor("input", [C_in_1, T, H, W], True).group_pad(order=(2, 3, 1), steps=(1, 2, 2))  # steps are in order 1, 2, 3
filter_expand_conv = TensorTerm.Tensor("filter_expand_conv", filter_shape_1st_conv, False)  # filter plaintext matrix
bias_expand_conv = TensorTerm.Tensor("bias_expand_conv", [C_out_1, T, H, W], False)  # bias plaintext vector

filter_depth_conv = TensorTerm.Tensor("filter_depth_conv", filter_shape_2nd_conv, False)  # filter plaintext matrix
bias_depth_conv = TensorTerm.Tensor("bias_depth_conv", [C_out_2, T, H//2, W//2], False)  # bias plaintext vector

filter_project_conv = TensorTerm.Tensor("filter_project_conv", filter_shape_3rd_conv, False)  # filter plaintext matrix
bias_project_conv = TensorTerm.Tensor("bias_project_conv", [C_out_3, T, H//2, W//2], False)  # bias plaintext vector

# --- EXPAND CONV ---
output_expand_conv = TensorTerm.conv3d(input, filter_expand_conv, stride=STRIDE_conv_1, padding="same") + bias_expand_conv

# --- DEPTH CONV ---
output_depth_conv = TensorTerm.depthwise_conv3d(output_expand_conv, filter_depth_conv, STRIDE_conv_2, "same") + bias_depth_conv

# output_depth_conv = (
#     TensorTerm.depthwise_conv3d(output_expand_conv, filter_depth_conv, STRIDE_conv_2, "same")
#     + bias_depth_conv
# ).group_pad(1, 3)   # 5*14*14 = 980 -> 1024 per channel


# --- PROJECT CONV ---
output_project_conv = TensorTerm.conv3d(output_depth_conv, filter_project_conv, stride=STRIDE_conv_3, padding="same") + bias_project_conv

# --- LAYOUT ASSIGNMENT ---

# Run layout assignment for last conv
# Since the other convs are in the input of last conv
args = argparse.Namespace(n=32768, rolls=True, backend="toy")

kernel = LayoutAssignment(output_project_conv, args).run()

# All layouts, inputs first and output last
for k in kernel.post_order():
    print(k)

# --- LOWERING --- BE WARNED: THIS TAKES A WHILE
circuit_ir = Lower(kernel).run()
inputs = {"input": concrete_input, "filter_expand_conv": concrete_filter_expand_conv, 
          "filter_depth_conv": concrete_filter_depth_conv, "filter_project_conv": concrete_filter_project_conv, 
          "bias_expand_conv": concrete_bias_expand_conv, "bias_depth_conv": concrete_bias_depth_conv, 
          "bias_project_conv": concrete_bias_project_conv}
results = Toy(circuit_ir, inputs, args).run() # this is the part that takes ages
print("results: ",results)