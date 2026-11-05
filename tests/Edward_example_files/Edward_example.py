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

# C_in_1 = 32 # channels input to first conv
# C_out_1 = 184

# C_in_2 = C_out_1
# C_out_2 = C_in_2 # 2nd conv keeps channels the same

# C_in_3 = C_out_2
# C_out_3 = 56

# T, H, and W, padded to powers of 2
# T = 8
# H = 32
# W = 32

# Channels padded to powers of 2
C_in_1 = 32 # channels input to first conv
C_out_1 = 256

C_in_2 = C_out_1
C_out_2 = C_in_2 # 2nd conv keeps channels the same

C_in_3 = C_out_2
C_out_3 = 64

STRIDE_conv_1 = 1
STRIDE_conv_2 = (1, 2, 2)
STRIDE_conv_3 = 1

# --- FILTERS ---
filter_shape_1st_conv = [C_out_1, C_in_1, 1, 1, 1]  # C_out, C_in, Kd, Kh, Kw
filter_shape_2nd_conv = [C_out_2, T, 3, 3] # C, Kd, Kh, Kw
filter_shape_3rd_conv = [C_out_3, C_in_3, 1, 1, 1] # C_out, C_in, Kd, Kh, Kw

# --- CONCRETE INPUTS AND KERNELS --- only needed if operation is actually executed
concrete_input = np.random.randn(C_in_1, T, H, W)  # input frames, [C_in, D_in, H_in, W_in]
concrete_filter_expand_conv = np.random.randn(filter_shape_1st_conv[0], filter_shape_1st_conv[1], filter_shape_1st_conv[2], filter_shape_1st_conv[3], filter_shape_1st_conv[4])
concrete_filter_depth_conv = np.random.randn(filter_shape_2nd_conv[0], filter_shape_2nd_conv[1], filter_shape_2nd_conv[2], filter_shape_2nd_conv[3])
concrete_filter_project_conv = np.random.randn(filter_shape_3rd_conv[0], filter_shape_3rd_conv[1], filter_shape_3rd_conv[2], filter_shape_3rd_conv[3], filter_shape_3rd_conv[4])

concrete_bias_expand_conv  = np.broadcast_to(np.random.randn(C_out_1)[:, None, None, None], (C_out_1, T, H, W)).copy()
concrete_bias_depth_conv   = np.broadcast_to(np.random.randn(C_out_2)[:, None, None, None], (C_out_2, T, H//2, W//2)).copy()
concrete_bias_project_conv = np.broadcast_to(np.random.randn(C_out_3)[:, None, None, None], (C_out_3, T, H//2, W//2)).copy()

# --- CREATE TENSOR OPERATIONS ---
# group_pad here pads 5*28*28 = 3920 to 4096
input = TensorTerm.Tensor("input", [C_in_1, T, H, W], True).group_pad(1, 3, steps=(1, 2, 2))  #input ciphertext matrix
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

# --- LOWERING --- BE WARNED: THIS TAKES AGES 
circuit_ir = Lower(kernel).run()
inputs = {"input": concrete_input, "filter_expand_conv": concrete_filter_expand_conv, 
          "filter_depth_conv": concrete_filter_depth_conv, "filter_project_conv": concrete_filter_project_conv, 
          "bias_expand_conv": concrete_bias_expand_conv, "bias_depth_conv": concrete_bias_depth_conv, 
          "bias_project_conv": concrete_bias_project_conv}
results = Toy(circuit_ir, inputs, args).run() # this is the part that takes ages
print("results: ",results)