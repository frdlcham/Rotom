import argparse
import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from lower.lower import Lower

# Depthwise conv, stride 2:
#   input  [T=5, 28, 28, C=184]  (NHWC frames)
#   weight [KT=5, 3, 3, C=184] + bias [184]
#   output [5, 14, 14, 184]

# Conv3d in Rotom expects:
#  input:  [C_in, D_in, H_in, W_in]   (secret)
#  filter: [C_out, C_in, Kd, Kh, Kw] (public)

# SHAPES - doesn't seem to matter if the dimensions are powers of 2 or not?
# layout always works, actual conv always fails with a mismatch between toy and the expected output
T = 5
H = 28
W = 28
C = 184

# T = 8
# H = 32
# W = 32
# C = 256
STRIDE = 2

filter_shape = [1, 1, T, 3, 3]  # C_out, C_in, Kd, Kh, Kw

# PADDING
H_P = 1 << (H - 1).bit_length()  # The power of 2 that is greater than or equal to H
W_P = 1 << (W - 1).bit_length()  # The power of 2 that is greater than or equal to W
H_O, W_O = H // STRIDE, W // STRIDE  # 14, 14 (logical output)
H_OP, W_OP = H_P // STRIDE, W_P // STRIDE  # 16, 16 (padded output)

# CONCRETE INPUTS AND KERNELS
concrete_input = np.random.randn(C, T, H, W)  # input frames, [C_in, D_in, H_in, W_in]
concrete_filter_depth_conv = np.random.randn(filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3], filter_shape[4])
concrete_bias_depth_conv = np.random.randn(C)  # bias for depthwise conv, one per output channel

# Create tensor operations
filter_depth_conv = TensorTerm.Tensor("filter_depth_conv", [filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3], filter_shape[4]], False)  # filter plaintext matrix
input = TensorTerm.Tensor("input", [C, T, H, W], True)  #input ciphertext matrix
bias_depth_conv = TensorTerm.Tensor("bias_depth_conv", [C], False)  # bias plaintext vector

output_conv = TensorTerm.conv3d(input, filter_depth_conv, stride=STRIDE, padding="same") + bias_depth_conv
# logically, output is [5=T, 28=H, 28=W, 184=C] (or [184=C, 5=T, 28=H, 28=W] in Rotom's conv3d layout)

# Run layout assignment for first depthwise conv, lower to circuit IR, execute with toy backend
args = argparse.Namespace(n=32768, rolls=True, backend="toy")
kernel = LayoutAssignment(output_conv, args).run()

# All layouts, inputs first and output last
for k in kernel.post_order():
    print(k)

# circuit_ir = Lower(kernel).run()
# inputs = {"input": concrete_input, "filter_depth_conv": concrete_filter_depth_conv, "bias_depth_conv": concrete_bias_depth_conv}
# results = Toy(circuit_ir, inputs, args).run() # This reports a mismatch between toy and the expected output
# print("results: ",results)
