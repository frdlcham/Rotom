# The convolutions and sizes here are based on (torchvision) ResNet-50
import argparse
import numpy as np

from assignment.assignment import LayoutAssignment
from frontends.tensor import TensorTerm
from ir.kernel_cost import KernelCost

# Description of what I'm trying to do (based on start of ResNet):
# #  | operator               | input          | weight         | output
# 1  | conv 7x7, stride 2     | 1x224x224x3    | 64x7x7x3       | 1x112x112x64
# m  | maxpool 3x3, stride 2  | 1x112x112x64   | –              | 1x56x56x64
# --
# 2  | reduce conv 1x1        | 1x56x56x64     | 64x64          | 1x56x56x64
# 3  | conv 3x3               | 1x56x56x64     | 64x3x3x64      | 1x56x56x64
# 4  | expand conv 1x1        | 1x56x56x64     | 256x64         | 1x56x56x256

# 5 takes m as input
# 5  | shortcut conv 1x1      | 1x56x56x64     | 256x64         | 1x56x56x256
# +1 | add(4, 5), ReLU        |                |                | 1x56x56x256
# --
# 6a | reduce conv 1x1        | 1x56x56x256    | 64x256         | 1x56x56x64
# 7a | conv 3x3               | 1x56x56x64     | 64x3x3x64      | 1x56x56x64
# 8a | expand conv 1x1        | 1x56x56x64     | 256x64         | 1x56x56x256

# +2 | add(8a, +1), ReLU      |                |                | 1x56x56x256
# --
# 6b | reduce conv 1x1        | 1x56x56x256    | 64x256         | 1x56x56x64
# 7b | conv 3x3               | 1x56x56x64     | 64x3x3x64      | 1x56x56x64
# 8b | expand conv 1x1        | 1x56x56x64     | 256x64         | 1x56x56x256

# +3 | add(8b, +2), ReLU      |                |                | 1x56x56x256
# -- 
# 9  | reduce conv 1x1        | 1x56x56x256    | 128x256        | 1x56x56x128
# 10 | conv 3x3, stride 2     | 1x56x56x128    | 128x3x3x128    | 1x28x28x128
# 11 | expand conv 1x1        | 1x28x28x128    | 512x128        | 1x28x28x512

# 12 takes +3 as input
# 12 | shortcut conv 1x1, s2  | 1x56x56x256    | 512x256        | 1x28x28x512
# +4 | add(11, 12), ReLU      |                |                | 1x28x28x512

# There's no maxpool in Rotom (because max isn't a polynomial)
# And I only really care about the cost of the convs
# So, to get the dimensions to match, I'm going to double the stride of the next conv

# To start with, I'm just going up to conv 4
# In part because I don't know what to do with the additions and ReLU's

# SHAPES
H = 224
W = 224

n = 65536
# conv2d doesn't seem to be able to handle H or W split across ciphertexts
# so n needs to be >= 65536

# Channels padded to powers of 2
C_in_1 = 4
C_in_1_unpadded = 3 # RGB
C_out_1 = 64

C_in_2 = C_out_1
C_out_2 = C_in_2

C_in_3 = C_out_2
C_out_3 = C_in_3

C_in_4 = C_out_3
C_out_4 = 256

STRIDE_conv_1 = 2
STRIDE_conv_2 = 2   # was 1; does the downsampling avg_pool used to do
STRIDE_conv_3 = 1
STRIDE_conv_4 = 1

# --- FILTERS ---
filter_shape_1st_conv = [C_out_1, C_in_1, 7, 7]  # C_out, C_in, f_h, f_w
filter_shape_2nd_conv = [C_out_2, C_in_2, 1, 1]  # C_out, C_in, f_h, f_w
filter_shape_3rd_conv = [C_out_3, C_in_3, 3, 3]  # C_out, C_in, f_h, f_w
filter_shape_4th_conv = [C_out_4, C_in_4, 1, 1]  # C_out, C_in, f_h, f_w

# --- CONCRETE INPUTS AND KERNELS --- only needed if operation is actually executed
concrete_input = np.concatenate([
    np.random.randn(C_in_1_unpadded, H, W),
    np.zeros((C_in_1 - C_in_1_unpadded, H, W))  # zero padding channel(s)
    ])  # input frames

concrete_filter_conv_1 = np.random.randn(*filter_shape_1st_conv)
concrete_filter_conv_2 = np.random.randn(*filter_shape_2nd_conv)
concrete_filter_conv_3 = np.random.randn(*filter_shape_3rd_conv)
concrete_filter_conv_4 = np.random.randn(*filter_shape_4th_conv)

# --- CREATE TENSOR OPERATIONS ---
input = TensorTerm.Tensor("input", [C_in_1, H, W], True)  # input ciphertext matrix

filter_conv_1 = TensorTerm.Tensor("filter_conv_1", filter_shape_1st_conv, False)  # filter plaintext matrix
filter_conv_2 = TensorTerm.Tensor("filter_conv_2", filter_shape_2nd_conv, False)  # filter plaintext matrix
filter_conv_3 = TensorTerm.Tensor("filter_conv_3", filter_shape_3rd_conv, False)  # filter plaintext matrix
filter_conv_4 = TensorTerm.Tensor("filter_conv_4", filter_shape_4th_conv, False)  # filter plaintext matrix

# --- EXPAND CONV ---
output_conv_1 = TensorTerm.conv2d(input, filter_conv_1, stride=STRIDE_conv_1, padding="same")
output_conv_2 = TensorTerm.conv2d(output_conv_1, filter_conv_2, stride=STRIDE_conv_2, padding="same")
output_conv_3 = TensorTerm.conv2d(output_conv_2, filter_conv_3, stride=STRIDE_conv_3, padding="same")
output_conv_4 = TensorTerm.conv2d(output_conv_3, filter_conv_4, stride=STRIDE_conv_4, padding="same")

# --- LAYOUT ASSIGNMENT ---

# Run layout assignment for last conv
# Since the other convs are in the input of last conv
args = argparse.Namespace(n=n, rolls=True, backend="toy")

kernel = LayoutAssignment(output_conv_4, args).run()

# --- KERNEL COST ---
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

# the grouped padding doesn't seem like it's going to save anything here
# and only the first conv really has stride 2 (and could thus benefit from steps)
# think I've picked a bad example