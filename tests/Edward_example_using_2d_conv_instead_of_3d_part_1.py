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

# Conv2d in Rotom expects:
# Input shape: [C_in, H_in, W_in]
# Filter shape: [C_out, C_in, H_f, W_f]

# Can't do the entire (3d) convolution in one go. Can I do 184 separate 2D convolutions?
# THERE IS A 3D CONVOLUTION LAYOUT ASSIGNMENT, i was wrong

# Rotom's conv2d has no batch dimension and no grouped/depthwise kernel.

# SHAPES
T = 5
# H = 28
# W = 28
# Using H = W = 28, 
# program finds kernel [0:5:1];[G:32][1:16:1][G:2][2:16:1][G:2] 
# but cannot actually calculate the convolution since 28 is not a power of 2

# Using H = W = 32,
# program finds kernel [0:5:1];[G:32][1:16:1][G:2][2:16:1][G:2] (exact same kernel)
H = 32
W = 32
C = 184
STRIDE = 2

filter_shape = [5, 5, 3, 3]  # C_out, C_in, K, K
amount_of_conv_filters = C  # depthwise conv, so one filter per channel

# PADDING
H_P = 1 << (H - 1).bit_length()  # The power of 2 that is greater than or equal to H
W_P = 1 << (W - 1).bit_length()  # The power of 2 that is greater than or equal to W
H_O, W_O = H // STRIDE, W // STRIDE  # 14, 14 (logical output)
H_OP, W_OP = H_P // STRIDE, W_P // STRIDE  # 16, 16 (padded output)

# CONCRETE INPUTS AND KERNELS
concrete_input = np.random.randn(T, H, W, C)  # input frames
concrete_filter_depth_conv_1 = np.random.randn(filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]) # one of the 184 depthwise filters
concrete_input_conv_1 = concrete_input[:, :, :, 0]  # the first channel of the input, which is what the first depthwise filter will operate on

# Create tensor operations
filter_depth_conv_i = TensorTerm.Tensor("filter_depth_conv_i", [filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]], False)  # filter plaintext matrix, _i because it's the same dimensions for all filters
input = TensorTerm.Tensor("input", [T, H, W], True)  #input ciphertext matrix. Leave out the channels, each depthswise conv handles 1 channel.

output_conv_i = TensorTerm.conv2d(input, filter_depth_conv_i, stride=2, padding="same")
# Output of the first depthwise conv is [T, H/stride, W/stride] = [5, 14, 14]
# logically, output is [5, 28, 28, 184] in total

# Run layout assignment for first depthwise conv, lower to circuit IR, execute with toy backend
args = argparse.Namespace(n=32768, rolls=True, backend="toy")
kernel = LayoutAssignment(output_conv_i, args).run()
# Will layout be different depending on which filter is used? No, because filter always has same shape. So keep this out of loop

# Can we use this in any meaningful way?
# Because the layout assignments now are very very sparse, probably due to the fact that a 3D convolution is 'flattened' into different 2D convolutions.

# All layouts, inputs first and output last
for k in kernel.post_order():
    print(k)

# sometimes this gives the following error:
# AssertionError: File "util/layout_util.py", line 422, in get_cts_by_dim: assert dim in ct_dims failed
# I don't know what causes it. Rerunning a couple of times usually gets it to work.
# also it seems to happen less when debugging
circuit_ir = Lower(kernel).run()

# output_array = np.zeros((T, H_O, W_O, C))

for i in range(C):
    inputs = {"input": concrete_input[:, :, :, i], "filter_depth_conv_i": concrete_filter_depth_conv_1} # currently this line assumes all filters are identical
    results = Toy(circuit_ir, inputs, args).run()
    print("results: ",results)

    if (i>0):
        # only run the first depthwise conv for now
        # doing the rest just makes it take longer
        # and it's not like i'm putting it in an output array at the moment
        continue

    # THIS LINE DOESN'T WORK: output in ciphertext is not in the same layout as logical output.
    # output_array[:, :H_O, :W_O, i] = results

# Add results of all convs in one tensor and add bias.