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
# Runtime might get a bit slow, but ultimately we don't really need to do the convolutions anyway.
# Just want to figure out layout stuff.
#
# Rotom's conv2d has no batch dimension and no grouped/depthwise kernel, so:
# - Frames and channels are stacked into one channel axis (ch = t*g + c), and the
#   temporal/depthwise structure is encoded as zeros in a dense filter.
# - Channels never mix, so the 184 channels are split into independent groups to
#   keep each conv2d small. Every group uses the same Rotom program.
# - Channel counts are padded to a power of two.
# - Spatial size is zero-padded 28 -> 32 because Rotom's stride-2 conv2d gives
#   wrong results for non-power-of-two spatial sizes. The extra outputs are dropped.
T, H, W, C = 5, 28, 28, 184
KT, K = 5, 3
STRIDE = 2
GROUP = 23  # channels per group; 8 groups of 23 -> 5 * 23 = 115 stacked channels

H_P = 1 << (H - 1).bit_length()  # 32
W_P = 1 << (W - 1).bit_length()  # 32
H_O, W_O = H // STRIDE, W // STRIDE  # 14, 14 (logical output)
H_OP, W_OP = H_P // STRIDE, W_P // STRIDE  # 16, 16 (padded output)
TG_P = 1 << (T * GROUP - 1).bit_length()  # 128

# Example data (replace with real values)
x = np.random.randn(T, H, W, C)  # input frames
w = np.random.randn(KT, K, K, C)  # depthwise temporal kernel
bias = np.random.randn(C)


def pack_group(c0, g):
    """Builds stacked input, dense filter and bias for channels [c0, c0 + g)."""
    # input: [T, H, W, g] -> [T*g, H, W], zero-padded to [TG_P, H_P, W_P]
    xs = np.zeros((TG_P, H_P, W_P))
    xs[: T * g, :H, :W] = x[:, :, :, c0 : c0 + g].transpose(0, 3, 1, 2).reshape(T * g, H, W)

    # filter: output frame t, channel c reads input frame s, channel c
    # with temporal tap s - t + KT//2 ("same" padding over time)
    ws = np.zeros((TG_P, TG_P, K, K))
    for t in range(T):
        for s in range(T):
            dt = s - t + KT // 2
            if 0 <= dt < KT:
                for c in range(g):
                    ws[t * g + c, s * g + c] = w[dt, :, :, c0 + c]

    # bias, broadcast over frames and space
    bs = np.zeros((TG_P, H_OP, W_OP))
    bs[: T * g] = np.tile(bias[c0 : c0 + g], T)[:, None, None]
    return xs, ws, bs


# Create tensor operations (same program for every group)
input = TensorTerm.Tensor("input", [TG_P, H_P, W_P], True)  # input ciphertext
filter_dw = TensorTerm.Tensor("filter_dw", [TG_P, TG_P, K, K], False)  # filter plaintext
bias_t = TensorTerm.Tensor("bias", [TG_P, H_OP, W_OP], False)  # bias plaintext

output_1 = TensorTerm.conv2d(input, filter_dw, stride=STRIDE, padding="same") + bias_t
# logically, output is [TG_P, 16, 16]; [: T*g, :14, :14] holds the real values

# Run layout assignment, lower to circuit IR, execute with toy backend
args = argparse.Namespace(n=32768, rolls=True, backend="toy")
kernel = LayoutAssignment(output_1, args).run()

print("kernel: ",kernel)


circuit_ir = Lower(kernel).run()

output = np.zeros((T, H_O, W_O, C))
for c0 in range(0, C, GROUP):
    g = min(GROUP, C - c0)
    xs, ws, bs = pack_group(c0, g)
    inputs = {"input": xs, "filter_dw": ws, "bias": bs}
    results = Toy(circuit_ir, inputs, args).run()  # ciphertexts in kernel.layout

    # Unpack this group's result back to [T, H_O, W_O, g] (plaintext evaluation)
    y = output_1.eval(inputs)[: T * g, :H_O, :W_O]
    output[:, :, :, c0 : c0 + g] = y.reshape(T, g, H_O, W_O).transpose(0, 2, 3, 1)

print("output shape: ", output.shape)  # (5, 14, 14, 184)




# Example 1: matrix-vector multiplication
# # Create tensor operations
# a = TensorTerm.Tensor("a", [64, 64], True)  # 64x64 ciphertext matrix
# b = TensorTerm.Tensor("b", [64], False)  # 64-element plaintext vector
# c = a @ b  # Matrix-vector multiplication

# # Run layout assignment, lower to circuit IR, execute with toy backend
# args = argparse.Namespace(n=4096, rolls=True, backend="toy")
# kernel = LayoutAssignment(c, args).run()
# circuit_ir = Lower(kernel).run()
# inputs = {"a": np.random.randn(64, 64), "b": np.random.randn(64)}
# results = Toy(circuit_ir, inputs, args).run()


# Example 2: 2D Convolution
# input_tensor = TensorTerm.Tensor("input", [32, 32, 3], True)
# filter_tensor = TensorTerm.Tensor("filter", [3, 3, 3, 64], False)

# output = TensorTerm.conv2d(input_tensor, filter_tensor, 
#                             stride=1, padding="same")