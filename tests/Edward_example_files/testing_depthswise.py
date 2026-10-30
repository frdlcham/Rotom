import argparse, sys, numpy as np
from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from lower.lower import Lower
from util.layout_util import apply_layout

#  Takes channels, spatial size and n as arguments
T, C, H, KT, K = 5, int(sys.argv[1]), int(sys.argv[2]), 5, 3 # assumes W = H, temporal kernel size 5, spatial kernel size 3
Hp = 1 << (H - 1).bit_length()   # pad spatial to power of 2
Hop = Hp // 2
TC = T * C
TCp = 1 << (TC - 1).bit_length()          # pad stacked channels to power of 2
rng = np.random.default_rng(0)
x = rng.normal(size=(T, H, H, C))          # NHWC frames
w = rng.normal(size=(KT, K, K, C))         # temporal depthwise kernel
bias = rng.normal(size=C)

# reference: depthwise 3D conv, temporal 'same' (pad 2), spatial stride 2 pad 1 (PyTorch-style)
Ho = (H + 2 - K) // 2 + 1
xp = np.pad(x, ((2, 2), (1, 1), (1, 1), (0, 0)))
ref = np.zeros((T, Ho, Ho, C))
for t in range(T):
    for i in range(Ho):
        for j in range(Ho):
            ref[t, i, j] = np.einsum("abcd,abcd->d", xp[t:t+KT, 2*i:2*i+K, 2*j:2*j+K], w) + bias

# stack frames into channels: ch = t*C + c
xs = np.zeros((TCp, Hp, Hp)); xs[:TC, :H, :H] = x.transpose(0, 3, 1, 2).reshape(TC, H, H)
W = np.zeros((TCp, TCp, K, K))
for t in range(T):
    for s in range(T):
        dt = s - t + KT // 2
        if 0 <= dt < KT:
            for c in range(C):
                W[t*C + c, s*C + c] = w[dt, :, :, c]
B = np.zeros((TCp, Hop, Hop)); B[:TC] = np.tile(bias, T)[:, None, None]

a = TensorTerm.Tensor("x", [TCp, Hp, Hp], True)
b = TensorTerm.Tensor("w", [TCp, TCp, K, K], False)
bb = TensorTerm.Tensor("bias", [TCp, Hop, Hop], False)
y = TensorTerm.conv2d(a, b, 2, "same") + bb
args = argparse.Namespace(n=int(sys.argv[3]), rolls=True, backend="toy")
kernel = LayoutAssignment(y, args).run()
inputs = {"x": xs, "w": W, "bias": B}
res = Toy(Lower(kernel).run(), inputs, args).run()
ev = y.eval(inputs)
print("layout:", kernel.layout)
print("rotom matches its evaluator:", all(np.allclose(p, q) for p, q in zip(apply_layout(ev, kernel.layout), res)))
out = ev[:TC, :Ho, :Ho].reshape(T, C, Ho, Ho).transpose(0, 2, 3, 1)
print("evaluator matches temporal depthwise reference:", np.allclose(out, ref))