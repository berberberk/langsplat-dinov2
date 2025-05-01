import matplotlib.pyplot as plt
import numpy as np

with open("/home/zagajnovni/semantic-splatting/DinoSplat/lerf_ovs/ramen/talk2dino_features/frame_00010.npz", "rb") as f:
    mat = np.load(f)["arr_0"]

plt.imshow(mat * 0.5 + 0.5)

plt.savefig("goida10.png")