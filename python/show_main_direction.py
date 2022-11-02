import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray

with open('/home/tomokimori/CLionProjects/3dreconGPU/volume_bin/cfrp_xyz3/PCA/CF_MAIND_X_256x256x256.raw') as f:
    u_raw = np.fromfile(f, dtype=np.float32)

with open('/home/tomokimori/CLionProjects/3dreconGPU/volume_bin/cfrp_xyz3/PCA/CF_MAIND_Y_256x256x256.raw') as f:
    v_raw = np.fromfile(f, dtype=np.float32)

with open('/home/tomokimori/CLionProjects/3dreconGPU/volume_bin/cfrp_xyz3/PCA/CF_MAIND_Z_256x256x256.raw') as f:
    w_raw = np.fromfile(f, dtype=np.float32)

num_voxel = 256
size = 32
skip = int(num_voxel / size)
eps = 1e-20

# !! x->y, y->z, z->x
y, z, x = np.meshgrid(np.linspace(0, num_voxel, size), np.linspace(0, num_voxel, size), np.linspace(0, num_voxel, size))

# u = u_raw.reshape([num_voxel, num_voxel, num_voxel])[::skip, ::skip, ::skip]
u = u_raw.reshape([num_voxel, num_voxel, num_voxel])
v = v_raw.reshape([num_voxel, num_voxel, num_voxel])
w = w_raw.reshape([num_voxel, num_voxel, num_voxel])

padding = np.zeros_like(u)
padding[50:216, 50:216, 50:216] = 1.0
# padding[135:155, 120:140, 130:150] = 1.0
u = u * padding
v = v * padding
w = w * padding

u = u[::skip, ::skip, ::skip]
v = v[::skip, ::skip, ::skip]
w = w[::skip, ::skip, ::skip]

eps2 = 0.01
uvw = u.reshape([-1, 1]) + v.reshape([-1, 1]) + w.reshape([-1, 1])
judge = np.where((np.abs(u.reshape([-1, 1])) < eps2) & (np.abs(v.reshape([-1, 1])) < eps2) & (np.abs(w.reshape([-1, 1])) < eps2), 0, 1)

judge = np.concatenate((np.concatenate((np.ones([size**3, 3]), judge), axis=1), np.ones([2 * size**3, 4])), axis=0)

"""
u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
     np.sin(np.pi * z))
"""

# Color by azimuthal angle
"""
r = np.concatenate((np.abs(u / np.sqrt(u**2 + v**2 + w**2)).reshape([-1, 1]), tmp), axis=1).reshape([-1, 1])
g = np.concatenate((np.abs(v / np.sqrt(u**2 + v**2 + w**2)).reshape([-1, 1]), tmp), axis=1).reshape([-1, 1])
b = np.concatenate((np.abs(w / np.sqrt(u**2 + v**2 + w**2)).reshape([-1, 1]), tmp), axis=1).reshape([-1, 1])
a = np.concatenate((np.ones(size**3).reshape([-1, 1]), np.zeros([size**3, 2])), axis=1).reshape([-1, 1])
"""
u += eps
v += eps
w += eps

r = np.abs(w / (np.sqrt(u**2 + v**2 + w**2))).reshape([-1, 1])
g = np.abs(v / (np.sqrt(u**2 + v**2 + w**2))).reshape([-1, 1])
b = np.abs(u / (np.sqrt(u**2 + v**2 + w**2))).reshape([-1, 1])
a = np.ones_like(r)

tmp = np.zeros([1, 4])
tmp = np.tile(tmp, (2 * size**3, 1))
rgba = np.concatenate((np.concatenate((r, g, b, a), axis=1), tmp), axis=0)
rgba = rgba * judge

# erase the top of arrow
"""
tmp = np.concatenate((np.ones((1, 4)), np.zeros((2, 4)))).reshape((3, 4))
tmp = np.tile(tmp, (size**3, 1))
"""

ax = plt.figure().add_subplot(projection='3d')
ax.set(xlabel='X', ylabel='Y', zlabel='Z')
ax.quiver(x, y, z, u, v, w, color=rgba, length=3, normalize=True)

plt.show()
