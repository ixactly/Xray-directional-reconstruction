import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

with open('/home/tomokimori/CLionProjects/3dreconGPU/volume_bin/oilpan/pca/OilPan-x-400x400x400-6.892175um.raw') as f:
    u_raw = np.fromfile(f, dtype=np.float32)

with open('/home/tomokimori/CLionProjects/3dreconGPU/volume_bin/oilpan/pca/OilPan-y-400x400x400-6.892175um.raw') as f:
    v_raw = np.fromfile(f, dtype=np.float32)

with open('/home/tomokimori/CLionProjects/3dreconGPU/volume_bin/oilpan/pca/OilPan-z-400x400x400-6.892175um.raw') as f:
    w_raw = np.fromfile(f, dtype=np.float32)

num_voxel = 400 # voxel suu
bin = 7
size = int(num_voxel / bin) + 1 # hyouzisuu huyasu
eps = 1e-20

y, z, x = np.meshgrid(np.linspace(0, num_voxel, size), np.linspace(0, num_voxel, size), np.linspace(0, num_voxel, size), indexing='xy')

u = u_raw.reshape([num_voxel, num_voxel, num_voxel], order='C')
v = v_raw.reshape([num_voxel, num_voxel, num_voxel], order='C')
w = w_raw.reshape([num_voxel, num_voxel, num_voxel], order='C')

padding = np.zeros_like(u)
# z, y, x
# padding[40:180, 40:165, 30:150] = 1.0
padding[:420, 30:, 5:450] = 1.0
u = u * padding
v = v * padding
w = w * padding

u = u[::bin, ::bin, ::bin]
v = v[::bin, ::bin, ::bin]
w = w[::bin, ::bin, ::bin]

# thresholding
eps2 = 0.05 #ika kirisuteru
uvw = u.reshape([-1, 1]) + v.reshape([-1, 1]) + w.reshape([-1, 1])
judge = np.where((np.abs(u.reshape([-1, 1])) < eps2) & (np.abs(v.reshape([-1, 1])) < eps2) & (np.abs(w.reshape([-1, 1])) < eps2), 0, 1)
# judge = np.where((np.abs(u.reshape([-1, 1])) + np.abs(v.reshape([-1, 1])) + np.abs(w.reshape([-1, 1]))) / 3 < eps2, 0, 1)

judge = np.concatenate((np.concatenate((np.ones([size ** 3, 3]), judge), axis=1), np.ones([2 * size ** 3, 4])), axis=0)

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

r = np.abs(u / (np.sqrt(u**2 + v**2 + w**2))).reshape([-1, 1])
g = np.abs(v / (np.sqrt(u**2 + v**2 + w**2))).reshape([-1, 1])
b = np.abs(w / (np.sqrt(u**2 + v**2 + w**2))).reshape([-1, 1])
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

""" show
ax = plt.figure().add_subplot(projection='3d')
ax.set(xlabel='X', ylabel='Y', zlabel='Z')
# ax.set_facecolor((0, 0, 0, 1))
ax.w_xaxis.set_pane_color((0., 0., 0., 1.))
ax.w_yaxis.set_pane_color((0., 0., 0., 1.))
ax.w_zaxis.set_pane_color((0., 0., 0., 1.))
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.quiver(x, y, z, u, v, w, color=rgba, length=5, normalize=True)
"""

# sphere
# 半径を指定
r = 1.0

# ラジアンを作成
res = 80
theta = np.linspace(start=0.0, stop=np.pi, num=res)
phi = np.linspace(start=0.0, stop=2.0*np.pi, num=res)

# 格子点を作成
T, U = np.meshgrid(theta, phi)
# print(T[:3, :3].round(2))
# print(U[:3, :3].round(2))

# 球面の座標を計算
X = r * np.sin(T) * np.cos(U)
Y = r * np.sin(T) * np.sin(U)
Z = r * np.cos(T)
alpha = np.ones(res * res)
XYZ = np.stack([X.flatten(), Y.flatten(),  Z.flatten()], axis=1)
# gif
frame_num = 4
length = 5
v_n = np.linspace(start=90.0, stop=90.0, num=frame_num+1)[:frame_num]
h_n = np.linspace(start=0.0, stop=360.0, num=frame_num+1)[:frame_num]

#fig, ax1, ax2 = plt.subplots(figsize=(15, 15), subplot_kw={'projection': '3d'}, constrained_layout=True)
fig = plt.figure(figsize=(10, 15), constrained_layout=True, facecolor='black')
ax_fiber = fig.add_subplot(1, 1, 1, projection='3d', facecolor='black')
ax_sphere = fig.add_subplot(5, 5, 16, projection='3d', facecolor='black')
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()

    # i番目の角度を取得
    v_ = v_n[i]
    h_ = h_n[i]

    ax_fiber.set_facecolor('black')
    ax_fiber.w_xaxis.set_pane_color((0., 0., 0., 1.))
    ax_fiber.w_yaxis.set_pane_color((0., 0., 0., 1.))
    ax_fiber.w_zaxis.set_pane_color((0., 0., 0., 1.))
    ax_fiber.set_xticks([])
    ax_fiber.set_yticks([])
    ax_fiber.set_zticks([])
    ax_fiber.quiver(y, x, z, v, u, w, color=rgba, length=length, normalize=True)
    ax_fiber.set_aspect('equal')
    ax_fiber.view_init(elev=v_, azim=h_)

    ax_sphere.scatter(Y, X, Z, color=(np.abs(XYZ))/r, alpha=1.0) # 散布図:(xyz座標の値により色付け)
    """
    ax.quiver([-1.5, 0, 0], [0, -1.5, 0], [0, 0, -1.5],
              [3, 0, 0], [0, 3, 0], [0, 0, 3],
              color='black', lw=2, arrow_length_ratio=0.0, zorder=-150*150*3) # x,y,z軸
    """

    ax_sphere.set_xticks([])
    ax_sphere.set_yticks([])
    ax_sphere.set_zticks([])
    ax_sphere.w_xaxis.set_pane_color((0., 0., 0., 1.))
    ax_sphere.w_yaxis.set_pane_color((0., 0., 0., 1.))
    ax_sphere.w_zaxis.set_pane_color((0., 0., 0., 1.))
    end = 0.5
    ax_sphere.quiver([-r, 0, 0], [0, -r, 0], [0, 0, -r],
              [-end, 0, 0], [0, -end, 0], [0, 0, -end],
              color='white', lw=2, zorder=-1e10) # x,y,z軸
    ax_sphere.quiver([r, 0, 0], [0, r, 0], [0, 0, r],
              [end, 0, 0], [0, end, 0], [0, 0, end],
              color='white', lw=2, zorder=-1e10)
    # ax_sphere.text(r+end, 0, end/2, r"$x$", fontfamily='serif', fontsize=15, fontstyle='italic', color='white')
    # ax_sphere.text(end/2, r+end, 0, r"$y$", fontfamily='serif', fontsize=15, fontstyle='italic', color='white')
    # ax_sphere.text(0, end/2, r+end, r"$z$", fontfamily='serif', fontsize=15, fontstyle='italic', color='white')

    ax_sphere.set_aspect('equal')
    ax_sphere.view_init(elev=v_, azim=h_)

update(0)
plt.show()


"""
ani = FuncAnimation(fig=fig, func=update, frames=frame_num, interval=800)
ani.save('gfrp_uct_with_sphere.gif')
"""


