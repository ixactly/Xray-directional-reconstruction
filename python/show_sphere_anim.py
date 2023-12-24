import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 半径を指定
r = 1.0

# ラジアンを作成
res = 150
t = np.linspace(start=0.0, stop=np.pi, num=res)
u = np.linspace(start=0.0, stop=2.0*np.pi, num=res)

# 格子点を作成
T, U = np.meshgrid(t, u)
# print(T[:3, :3].round(2))
# print(U[:3, :3].round(2))

# 球面の座標を計算
X = r * np.sin(T) * np.cos(U)
Y = r * np.sin(T) * np.sin(U)
Z = r * np.cos(T)
alpha = np.ones(res * res)
XYZ = np.stack([X.flatten(), Y.flatten(),  Z.flatten()], axis=1)
print(XYZ[:3, :3].round(2))
print(XYZ.shape)

frame_num = 15
v_n = np.linspace(start=30.0, stop=30.0, num=frame_num+1)[:frame_num]
h_n = np.linspace(start=0.0, stop=360.0, num=frame_num+1)[:frame_num]

fig, ax = plt.subplots(figsize=(8, 8), facecolor='white',
                       subplot_kw={'projection': '3d'})
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()

    # i番目の角度を取得
    v = v_n[i]
    h = h_n[i]

    # ax.plot_wireframe(X, Y, Z, alpha=0.5) # くり抜き曲面
    # ax.plot_surface(X, Y, Z, cmap='jet', color=(XYZ+r)*0.5/r, alpha=0.5) # 塗りつぶし曲面
    # ax.scatter(X, Y, Z, c=Z, cmap='viridis', alpha=0.5) # 散布図:(z軸の値により色付け)
    ax.scatter(X, Y, Z, color=(np.abs(XYZ))/r, alpha=1.0, zorder=10) # 散布図:(xyz座標の値により色付け)

    end = 0.5
    ax.quiver([-r, 0, 0], [0, -r, 0], [0, 0, -r],
              [-end, 0, 0], [0, -end, 0], [0, 0, -end],
              color='black', lw=2, zorder=-1e10) # x,y,z軸
    ax.quiver([r, 0, 0], [0, r, 0], [0, 0, r],
              [end, 0, 0], [0, end, 0], [0, 0, end],
              color='black', lw=2, zorder=-1e10)
    ax.text(r+end, 0, end/2, r"$x$", fontfamily='serif', fontsize=15, fontstyle='italic')
    ax.text(end/2, r+end, 0, r"$y$", fontfamily='serif', fontsize=15, fontstyle='italic')
    ax.text(0, end/2, r+end, r"$z$", fontfamily='serif', fontsize=15, fontstyle='italic')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')
    ax.view_init(elev=v, azim=h)

update(0)
plt.show()

"""
ani = FuncAnimation(fig=fig, func=update, frames=frame_num, interval=1000)
ani.save('sphere_3d.gif')
"""