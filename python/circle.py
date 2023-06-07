import numpy as np
import matplotlib.pyplot as plt

# Figureを追加
fig = plt.figure(figsize=(12, 12))
# 3DAxesを追
ax = fig.add_subplot(111, projection='3d')

# 軸ラベルを設定
ax.set_xlabel("x", size=14)
ax.set_ylabel("y", size=14)
ax.set_zlabel("z", size=14)
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_aspect('equal')

# パラメータ分割数
n = 128


# 回転行列 R
# phi = [0, np.pi / 4.0, np.pi / 2.0, np.pi * 3 / 4.0, np.pi]


def ax_plot_circle(phi_list, theta_list, graph, col):
    for ph in phi_list:
        for th in theta_list:
            Rz = np.array([[np.cos(ph), -np.sin(ph), 0], [np.sin(ph), np.cos(ph), 0], [0, 0, 1]])
            Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
            R = Rz @ Ry
            # パラメータtを作成
            t = np.linspace(0, 2 * np.pi, n)

            # らせんの方程式
            x = np.cos(t)
            y = np.sin(t)
            z = np.zeros(np.size(t))

            xyz = np.vstack([x, y, z])
            xyz = R @ xyz

            x_rot = xyz[0, :]
            y_rot = xyz[1, :]
            z_rot = xyz[2, :]

            # 曲線を描画
            graph.plot(x_rot, y_rot, z_rot, color=col)

phi = [0]
theta = [0]
ax_plot_circle(phi, theta, ax, "red")

phi = [0]
theta = [np.pi / 2.0]
ax_plot_circle(phi, theta, ax, "green")

phi = [np.pi / 2.0]
theta = [np.pi / 2.0]
ax_plot_circle(phi, theta, ax, "blue")

phi = [0]
theta = [np.pi / 4.0, np.pi * 3 / 4.0]
ax_plot_circle(phi, theta, ax, "red")

phi = [np.pi / 4.0, 3 * np.pi / 4.0]
theta = [np.pi / 2.0]
ax_plot_circle(phi, theta, ax, "green")

phi = [np.pi / 2.0]
theta = [np.pi / 4.0, np.pi * 3 / 4.0]
ax_plot_circle(phi, theta, ax, "blue")

# conventional
"""
vert = np.arccos(1 / np.sqrt(3))
theta = [vert]
phi = [np.pi / 4.0, 3 * np.pi / 4.0, 5 * np.pi / 4.0, 7 * np.pi / 4.0]
ax_plot_circle(phi, theta, ax, "pink")
"""
plt.show()
