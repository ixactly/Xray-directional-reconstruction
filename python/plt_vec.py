import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D座標設定関数
def coordinate_3d(axes, range_x, range_y, range_z, grid = True):
    axes.set_xlabel("x", fontsize = 16)
    axes.set_ylabel("y", fontsize = 16)
    axes.set_zlabel("z", fontsize = 16)
    axes.set_xlim(range_x[0], range_x[1])
    axes.set_ylim(range_y[0], range_y[1])
    axes.set_zlim(range_z[0], range_z[1])
    ax.set_aspect('equal')
    if grid == True:
        axes.grid()

# 3Dベクトル描画関数
def visual_vector_3d(axes, loc, vector, color = "red"):
    axes.quiver(loc[0], loc[1], loc[2],
                vector[0], vector[1], vector[2],
                color = color, length = 1,
                arrow_length_ratio = 0.2)
# FigureとAxes
fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(111, projection='3d')

# 3D座標を設定
coordinate_3d(ax, [-2, 2], [-2, 2], [-2, 2], grid = True)

# 始点を設定
loc = [0, 0, 0]

# 3Dベクトルを配置
'''
vec_list = [[0.666667, 0.666667, -0.333333],
            [-0.333333, 0.666667, 0.666667],
            [0.666667, -0.333333, 0.666667],
            [0.57735, 0.57735, 0.57735],
            [0.19245, -0.96225, 0.19245],
            [-0.19245, -0.19245, 0.96225],
            [0.96225, -0.19245, -0.19245],
            [-0.666667, -0.666667, 0.333333],
            [0.333333, -0.666667, -0.666667],
            [-0.666667, 0.333333, -0.666667],
            [-0.57735, -0.57735, -0.57735],
            [-0.19245, 0.96225, -0.19245],
            [0.19245, 0.19245, -0.96225],
            [-0.96225, 0.19245, 0.19245]]
'''
"""
vec_list = [[1.0, 0.00026, 0.000026],
            [0.500000, -0.500000, 0.707107], 
            [-0.500000, 0.500000, 0.707107],
            [-0.707107, -0.707107, 0.000000]]
"""
vec_list = [[1.0, 0.0, 0.000026],
            [0.0, 0, 1.00],
            [0.7071, 0.7071, 0],
            [0.7071, -0.7071, 0]]
visual_vector_3d(ax, loc, vec_list[0], "pink")
visual_vector_3d(ax, loc, vec_list[1], "green")
visual_vector_3d(ax, loc, vec_list[2], "blue")
visual_vector_3d(ax, loc, vec_list[3], "red")

"""
for vec in vec_list:
    visual_vector_3d(ax, loc, vec, "red")
"""

plt.show()