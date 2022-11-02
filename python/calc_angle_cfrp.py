import numpy as np

num_voxel = 256
with open('/home/tomokimori/CLionProjects/3dreconGPU/volume_bin/cfrp_xyz3/PCA/CF_MAIND_X_256x256x256.raw') as f:
    x_raw = np.fromfile(f, dtype=np.float32).reshape([num_voxel, num_voxel, num_voxel])

with open('/home/tomokimori/CLionProjects/3dreconGPU/volume_bin/cfrp_xyz3/PCA/CF_MAIND_Y_256x256x256.raw') as f:
    y_raw = np.fromfile(f, dtype=np.float32).reshape([num_voxel, num_voxel, num_voxel])

with open('/home/tomokimori/CLionProjects/3dreconGPU/volume_bin/cfrp_xyz3/PCA/CF_MAIND_Z_256x256x256.raw') as f:
    z_raw = np.fromfile(f, dtype=np.float32).reshape([num_voxel, num_voxel, num_voxel])

x_trim = x_raw[135:155, 120:140, 130:150]
y_trim = y_raw[135:155, 120:140, 130:150]
z_trim = z_raw[135:155, 120:140, 130:150]

angle_yz = np.arctan2(np.abs(y_trim), np.abs(z_trim))
angle_yx = np.arctan2(np.abs(y_trim), np.abs(x_trim))

print(np.mean(np.degrees(angle_yz)))
print(np.mean(np.degrees(angle_yx)))

print(y_trim)
