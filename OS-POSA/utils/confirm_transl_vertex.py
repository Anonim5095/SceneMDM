import os
import glob
import pickle
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pickle_dir  = "./save/debug_Werkraum_default/pkl/Werkraum"
obj_dir     = "./save/debug_Werkraum_default/meshes/Werkraum"

# pkl_trans = []
# for pkl_path in glob.glob(pickle_dir + "/*.pkl"):
#     bdata = pickle.load(open(pkl_path, 'rb'))
#     pkl_trans.append(bdata['transl'][0])
# pkl_trans = np.array(pkl_trans)

# obj_trans = []
# for obj_path in glob.glob(obj_dir + "/*.obj"):
#     mesh = trimesh.load_mesh(obj_path)
#     obj_trans.append(np.mean(mesh.vertices, axis=0))
# obj_trans = np.array(obj_trans)

# # 可視化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pkl_trans[:,0], pkl_trans[:,1], pkl_trans[:,2], c='r', label='pkl')
# ax.scatter(obj_trans[:,0], obj_trans[:,1], obj_trans[:,2], c='b', label='obj')
# for i, txt in enumerate(range(14)):
#     ax.text(pkl_trans[i,0], pkl_trans[i,1], pkl_trans[i,2], str(txt), color='black')
#     ax.text(obj_trans[i,0], obj_trans[i,1], obj_trans[i,2], str(txt), color='black')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.legend()
# plt.show()



index = 3
pkl_path = glob.glob(pickle_dir + "/*.pkl")[index]
obj_path = glob.glob(obj_dir + "/*.obj")[index]

trans = pickle.load(open(pkl_path, 'rb'))['transl'][0]
mesh = trimesh.load_mesh(obj_path)
points = mesh.vertices

# 可視化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.', label='Point Cloud')
ax.scatter(trans[0], trans[1], trans[2], c='r', s=100, label='Point A')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.legend()
plt.show()

print()


