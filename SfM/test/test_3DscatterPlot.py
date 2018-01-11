import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

#mat1 = '../data/4a.mat'
#data = sio.loadmat(mat1)
#m = data['data']
#ax.scatter(x[:1000], y[:1000], z[:1000], c='y')
#ax.scatter(x[1000:4000], y[1000:4000], z[1000:4000], c = 'r')
#ax.scatter(x[4000:], y[4000:], z[4000:], c = 'g')

data = np.load('../data/capsule_point_cloud_1_11.npy')
sio.savemat('../data/capsule_point_cloud_1_11.mat', {'data' : data})
m = sio.loadmat('../data/capsule_structure_cloud_1_11.mat')
m = m['data']
print(m.shape)

path = sio.loadmat('../data/path_1_11.mat')['data']
print(path.shape)

data = list(m.transpose())
A = []
for d in data:
    if math.fabs(d[0]) > 50 or math.fabs(d[1])>50 or math.fabs(d[2])>200:
        continue
    else:
        A.append(d)

data = np.array(A).transpose()
print(data.shape)

x, y, z = data[0], data[1], data[2]
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, 'b')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

fig1 = plt.figure()
ax1=fig.add_subplot(111, projection='3d')
for i in range(path.shape[0]):
    ax1.scatter(path[i][0][0], path[i][0][1], path[i][0][2])



