"""绘制标量正态逆卡方分布"""
import numpy as np
import matplotlib.pylab as plt
from gaussInvWishart import gauss_inv_wishart_log_prob
from mpl_toolkits.mplot3d import axes3d
plt.rcParams["figure.figsize"]=(21,6)
mu = [0, 0, 0]
k = [1, 5, 1]
v = [1, 1, 5]
s = [1, 1, 1]

n_points = 100
fig = plt.figure()
for m in range(len(mu)):
    x_pos = np.linspace(-0.9 , 1, n_points)
    y_pos = np.linspace(0.1 , 2, n_points)
    x, y = np.meshgrid(x_pos, y_pos)
    xy = np.dstack((x, y)).reshape((-1, 2))
    z = gauss_inv_wishart_log_prob(mu[m], k[m], v[m], s[m], xy[:,0], xy[:,1])
    z_reshape = np.exp(z.reshape(x.shape))


    ax = fig.add_subplot(1,3,m+1,projection='3d')
    ax.plot_surface(x, y, z_reshape, rstride=2, cstride=2, alpha=1, cmap=plt.get_cmap('winter') )
    cset = ax.contour(x, y, z_reshape, zdir='z', offset=0, cmap=plt.get_cmap('hot'))
    cset = ax.contour(x, y, z_reshape, zdir='x', offset=-1, cmap=plt.get_cmap('hot'))
    cset = ax.contour(x, y, z_reshape, zdir='y', offset=2, cmap=plt.get_cmap('hot'))
    ax.set_xlabel(r'$\mu$')
    ax.set_xlim(-1, 1)
    ax.set_ylabel(r'$\theta^2$')
    ax.set_ylim(0, 2)
    ax.set_zlabel('Z')
    #ax.set_zlim(0, 0.5)
plt.show()


