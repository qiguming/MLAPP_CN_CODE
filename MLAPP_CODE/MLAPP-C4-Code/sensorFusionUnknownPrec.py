"""在未知传感器精度的前提下，进行传感器融合"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams["figure.figsize"]=(10, 5)
# 观察到的样本
xs = np.array([1.1,1.9])
ys = np.array([2.9,4.1])
nx = xs.size
ny = ys.size
x_bar = np.mean(xs)
y_bar = np.mean(ys)
s_x = np.var(xs - x_bar)
s_y = np.var(ys - y_bar)

# 极大似然
lam_x = 1/s_x
lam_y = 1/s_y
post_prec = nx * lam_x + ny * lam_y
theta = (x_bar*nx*lam_x + y_bar*ny*lam_y)/post_prec
post_var = 1/post_prec

for iter in range(10):
    lam_x = nx/np.sum((xs-theta)**2)
    lam_y = ny/np.sum((ys-theta)**2)
    theta = (x_bar*lam_x*nx + y_bar*lam_y*ny)/(nx*lam_x + ny*lam_y)
post_var = 1/(nx*lam_x+ny*lam_y)

grid_theta = np.linspace(-2,6,100)
prob_theta = stats.norm.pdf(grid_theta, loc=theta, scale=post_var**0.5)

ax1 = plt.subplot(121)
ax1.plot(grid_theta, prob_theta, linewidth=3)
ax1.set_ylim([0,1.5])

# 贝叶斯方法
fx = (x_bar - grid_theta)**2 + s_x
fy = (y_bar - grid_theta)**2 + s_y
post = (1/fx)*(1/fy)
ax2 = plt.subplot(122)
ax2.plot(grid_theta, post, linewidth=3)
ax2.set_ylim([0,1.5])

plt.show()

