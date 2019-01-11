import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,4)

mu_true = np.array([0.5, 0.5])
sigma_true = 0.1*np.array([[2,1],[1,1]])
n_sample = 10
x_sample = stats.multivariate_normal.rvs(mu_true, sigma_true, n_sample)

# 绘制采样数据和真实数据
ax1 = plt.subplot(131)
ax1.plot(x_sample[:,0], x_sample[:,1], 'bo', markersize=8)
plt.title('data')
ax1.plot(mu_true[0], mu_true[1], marker='x', color='black', markersize=15)
ax1.set_xlim([-1,1])
ax1.set_ylim([-1,1])


# 绘制先验分布
ax2 = plt.subplot(132)
prior_mu = np.array([0,0])
prior_sigma = 0.1*np.eye(2)

xy_range = np.linspace(-1,1,100)
X,Y=np.meshgrid(xy_range, xy_range)
XY = np.dstack((X, Y)).reshape((-1, 2))
Z = stats.multivariate_normal.pdf(XY, prior_mu, prior_sigma).reshape(X.shape)
ax2.contour(X, Y, Z)
plt.title('prior')
ax2.set_xlim([-1,1])
ax2.set_ylim([-1,1])

# 更新协方差矩阵
s_0 = prior_sigma
s0_inv = np.linalg.inv(s_0)
s = sigma_true
s_inv = np.linalg.inv(s)
s_n = np.linalg.inv(s0_inv + n_sample*s_inv)

# 更新期望
mu_0 = prior_mu
x_bar = np.mean(x_sample, 0)[:, np.newaxis]
mu_n = np.dot(s_n, n_sample*s_inv.dot(x_bar) + s0_inv.dot(mu_0[:, np.newaxis]))

# 绘制后验分布
post_mu = mu_n.flatten()
post_sigma = s_n
ax3 = plt.subplot(133)
post_z = stats.multivariate_normal.pdf(XY, post_mu, post_sigma).reshape(X.shape)
ax3.contour(X, Y, post_z)
plt.title('post after 10 obs')
ax3.set_xlim([-1,1])
ax3.set_ylim([-1,1])

plt.show()