import numpy as np
from scipy import stats
from Gaussian2D import gaussian_2d
import matplotlib.pyplot as plt
from GaussCondition import gauss_condition
plt.rcParams['figure.figsize']=(16, 5)

# 绘制2维高斯分布
mu = np.array([0, 0])
rho = 0.8    # 相关系数
sigma = np.array([[1, rho],[rho, 1]])
ax1 = plt.subplot(131)
z_countor = gaussian_2d(mu, sigma)
ax1.plot(z_countor[0, :], z_countor[1, :], linewidth=2)
D, U = np.linalg.eig(sigma)
sf = -2.5
ax1.plot([mu[0], mu[0]+sf*np.power(D[0],0.5)*U[0,0]],[mu[1], mu[1]+sf*np.power(D[0],0.5)*U[1,0]])
ax1.plot([mu[0], mu[0]+sf*np.power(D[1],0.5)*U[0,1]],[mu[1], mu[1]+sf*np.power(D[1],0.5)*U[1,1]])
x2 = 1
ax1.plot([-5, 5],[x2, x2], color='r')
ax1.set_xlim((-5, 5))
ax1.set_ylim((-3, 3))
plt.axis('equal')

# 绘制无条件边缘分布
ax2 = plt.subplot(132)
marg_mu = mu[1]; marg_sigm = sigma[0,0]
xs = np.arange(-5, 5, 0.2)
ps = stats.norm.pdf(xs,marg_mu,marg_sigm)
ax2.plot(xs, ps,'k-', linewidth = 3)
ax2.set_ylim((0, 0.5))

# 绘制条件概率分布
ax3 = plt.subplot(133)
class Model:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
model = Model(mu, sigma=sigma)

mu_v, sigma_v = gauss_condition(model, 1, x2)
ps = stats.norm.pdf(xs, mu_v, sigma_v)
ax3.plot(xs, ps.flatten(), 'k-', linewidth=3)
ax3.set_ylim((0, 1.5))
#plt.axis('square')
plt.show()


