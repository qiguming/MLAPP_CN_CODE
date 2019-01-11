"""根据已有观察值，对函数进行插值处理"""
import numpy as np
from functools import reduce
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(1)    # 设置随机种子
D = 150              # 数据的总量（含观测和未观测到的值）
n_obs = 10           # 观测到的样本点的数量
xs = np.linspace(0, 1, D)  # 定义函数的支撑集
perm = np.random.permutation(D)   # 索引号打乱
obs_index = perm[range(10)]      #观测值的索引号
hid_index = np.array(list(set(perm)-set(obs_index)))  # 未观测值的索引号

x_obs = np.random.randn(n_obs)[:, np.newaxis]   # 生成n_obs个观测值

data = np.array([[-1]*D, [2]*D, [-1]*D])
diags = np.array([0, 1, 2])
all_matrix = spdiags(data, diags, D, D).toarray()
L = (1/2)*all_matrix[0:D-2]
print(L)

# 先验精度值lambda 仅仅影响方差
lambdas = [30, 0.01]
lambda_index = 0
L = lambdas[lambda_index]*L

L1 = L[:, hid_index]
L2 = L[:, obs_index]
laml1 = np.dot(L1.T, L1)
laml2 = np.dot(L1.T, L2)

postdist_sigma = np.linalg.inv(laml1)
postdist_mu = reduce(np.dot,(-np.linalg.inv(laml1), laml2, x_obs))


### 绘图
plt.figure()
plt.style.use('ggplot')
plt.plot(xs[hid_index], postdist_mu, linewidth=2)
plt.plot(xs[obs_index], x_obs, 'ro', markersize=12)
plt.title(r'$\lambda$={}'.format(lambdas[lambda_index]))

xbar = np.zeros(D)
xbar[hid_index] = postdist_mu.flatten()
xbar[obs_index] = x_obs.flatten()

sigma = np.zeros(D)
sigma[hid_index] = (np.diag(postdist_sigma))**0.5
sigma[obs_index] = 0

# 绘制边缘后验分布的标准误差带
plt.figure()
plt.style.use('ggplot')
f1 = xbar + 2*sigma
f2 = xbar - 2*sigma
plt.fill_between(xs, f2, f1, color=(0.8,0.8,0.8))
plt.plot(xs[hid_index], postdist_mu, linewidth=2)
plt.plot(xs[obs_index], x_obs, 'ro', markersize=12)
#plt.ylim([-5,5])
plt.title(r'$\lambda$={}'.format(lambdas[lambda_index]))
for i in range(3):
    fs = np.zeros(D)
    # for j, single_index in  enumerate(hid_index):
    fs[hid_index] = stats.multivariate_normal.rvs(postdist_mu.flatten(), postdist_sigma, 1)
    fs[obs_index] = x_obs.flatten()
    plt.plot(xs, fs,'k-',linewidth=1)   
plt.show()



