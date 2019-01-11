"""实现含噪数据的函数插值"""
import numpy as np
from scipy.sparse import spdiags
from functools import reduce
import matplotlib.pyplot as plt
from scipy import stats
D = 150                           # 支撑集中共有D个点
N_OBS = 10                        # 观测值的数目
X_S = np.linspace(0, 1, D)        # 支撑集
PERM = np.random.permutation(D)   # 随机打乱的顺序
OBS_INDEX = PERM[0:N_OBS]         # 观测值对应的索引号
HID_INDEX = np.setdiff1d(PERM, OBS_INDEX)  # 隐藏值对应的索引号

# 构造含噪的观测值
obs_noise_var = 1                                    # 观测值的噪音方差
y = (obs_noise_var**0.5)*np.random.randn(N_OBS, 1)     # 观察到的含噪数据
A = np.zeros((N_OBS, D))
for i in range(N_OBS):
    A[i, OBS_INDEX[i]] = 1

# 构建(D-2)*(D)的三对角矩阵
data = np.array([[-1]*D, [2]*D, [-1]*D])
diags = np.array([0, 1, 2])
all_matrix = spdiags(data, diags, D, D).toarray()
L = (1/2)*all_matrix[0:D-2]

# 先验精度值lambda 仅仅影响方差
lambdas = [30, 0.01]
lambda_index = 1
L = lambdas[lambda_index]*L

prec_mat = np.dot(L.T, L) + 1e-3
prior_mu = np.zeros((D, 1))
prior_sigma = np.linalg.inv(prec_mat)

obs_sigma = obs_noise_var*np.eye(N_OBS)
obs_prec = np.linalg.inv(obs_sigma)

post_sigma = np.linalg.inv(prec_mat + reduce(np.dot, (A.T, obs_prec, A)) )
post_mu = reduce(np.dot, (post_sigma, A.T, obs_prec, y))

# 绘制图像
plt.plot(X_S[OBS_INDEX], y.flatten(), 'kx', markersize=14, linewidth=3)
plt.plot(X_S, post_mu.flatten(), 'k-', linewidth=2)
plt.title(r'$\lambda$={}'.format(lambdas[lambda_index]))

# 绘制边缘后验分布的标准差
post_diag_sigma = np.diag(post_sigma)
f1 = post_mu.flatten() + 2*(post_diag_sigma**0.5)
f2 = post_mu.flatten() - 2*(post_diag_sigma**0.5)

plt.fill_between(X_S, f1, f2, color=(0.8, 0.8, 0.8))

# 绘制采样函数曲线
for i in range(3):
    fs = stats.multivariate_normal.rvs(post_mu.flatten(), post_sigma, 1)
    plt.plot(X_S, fs.flatten(), 'k-', linewidth=1)

plt.xlim([0,1])

plt.show()