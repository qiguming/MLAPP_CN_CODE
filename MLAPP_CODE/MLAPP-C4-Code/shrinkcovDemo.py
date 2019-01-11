"""图4.17.对比协方差矩阵的极大似然解与最大后验估计的优劣势"""

import numpy as np
from CovCond import cov_cond
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)

dim = 50
cond_number = 10
a = np.random.randn(dim)[:, np.newaxis]
sigma, _ = cov_cond(cond_number, a)
evals_true = -np.sort(-np.linalg.eigvals(sigma)) # 按照降序的方式获取真实的协方差矩阵的特征值

mu = np.zeros(dim)
f = [2,1,0.5]


for i in range(len(f)):
    n_sampels = f[i]*dim
    X = multivariate_normal.rvs(mu, sigma, int(n_sampels))
    s_mle = np.cov(X, rowvar=False)
    evals_mle = -1*np.sort(-1*np.linalg.eigvals(s_mle))
    cond_number_mle = np.linalg.cond(s_mle, 2)

    _lambda = 0.9
    s_shrink = _lambda*np.diag(np.diag(s_mle)) + (1 - _lambda)*s_mle
    evals_shrink = -1*np.sort(-1*np.linalg.eigvals(s_shrink))
    cond_number_shrink = np.linalg.cond(s_shrink, 2)

    ax = plt.subplot(1, 3, i+1)
    ax.plot(evals_true, 'k-', linewidth=3, label="true,k={}".format(cond_number) )
    ax.plot(evals_mle, 'b:', linewidth=3, label="mle,k={:.3}".format(cond_number_mle))
    ax.plot(evals_shrink, 'r-.', linewidth=3, label="shrink,k={:.3}".format(cond_number_shrink))
    plt.legend()
plt.show()



