"""实现在1维空间中，对方差的序列化更新，其中分布期望已知"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#np.random.seed(1)

# 先验分布（逆威舍特分布）
prior_dof = 0.001
prior_sigma = 0.001

# 真实的模型
true_mu = 5
true_sigma = 10

# 生成的随机样本
n = 500
X = stats.norm.rvs(loc=true_mu, scale=true_sigma**0.5, size=n)

ns = [2, 5, 50, 100]
color = ['black', 'blue', 'red', 'green']
for i in range(len(ns)):
    n_samples = ns[i]
    data = X[0:n_samples]
    v_0 = prior_dof
    T_0 = prior_sigma
    # 进行更新
    v_n = v_0 + n_samples
    x_bar = np.mean(data)
    data = data - x_bar
    if n > 0:
        # T_n = T_0 + data.dot(data) + n_samples*((x_bar-true_mu)**2)
        T_n = T_0 + data.dot(data)
    else:
        T_n = T_0
    post_dof = v_n
    post_sigma = T_n

    xs = np.arange(0.1,15,0.1)
    pdf_xs = stats.invwishart.pdf(xs, post_dof, post_sigma)
    plt.plot(xs, pdf_xs, color=color[i], linewidth=3, label="N={}".format(n_samples))
plt.legend()
plt.ylim([0, 0.35])
plt.show()

