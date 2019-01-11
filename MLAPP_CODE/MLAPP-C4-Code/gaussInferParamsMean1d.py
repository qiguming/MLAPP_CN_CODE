import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams["figure.figsize"]=(15,7)
prior_var_list = [1, 5]   # 先验分布的方差
sigma = 1            # 似然函数的方差，假设已知
X = [3]                # 观察到的数据
for i in range(len(prior_var_list)):
    ## 先验分布
    prior_var = prior_var_list[i]
    prior_mu = 0

    x_bar = np.mean(X)
    n = len(X)

    lik_sigma = sigma     # 似然函数的方差（已知）
    lik_mu = x_bar        # 极大似然估计

    s_0 = prior_var       # 先验分布的方差
    s0_inv = 1/s_0        # 先验分布的精度
    mu_0 = prior_mu       # 先验分布的期望
    s = sigma             # 似然函数的方差
    s_inv = 1/s           # 似然函数的精度
    s_n = 1/(s0_inv + n*s_inv)  # 后验分布的方差

    post_mu = s_n*(s0_inv*mu_0 + n*x_bar*s_inv)  # 后验分布的期望
    post_sigma = s_n                             # 后验分布的方差

    ax = plt.subplot(1,2,i+1)
    x_range = np.arange(-5,5,0.25)

    ax.plot(x_range, stats.norm.pdf(x_range, loc=mu_0, scale=s_0 ** 0.5), color='blue', linewidth=2, label="prior")
    ax.plot(x_range, stats.norm.pdf(x_range, loc=lik_mu, scale=lik_sigma ** 0.5), color='red', linewidth=2, label="lik")
    ax.plot(x_range, stats.norm.pdf(x_range, loc=post_mu, scale=post_sigma ** 0.5), color='black', linewidth=2, label='post')
    ax.set_ylim([0, 0.6])
    ax.set_xlim([-5,5])
    plt.legend()
plt.show()






