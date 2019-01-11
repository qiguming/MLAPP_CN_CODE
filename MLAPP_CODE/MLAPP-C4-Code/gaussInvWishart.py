from scipy import stats
import numpy as np
def gauss_inv_wishart_log_prob(mu, k, dof, sigma, m, s):
    """计算高斯逆卡方分布的对数概率
    Input:
    mu: 高斯分布的先验期望
    k: 高斯分布的置信强度
    dof: 逆卡方分布的自由度
    sigma: 逆卡方分布的先验期望
    m: 需要计算的期望值的坐标， shape=(n_points,)
    s: 需要计算的方差值的坐标, shape=(n_points,)
    """
    n_points = len(m)
    log_prob_gauss = np.zeros(n_points)
    for i in range(n_points):
        gauss_sigma = s[i]/k     # 高斯分布的方差
        log_prob_gauss[i] = stats.norm.logpdf(m[i], loc=mu, scale=gauss_sigma**0.5)
    
    log_prob_iw = stats.invwishart.logpdf(s, dof, sigma)

    log_p = log_prob_gauss + log_prob_iw
    return log_p

