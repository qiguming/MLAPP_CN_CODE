"""利用MC进行分位数计算"""
from scipy import stats,interpolate
import numpy as np

mu = 0; sigma=1
S = 1000     # 样本的数量
xs = stats.norm.rvs(mu, sigma, S)
qs = [0.025, 0.5, 0.975]
q_exact = stats.norm.ppf(qs, loc=mu, scale=sigma**0.5)
print(q_exact)

ys = stats.norm.cdf(xs, loc=mu, scale=sigma**0.5)
f = interpolate.interp1d(ys, xs)
q = f(qs)
print(q)
