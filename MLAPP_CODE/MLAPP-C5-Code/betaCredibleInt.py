"""计算分位数"""
from scipy import stats
import numpy as np

S = 47
N = 100
a = S + 1
b = (N -S) + 1
alpha = 0.05

lu = stats.beta.ppf([alpha/2, 1-alpha/2], a, b)
print(lu)

## MC方法
S = 1000
X = stats.beta.rvs(a, b, size=S)
X = np.sort(X, axis=0)
l = X[round(S*alpha/2)]
u = X[round(S*(1-alpha)/2)]
print(l,u)