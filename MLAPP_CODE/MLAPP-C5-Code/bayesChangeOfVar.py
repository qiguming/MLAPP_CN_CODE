import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def ginv(x):
    y = 1/(1+np.exp(-1*x + 5))
    return y

mu =6; sigma=1
n_samples=1e6
x = stats.norm.rvs(loc=mu, scale=sigma**0.5, size=int(n_samples))

y = ginv(x)

plt.hist(x, bins=50, align='mid', orientation='vertical', rwidth=0.8,density=True, color='red')
plt.hist(y, bins=50, align='mid', orientation='horizontal', rwidth=0.8,density=True, color='green')

x_s = np.linspace(np.min(x), np.max(x), 100)
y_s = ginv(x_s)
plt.plot(x_s, y_s, color='blue', linestyle='-', linewidth=3)

plt.vlines(mu, 0, ginv(mu), color='black', linewidth=3)
plt.hlines(ginv(mu), 0, mu, color='black', linewidth=3)
plt.show()