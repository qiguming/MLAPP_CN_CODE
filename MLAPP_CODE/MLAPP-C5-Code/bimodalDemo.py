import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
mu = np.array([0,2])
sigma = np.array([1, 0.05])
w = np.array([0.5, 0.5])
xs = np.arange(-2, mu[1]*2, 0.01)
p = w[0] * stats.norm.pdf(xs, mu[0], sigma[0]) + w[1] * stats.norm.pdf(xs, mu[1], sigma[1])
plt.plot(xs, p, 'k-', linewidth=3)
_mu = np.mean(xs*p)
plt.vlines(_mu, 0, np.max(p), color='red', linewidth=3)
plt.ylim([0,5])
plt.show()