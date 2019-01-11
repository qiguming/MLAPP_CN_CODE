import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams['figure.figsize']=[8,7]

k0s = np.array([0,1,2,3])
xrange = np.arange(-1,2.5,0.05)
n = 5
theta_true = 1
sigma_true = 1
theta_prior = 0
markers = ['*','o','v','^']
colors = ['black','blue','green','red']
legends = ['postMean0','postMean1','postMean2','postMean3']

for i, ki in enumerate(k0s):
    w = n/(n+ki)
    v = (w**2)*(sigma_true**2)/n
    theta_post = w*theta_true + (1-w)*theta_prior
    plt.plot(xrange, norm.pdf(xrange, theta_post, v**0.5), color=colors[i], 
            marker=markers[i], markersize=5, linewidth=2,label=legends[i])
plt.legend()
plt.ylim([0,1.5])
plt.show()

n_list = np.arange(1,50,2)

for i, ki in enumerate(k0s):
    w = n_list/(n_list+ki)
    mse_map = ((1-w)*(theta_prior-theta_true))**2 + (w**2)*sigma_true/n_list
    mse_mean = sigma_true/n_list
    mse_ratio = mse_map/mse_mean
    plt.plot(n_list, mse_ratio, color=colors[i], 
            marker=markers[i], markersize=5, linewidth=2,label=legends[i])
plt.legend()
plt.ylim([0.5,1.3])
plt.show()




