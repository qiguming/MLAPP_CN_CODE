"""绘制beta分布的CI和HPD"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from HdiFromIcdf import hdi_from_icdf
plt.rcParams["figure.figsize"]=(15,7)

a = 3; b = 9; alpha = 0.05
l = stats.beta.ppf(alpha/2, a, b)
u = stats.beta.ppf(1-alpha/2, a, b)
CI = [l, u]

xs = np.linspace(0.001, 0.999, 40)
ps = stats.beta.pdf(xs, a=a, b=b)

icdf = lambda p:stats.beta.ppf(p, a=a, b=b)
HPD = hdi_from_icdf(icdf, 0.95)

Ints = [CI, HPD]

for i in range(len(Ints)):
    l = Ints[i][0]
    u = Ints[i][1]
    pl = stats.beta.pdf(l, a=a, b=b)
    pu = stats.beta.pdf(u, a=a, b=b)
    ax = plt.subplot(1,2,i+1)
    ax.plot(xs, ps, 'k-', linewidth=3)
    plt.vlines(l,0,pl,linewidth=3,color='red')
    plt.vlines(u,0,pu,linewidth=3,color='red')
    plt.plot([l,u],[pl,pu],linewidth=3, color="red")
    plt.ylim([0, 4])
print(Ints)
plt.show()
