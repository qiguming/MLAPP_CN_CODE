"""实现亚马逊购物的案例"""
from scipy import stats, interpolate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.integrate import trapz, dblquad

def kernel_estimate(samples, step_by):
    """本函数模仿seaborn教程：http://seaborn.pydata.org/tutorial/distributions.html#plotting-univariate-distributions"""
    bandwidth = 1.06 * samples.std() * samples.size ** (-1/5.)
    support = np.arange(np.min(samples), np.max(samples), step_by)
    kernels = []
    for single_sample in samples:
        kernel = stats.norm(single_sample, bandwidth).pdf(support)
        kernels.append(kernel)
    density = np.sum(kernels, axis=0)
    density /= trapz(density, support)
    pro_mass_per_int = step_by * density
    cdf = np.cumsum(pro_mass_per_int)
    return support, density, cdf

y1=90; n1=100; y2=2; n2=2
S = 100000 # 采集的样本的数量
alphas = [1, 1]
theta1 = stats.beta.rvs(y1 + alphas[0], n1-y1+alphas[1], size=S)
theta2 = stats.beta.rvs(y2 + alphas[0], n2-y2+alphas[1], size=S)
diff = theta1 - theta2
#sns.distplot(diff, hist=False, kde=True)
# 我尝试使用sklearn中的核密度估计方法，发现结果不合理
#samples = diff[:, np.newaxis]
#kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(samples)
#sort_samples = np.sort(samples)
#p_diff = kde.score_samples(sort_samples)
#p_cumsum = np.cumsum(p_diff)
#f_int = interpolate.interp1d(p_cumsum, x_s.flatten())
#q=f_int([0.025,0.5,0.975])
#plt.plot(x_s, p_diff)
#plt.vlines(q[0],0,1,linewidth=3,color='r')
#plt.vlines(q[2],0,1,linewidth=3,color='r')

##################################################################
# support, density, cdf = kernel_estimate(diff, 0.01)
# plt.plot(support, density)
# #plt.plot(support, cdf)

# fn_inter = interpolate.interp1d(cdf, support)
# alpha = [0.025, 0.975]
# p = fn_inter(alpha)

# plt.vlines(p[0],0,2.5,linewidth=3,color='r')
# plt.vlines(p[1],0,2.5,linewidth=3,color='r')
####################################################################

# 计算delta>0的比例，采样两种方式
# MC
fraction_mc = np.sum(diff>0)/len(diff)
print(fraction_mc)
## 数值积分（此过程运行时间较长，计算结果为0.713）
#fraction_int = dblquad(lambda theta1, theta2: 
#                            stats.beta.pdf(theta1,y1+1,n1-y1+1)*stats.beta.pdf(theta2,y2+1,n2-y2+1)*(theta1>theta2),
#                            0,1,0,1)
#print(fraction_int)

## 绘制图5.5a
x_support = np.linspace(0,1,100)
p1 = stats.beta(y1+1,n1-y1+1).pdf(x_support)
p2 = stats.beta(y2+1,n2-y2+1).pdf(x_support)
plt.plot(x_support,p1,'r-',linewidth=3,label=r"p($\theta_1$|data)")
plt.plot(x_support,p2,'g.',linewidth=3,label=r"p($\theta_2$|data)")
plt.legend()


plt.show()

