import numpy as np
from unidrnd import unid_rnd
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams["figure.figsize"]=(10,15)
#  真实的模型参数值
theta = 0.7
#  采样的样本数量
n_samples = [10, 100]
x_label = ['(a)','(b)','(c)','(d)']
for index,n_sample in enumerate(n_samples):
    B = 10000                # 重复试验的次数
    X = np.random.rand(n_sample) < theta
    estimator = lambda X: np.mean(X)
    bmle = estimator(X)
    mleBoot = np.zeros(B)
    mleBootNP = np.zeros(B)
    for b in range(B):
        Xb = np.random.rand(n_sample) < bmle        # 有参数采样
        mleBoot[b] = estimator(Xb)                       
        ndx = unid_rnd(n_sample, n_sample)          # 无参数采样
        Xnonparam = X[ndx]
        mleBootNP[b] = estimator(Xnonparam)
    
    ax1 = plt.subplot(2,2,index+1)
    ax1.hist(mleBoot, density=True)
    ax1.set_title('Boot:true={},n={},mle={},se={:.2f}'.format(theta,n_sample,bmle,np.std(mleBoot)))
    ax1.set_xlabel(x_label[index])
    #ax2 = plt.subplot(122)
    #ax2.hist(mleBootNP, density=True)

    # 后验分布
    N1 = np.sum(X == 1)
    N0 = np.sum(X == 0)
    alpha1 =1 ; alpha0 = 0
    a = N1 + alpha1
    b = N0 + alpha0
    X_post = stats.beta.rvs(a=a,b=b,size=B)
    ax2 = plt.subplot(2,2,index+3)
    ax2.hist(X_post, density=True)
    ax2.set_title('Bayes:true={},n={},post_mean={:.2f},se={:.2f}'.format(theta,n_sample,np.mean(X_post),np.std(X_post)))
    ax2.set_xlabel(x_label[index+2])

plt.show()