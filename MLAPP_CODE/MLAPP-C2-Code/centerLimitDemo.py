
# coding: utf-8

# In[10]:


from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,5)


# In[12]:


# 从beta(1,5)中采集样本
x1=beta.rvs(a=1,b=5,size=10000)
bins=np.linspace(0,1,100)
ax1 = plt.subplot(121)
result=ax1.hist(x1,bins,align='left',rwidth=0.7)
ax1.set_xlabel('(a)')

N=5   # 采集5次
xx = np.empty((N,10000))
for i in range(N):
    xx[i,:]=beta.rvs(a=1,b=5,size=10000)
x2 = np.sum(xx,axis=0)/N
ax2 = plt.subplot(122)
result2=ax2.hist(x2,bins,align='left',rwidth=0.7)
ax2.set_xlabel('(b)')

