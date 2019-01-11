
# coding: utf-8

# In[10]:


from scipy.stats import binom  # 导入伯努利分布的实例
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize']=(15,5)


# In[49]:


n,p1,p2=10,0.25,0.9   #设置相关参数


# In[57]:


fontdict = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
x1 = np.arange(binom.ppf(0,n,p1),
             binom.ppf(1,n,p1)+1)
ax1=plt.subplot(121)
#ax1.plot(x1,binom.pmf(x1,n,p1),'bo',ms=1)
ax1.vlines(x1, 0, binom.pmf(x1, n, p), colors='b', lw=20, alpha=1)
ax1.set_xlim((-0.9,10.9))
ax1.set_ylim((0,0.35))
ax1.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax1.set_xlabel('(a)',fontdict=fontdict)
ax1.tick_params(labelsize=15)
ax1.set_title(r'$\theta=0.250$',fontdict=fontdict)

x2 = np.arange(binom.ppf(0,n,p2),
             binom.ppf(1,n,p2)+1)
ax2=plt.subplot(122)
#ax2.plot(x2,binom.pmf(x2,n,p2),'bo',ms=1)
ax2.vlines(x2, 0, binom.pmf(x2, n, p2), colors='b', lw=20, alpha=1)
ax2.set_xlim((-0.9,10.9))
ax2.set_ylim((0,0.4))
ax2.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax2.set_xlabel('(b)',fontdict=fontdict)
ax2.tick_params(labelsize=15)
ax2.set_title(r'$\theta=0.900$',fontdict=fontdict)

labels = ax1.get_xticklabels() + ax1.get_yticklabels()+ax2.get_xticklabels() + ax2.get_yticklabels()
temp=[label.set_fontname('Times New Roman') for label in labels]

