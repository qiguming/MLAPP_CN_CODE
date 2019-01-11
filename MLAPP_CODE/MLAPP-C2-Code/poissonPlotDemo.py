
# coding: utf-8

# In[3]:


from scipy.stats import poisson  # 导入泊松分布的实例
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize']=(15,5)


# In[2]:


mu1,mu2=1,10  #设置相关参数，即泊松分布中的参数lambda


# In[10]:


fontdict = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
x1 = np.arange(poisson.ppf(0,mu1),
             poisson.ppf(0.99,mu1)+1)
ax1=plt.subplot(121)
#ax1.plot(x1,binom.pmf(x1,n,p1),'bo',ms=1)
ax1.vlines(x1, 0, poisson.pmf(x1,mu1), colors='b', lw=10, alpha=1)
ax1.set_xlim((-0.9,30))
ax1.set_ylim((0,0.4))
ax1.xaxis.set_major_locator(plt.MultipleLocator(5.0))
ax1.set_xlabel('(a)',fontdict=fontdict)
ax1.tick_params(labelsize=15)
ax1.set_title(r'$Poi(\lambda=1.000)$',fontdict=fontdict)

x2 = np.arange(poisson.ppf(0,mu2),
             poisson.ppf(0.99,mu2)+1)
ax2=plt.subplot(122)
#ax2.plot(x2,binom.pmf(x2,n,p2),'bo',ms=1)
ax2.vlines(x2, 0, poisson.pmf(x2, mu2), colors='b', lw=10, alpha=1)
ax2.set_xlim((-0.9,30))
ax2.set_ylim((0,0.14))
ax2.xaxis.set_major_locator(plt.MultipleLocator(5.0))
ax2.set_xlabel('(b)',fontdict=fontdict)
ax2.tick_params(labelsize=15)
ax2.set_title(r'$Poi(\lambda=10.000)$',fontdict=fontdict)

labels = ax1.get_xticklabels() + ax1.get_yticklabels()+ax2.get_xticklabels() + ax2.get_yticklabels()
temp=[label.set_fontname('Times New Roman') for label in labels]
plt.savefig('1.png')  # 将图片保存到当前目录，文件名为1.png

