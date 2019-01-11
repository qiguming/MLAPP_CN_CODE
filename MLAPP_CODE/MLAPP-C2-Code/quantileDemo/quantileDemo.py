
# coding: utf-8

# In[91]:


#%matplotlib inline
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,5)
#%matplotlib inline


# In[103]:


# 求取绘制cdf的数据
cdf_result=np.linspace(0,1,1000)
x=norm.ppf(cdf_result)
# 求取绘制pdf的数据
xx=np.linspace(-4,4,50)
yy=norm.pdf(x=xx,loc=0,scale=1)
# 开始绘图
fontdict = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
ax1=plt.subplot(121)
ax1.plot(x,cdf_result)
ax1.set_title('CDF',fontdict=fontdict)
ax1.set_ylim((0,1))
ax1.tick_params(labelsize=15)

ax2=plt.subplot(122)
ax2.plot(xx,yy)
ax2.set_title('PDF',fontdict=fontdict)
ax2.set_ylim((0,0.5))
ax2.set_xlim((-4,4))
ax2.tick_params(labelsize=15)

alpha=0.05
x1=norm.ppf(alpha/2)
x2=norm.ppf(1-alpha/2)
plt.xticks([x1,x2,0],[r'$\Phi^{-1}(\frac{\alpha}{2})$',r'$\Phi^{-1}(1-\frac{\alpha}{2})$',0],fontsize=18)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()+ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

ax2.fill_between(xx,yy,where=((-4<xx)&(xx<=x1))|((x2<=xx)&(xx<4)),facecolor='blue')
ax2.annotate(r'$\frac{\alpha}{2}$',xy=(x1-0.5, norm.pdf(x1-0.5)),xytext=(-40, 40),
            textcoords='offset points', fontsize=20,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
ax2.annotate(r'$\frac{\alpha}{2}$',xy=(x2+0.5, norm.pdf(x2+0.5)),xytext=(40, 40),
            textcoords='offset points', fontsize=20,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.show()